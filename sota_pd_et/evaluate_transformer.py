"""Evaluate a frozen pretrained transformer with participant-level nested CV.

Only stateless, externally pretrained embeddings are shared between folds.
StandardScaler and the class-balanced ridge classifier are fitted separately
inside each inner/outer training fold. The saved MultiROCKET-HYDRA assignments
must match exactly before a paired, descriptive comparison is produced.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import DEFAULT_DATA_ROOT, LABELS, PROJECT_ROOT, load_cohort, sha256_file
from .evaluate import (
    PERFORMANCE_METRICS, choose_alpha, derive_seed, describe_values,
    make_inner_splits, make_plan, metrics, prediction_rows, validate_split,
    write_csv, write_json,
)
from .model import fit_ridge
from .transformer import EncoderConfig, extract_features


DEFAULT_BASELINE = PROJECT_ROOT / "artifacts" / "pads_mr_hydra"


def read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_baseline(directory: Path, cohort, plan, alphas=None):
    """Check identities, raw-data hashes and every outer/inner assignment."""
    directory = directory.expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("The baseline manifest must describe a complete run.")
    verified_alphas = None
    if alphas is not None:
        expected = np.asarray(alphas, dtype=float)
        recorded = np.asarray(manifest.get("config", {}).get("alphas", []), dtype=float)
        if expected.ndim != 1 or not np.array_equal(recorded, expected):
            raise ValueError("Baseline ridge alpha candidates differ from this run.")
        verified_alphas = expected.tolist()
    baseline_plan = json.loads((directory / "splits.json").read_text())
    if baseline_plan != plan:
        raise ValueError("Baseline outer/inner participant assignments differ from this run.")
    baseline_data = manifest.get("dataset", {})
    for key in (
        "dataset", "checksum_list_sha256", "verified_source_sha256", "label_mapping",
        "participant_count", "condition_counts", "movement_shape", "channel_names",
    ):
        if key not in cohort.provenance or baseline_data.get(key) != cohort.provenance[key]:
            raise ValueError(f"Baseline dataset provenance differs: {key}.")
    expected_cohort = [(str(pid), int(target))
                       for pid, target in zip(cohort.ids, cohort.y, strict=True)]
    baseline_cohort = [(r["participant_id"], int(r["target_et"]))
                       for r in read_csv(directory / "cohort.csv")]
    if baseline_cohort != expected_cohort:
        raise ValueError("Baseline cohort IDs, target labels or row order differ.")
    targets = dict(expected_cohort)
    expected_keys = {(s["repeat"], s["fold"], pid)
                     for s in plan for pid in s["test_ids"]}
    predictions = {}
    for row in read_csv(directory / "oof_predictions.csv"):
        key = (int(row["repeat"]), int(row["fold"]), row["participant_id"])
        if key in predictions:
            raise ValueError("Duplicate baseline outer prediction for a participant.")
        score = float(row["decision_score_et"])
        target = int(row["target_et"])
        if key not in expected_keys or targets.get(key[2]) != target:
            raise ValueError("Baseline OOF fold, participant ID or target differs.")
        if not np.isfinite(score) or int(row["predicted_et"]) != int(score > 0):
            raise ValueError("Baseline OOF score/prediction is nonfinite or inconsistent.")
        predictions[key] = {"target_et": target, "decision_score_et": score,
                            "predicted_et": int(score > 0)}
    if set(predictions) != expected_keys:
        raise ValueError("Baseline OOF predictions do not cover every expected participant.")
    return predictions, {
        "directory": str(directory),
        "validation": "Exact dataset hashes, ordered cohort, outer/inner splits and OOF targets verified.",
        "verified_alpha_candidates": verified_alphas,
        "artifact_sha256": {name: sha256_file(directory / name) for name in (
            "manifest.json", "cohort.csv", "splits.json", "oof_predictions.csv", "summary.json",
        )},
    }


def scaled_features(Z, train, valid):
    """The scaler sees only train; float64 stabilizes high-dimensional ridge."""
    scaler = StandardScaler()
    training = np.ascontiguousarray(scaler.fit_transform(Z[train]), dtype=np.float64)
    validation = np.ascontiguousarray(scaler.transform(Z[valid]), dtype=np.float64)
    if not np.isfinite(training).all() or not np.isfinite(validation).all():
        raise ValueError("Nonfinite scaled embeddings.")
    return scaler, training, validation


def tune_alpha(Z, y, alphas, inner_splits):
    """Reuse frozen embeddings, never a fitted scaler, between inner folds."""
    oof = np.full((len(alphas), len(y)), np.nan)
    fold_ba = np.empty((len(alphas), len(inner_splits)))
    for fold, (train, valid) in enumerate(inner_splits):
        validate_split(train, valid, y)
        _, training, validation = scaled_features(Z, train, valid)
        for index, alpha in enumerate(alphas):
            classifier = fit_ridge(training, y[train], alpha)
            scores = classifier.decision_function(validation)
            oof[index, valid] = scores
            fold_ba[index, fold] = balanced_accuracy_score(y[valid], scores > 0)
    if not np.isfinite(oof).all():
        raise RuntimeError("Missing/nonfinite inner out-of-fold scores.")
    search = [{
        "alpha": float(alpha),
        "inner_mean_balanced_accuracy": float(np.mean(fold_ba[index])),
        "inner_std_balanced_accuracy": float(np.std(fold_ba[index], ddof=1)),
        "inner_pooled_balanced_accuracy": float(balanced_accuracy_score(y, oof[index] > 0)),
        "inner_pooled_roc_auc": float(roc_auc_score(y, oof[index])),
    } for index, alpha in enumerate(alphas)]
    selected = choose_alpha(search)
    return selected, search, oof[list(alphas).index(selected)]


def paired_comparison(rows, baseline):
    """Recompute paired metrics from aligned held-out participant predictions."""
    paired = []
    for row in rows:
        key = (row["repeat"], row["fold"], row["participant_id"])
        reference = baseline[key]
        paired.append({
            "repeat": row["repeat"], "fold": row["fold"],
            "participant_id": row["participant_id"], "target_et": row["target_et"],
            "baseline_decision_score_et": reference["decision_score_et"],
            "transformer_decision_score_et": row["decision_score_et"],
            "baseline_predicted_et": reference["predicted_et"],
            "transformer_predicted_et": row["predicted_et"],
        })
    comparisons = []
    for repeat in sorted({row["repeat"] for row in paired}):
        selected = [row for row in paired if row["repeat"] == repeat]
        y = np.array([r["target_et"] for r in selected])
        a = np.array([r["baseline_decision_score_et"] for r in selected])
        b = np.array([r["transformer_decision_score_et"] for r in selected])
        before, after = metrics(y, a), metrics(y, b)
        a_correct, b_correct = (a > 0) == y, (b > 0) == y
        comparisons.append({
            "repeat": repeat, "baseline": before, "transformer": after,
            "transformer_minus_baseline": {key: after[key] - before[key]
                                           for key in PERFORMANCE_METRICS},
            "paired_correctness_counts": {
                "both_correct": int(np.sum(a_correct & b_correct)),
                "transformer_only_correct": int(np.sum(~a_correct & b_correct)),
                "baseline_only_correct": int(np.sum(a_correct & ~b_correct)),
                "both_wrong": int(np.sum(~a_correct & ~b_correct)),
            },
        })
    return paired, {
        "repeat_comparisons": comparisons,
        "interpretation": [
            "A paired descriptive comparison on the same PADS cohort and participant splits.",
            "This additional model was tried after inspecting the baseline; results are exploratory.",
            "No statistical significance, external clinical validation or universal model ranking is established.",
            "Pooled AUROC/AP compare scores from different fitted heads; inspect fold metrics too.",
        ],
    }


def installed_versions():
    packages = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "joblib", "threadpoolctl",
                    "huggingface-hub", "safetensors", "mantis-tsfm"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--alphas", type=float, nargs="+", default=np.logspace(-3, 3, 7).tolist())
    parser.add_argument("--fit-final", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be an integer in [0, 2**32-1].")
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive integers.")
    if not args.alphas or any(not np.isfinite(a) or a <= 0 for a in args.alphas):
        parser.error("All ridge alpha candidates must be finite and strictly positive.")
    args.alphas = sorted(set(args.alphas))
    return args


def run(args):
    started = time.perf_counter()
    cohort = load_cohort(args.data_root)
    X, y, ids = cohort.X, cohort.y, cohort.ids
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or PROJECT_ROOT / "artifacts" / "pd_et_transformer" / stamp).expanduser().resolve()
    for protected in (args.data_root.expanduser().resolve(), Path(__file__).parent.resolve(),
                      args.baseline.expanduser().resolve()):
        if output.is_relative_to(protected):
            raise ValueError(f"Choose output outside the protected dataset, source and baseline directories: {protected}")
    output.mkdir(parents=True, exist_ok=False)
    config = EncoderConfig(batch_size=args.batch_size, device=args.device)
    manifest = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "python": sys.version, "platform": platform.platform(),
        "package_versions": installed_versions(),
        "config": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "output": str(output), "encoder": asdict(config)},
        "source_code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
                               for p in sorted(Path(__file__).parent.rglob("*.py"))},
        "dataset": cohort.provenance,
        "protocol": {
            "unit": "One participant; all tasks, wrists, sensors, axes and internal windows stay together.",
            "encoder": "Externally pretrained, frozen, in evaluation mode; no PADS adaptation.",
            "cached_features": "Only fixed, per-participant embeddings are cached across folds.",
            "fitted_preprocessing": "StandardScaler fitted separately inside each inner/outer training fold.",
            "alpha_selection": "Maximum mean inner-fold balanced accuracy; ties choose larger alpha.",
            "class_weight": "n_train / (2 * class_count), recomputed within each ridge fit.",
            "decision_rule": "ET if ridge margin > 0; PD otherwise. Scores are not probabilities.",
            "comparison": "Same saved baseline cohort and splits; exploratory paired comparison.",
        },
    }
    write_json(output / "manifest.json", manifest)
    try:
        plan = make_plan(y, ids, args.outer_folds, args.inner_folds, args.repeats, args.seed)
        baseline, baseline_provenance = validate_baseline(args.baseline, cohort, plan, args.alphas)
        manifest["baseline"] = baseline_provenance
        write_json(output / "manifest.json", manifest)
        write_csv(output / "cohort.csv", cohort.rows)
        write_json(output / "splits.json", plan)
        print(f"Verified matching baseline: {len(ids)} participants, {len(plan)} outer folds.", flush=True)
        print("Extracting frozen pretrained transformer embeddings...", flush=True)
        Z, encoder_provenance = extract_features(X, config, output / "checkpoint")
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        if Z.ndim != 2 or Z.shape[0] != len(y) or Z.shape[1] < 1 or not np.isfinite(Z).all():
            raise ValueError("Encoder must return one finite feature vector per participant.")
        feature_path = output / "frozen_features.npy"
        np.save(feature_path, Z, allow_pickle=False)
        manifest["encoder"] = encoder_provenance
        manifest["feature_cache"] = {"path": feature_path.name, "shape": list(Z.shape),
                                     "dtype": str(Z.dtype), "sha256": sha256_file(feature_path)}
        write_json(output / "encoder_provenance.json", encoder_provenance)
        write_json(output / "manifest.json", manifest)
        predictions, fold_results = [], []
        oof = np.full((args.repeats, len(y)), np.nan)
        coverage = np.zeros((args.repeats, len(y)), dtype=int)
        for split in plan:
            repeat, fold = split["repeat"], split["fold"]
            train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
            inner = [(np.array(s["train_positions"]), np.array(s["validation_positions"]))
                     for s in split["inner_splits"]]
            destination = output / f"repeat_{repeat:02d}_fold_{fold:02d}"
            destination.mkdir()
            fold_started = time.perf_counter()
            print(f"Repeat {repeat}, outer fold {fold}: tuning ridge head...", flush=True)
            alpha, search, inner_scores = tune_alpha(Z[train], y[train], args.alphas, inner)
            write_csv(destination / "alpha_search.csv", search)
            write_csv(destination / "inner_oof_predictions.csv", prediction_rows(ids[train], y[train], inner_scores))
            _, training, testing = scaled_features(Z, train, test)
            classifier = fit_ridge(training, y[train], alpha)
            scores = classifier.decision_function(testing)
            result = {"repeat": repeat, "fold": fold, "selected_alpha": alpha,
                      "elapsed_seconds": time.perf_counter() - fold_started, **metrics(y[test], scores)}
            rows = prediction_rows(ids[test], y[test], scores, repeat=repeat, fold=fold)
            write_csv(destination / "test_predictions.csv", rows)
            write_json(destination / "metrics.json", result)
            predictions.extend(rows)
            fold_results.append(result)
            oof[repeat - 1, test] = scores
            coverage[repeat - 1, test] += 1
            print(f"    alpha={alpha:g}; BA={result['balanced_accuracy']:.4f}, "
                  f"AUROC={result['roc_auc']:.4f}, ET recall={result['et_recall']:.4f}", flush=True)
        if not np.all(coverage == 1) or not np.isfinite(oof).all():
            raise RuntimeError("Each participant needs exactly one finite outer prediction per repeat.")
        repeat_results = [{"repeat": r + 1, **metrics(y, oof[r])} for r in range(args.repeats)]
        summary = {
            "primary_metric": "balanced_accuracy", "positive_class": "Essential Tremor (ET=1); PD=0.",
            "cohort_counts": cohort.provenance["condition_counts"],
            "repeat_metrics": repeat_results,
            "across_repeats": {key: describe_values([r[key] for r in repeat_results]) for key in PERFORMANCE_METRICS},
            "outer_fold_metrics": fold_results,
            "across_outer_folds_descriptive": {key: describe_values([r[key] for r in fold_results])
                                               for key in PERFORMANCE_METRICS},
            "always_predict_pd_baseline": metrics(y, -np.ones(len(y))),
            "average_precision_no_skill_reference": float(np.mean(y)),
            "interpretation": [
                "All model test predictions come from outer held-out participants.",
                "Encoder weights are fixed; embedding scalers and ridge heads are fitted within training folds.",
                "Margins are uncalibrated scores, not probabilities.",
                "Fold/repeat standard deviations are descriptive, not confidence intervals; training sets overlap.",
                "The cohort contains only 28 ET participants; repeats do not add independent participants.",
                "This exploratory experiment on published PADS preprocessing establishes no external clinical validity or universal frontier ranking.",
            ],
        }
        predictions.sort(key=lambda row: (row["repeat"], row["participant_id"]))
        paired_rows, comparison = paired_comparison(predictions, baseline)
        write_csv(output / "oof_predictions.csv", predictions)
        write_csv(output / "fold_metrics.csv", [{k: v for k, v in r.items() if not isinstance(v, list)}
                                               for r in fold_results])
        write_json(output / "summary.json", summary)
        write_csv(output / "paired_predictions.csv", paired_rows)
        write_json(output / "baseline_comparison.json", comparison)
        if args.fit_final:
            final_seed = derive_seed(args.seed, args.repeats, args.outer_folds)
            inner = make_inner_splits(y, args.inner_folds, final_seed)
            write_json(output / "final_inner_splits.json", [
                {"train_ids": ids[a].tolist(), "validation_ids": ids[b].tolist()} for a, b in inner
            ])
            alpha, search, scores = tune_alpha(Z, y, args.alphas, inner)
            write_csv(output / "final_alpha_search.csv", search)
            write_csv(output / "final_inner_oof_predictions.csv", prediction_rows(ids, y, scores))
            scaler = StandardScaler()
            training = np.ascontiguousarray(scaler.fit_transform(Z), dtype=np.float64)
            classifier = fit_ridge(training, y, alpha)
            joblib.dump({
                "format_version": 1, "model_type": "frozen_transformer_ridge",
                "scaler": scaler, "classifier": classifier, "encoder_config": asdict(config),
                "encoder_provenance": encoder_provenance, "checkpoint_directory": "checkpoint",
                "input_shape": list(X.shape[1:]), "feature_dimension": Z.shape[1],
                "channel_names": cohort.provenance["channel_names"], "label_mapping": LABELS,
                "decision_threshold": 0.0, "selected_alpha": alpha, "training_ids": ids.tolist(),
                "package_versions": manifest["package_versions"],
                "source_code_sha256": manifest["source_code_sha256"],
                "note": "Head trained on every participant. Evaluation is outer CV; matching pretrained checkpoint is required for inference.",
            }, output / "final_model.joblib", compress=3)
        manifest.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        print(f"Saved transformer experiment: {output}", flush=True)
        return output
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        raise


def main(argv=None):
    args = parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        return run(args)


if __name__ == "__main__":
    main()
