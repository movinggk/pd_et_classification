"""Run honest, participant-level nested cross-validation on the current PADS data.

Run from the project root: python -m sota_pd_et.evaluate --n-jobs 4 --fit-final
No training or evaluation occurs when this module is imported.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .data import DEFAULT_DATA_ROOT, LABELS, PROJECT_ROOT, load_cohort, sha256_file
from .model import FeatureMap, FittedModel, ModelConfig, fit_ridge


PERFORMANCE_METRICS = (
    "balanced_accuracy", "accuracy", "roc_auc", "average_precision_et", "et_recall",
    "pd_recall", "et_precision", "f1_et", "macro_f1", "matthews_correlation",
)


def write_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]):
    import csv

    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def derive_seed(seed: int, *parts: int) -> int:
    return int(np.random.SeedSequence([seed, *parts]).generate_state(1)[0])


def validate_split(train, test, y):
    train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
    if not len(train) or not len(test):
        raise ValueError("A training or validation/test split is empty.")
    if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
        raise ValueError("Duplicate participants inside a split.")
    if np.intersect1d(train, test).size:
        raise ValueError("Participant leakage between training and validation/test.")
    if train.min() < 0 or test.min() < 0 or max(train.max(), test.max()) >= len(y):
        raise ValueError("Split index outside the cohort.")
    if not np.array_equal(np.unique(y[train]), [0, 1]):
        raise ValueError("Both classes must occur in every training split.")
    if not np.array_equal(np.unique(y[test]), [0, 1]):
        raise ValueError("Both classes must occur in every validation/test split.")


def make_inner_splits(y, n_splits: int, seed: int):
    if n_splits < 2 or np.bincount(y, minlength=2).min() < n_splits:
        raise ValueError("Inner fold count exceeds the number of participants in a class.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y))
    seen = np.zeros(len(y), dtype=int)
    for train, valid in splits:
        validate_split(train, valid, y)
        seen[valid] += 1
    if not np.all(seen == 1):
        raise RuntimeError("Each inner participant must be validated exactly once.")
    return splits


def make_plan(y, ids, outer_folds: int, inner_folds: int, repeats: int, seed: int):
    if len(np.unique(ids)) != len(ids) or len(ids) != len(y):
        raise ValueError("Require exactly one uniquely identified row per participant.")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("The target must contain exactly PD=0 and ET=1.")
    if repeats < 1 or outer_folds < 2 or np.bincount(y).min() < outer_folds:
        raise ValueError("Invalid repeat count or outer fold count for this cohort.")
    plan = []
    for repeat in range(repeats):
        split_seed = derive_seed(seed, repeat)
        outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=split_seed)
        seen = np.zeros(len(y), dtype=int)
        for fold, (train, test) in enumerate(outer.split(np.zeros(len(y)), y)):
            validate_split(train, test, y)
            seen[test] += 1
            model_seed = derive_seed(seed, repeat, fold)
            inner = make_inner_splits(y[train], inner_folds, model_seed)
            plan.append({
                "repeat": repeat + 1,
                "fold": fold + 1,
                "model_seed": model_seed,
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "train_ids": ids[train].tolist(),
                "test_ids": ids[test].tolist(),
                "inner_splits": [{
                    "train_positions": a.tolist(),
                    "validation_positions": b.tolist(),
                    "train_ids": ids[train[a]].tolist(),
                    "validation_ids": ids[train[b]].tolist(),
                } for a, b in inner],
            })
        if not np.all(seen == 1):
            raise RuntimeError("Each participant must be tested exactly once per repeat.")
    return plan


def metrics(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("Need one finite decision score per participant.")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Metrics require both PD and ET.")
    predicted = (scores > 0.0).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "accuracy": float(accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision_et": float(average_precision_score(y, scores)),
        "et_recall": tp / (tp + fn),
        "pd_recall": tn / (tn + fp),
        "et_precision": float(precision_score(y, predicted, zero_division=0)),
        "f1_et": float(f1_score(y, predicted, zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(y, predicted)),
        "n_pd": tn + fp,
        "n_et": tp + fn,
        "pd_correct": tn,
        "pd_called_et": fp,
        "et_called_pd": fn,
        "et_correct": tp,
        "confusion_matrix_true_rows_predicted_columns_pd_et": [[tn, fp], [fn, tp]],
    }


def choose_alpha(rows):
    """Maximize mean inner-fold balanced accuracy; ties favor larger alpha."""
    best = max(row["inner_mean_balanced_accuracy"] for row in rows)
    candidates = [r for r in rows if abs(r["inner_mean_balanced_accuracy"] - best) <= 1e-12]
    return max(candidates, key=lambda r: r["alpha"])["alpha"]


def tune_alpha(X, y, config, alphas, inner_splits, seed):
    """Refit features/scalers separately in each inner fold; reuse only across alphas."""
    oof_scores = np.full((len(alphas), len(y)), np.nan)
    fold_ba = np.empty((len(alphas), len(inner_splits)))
    for fold, (train, valid) in enumerate(inner_splits):
        validate_split(train, valid, y)
        print(f"    Inner {fold + 1}/{len(inner_splits)}: feature extraction", flush=True)
        feature_map = FeatureMap(config, derive_seed(seed, fold))
        Z_train = feature_map.fit_transform(X[train])
        Z_valid = feature_map.transform(X[valid])
        for index, alpha in enumerate(alphas):
            classifier = fit_ridge(Z_train, y[train], alpha)
            scores = classifier.decision_function(Z_valid)
            oof_scores[index, valid] = scores
            fold_ba[index, fold] = balanced_accuracy_score(y[valid], scores > 0.0)
        del feature_map, Z_train, Z_valid, classifier
        gc.collect()
    if not np.isfinite(oof_scores).all():
        raise RuntimeError("Missing/nonfinite inner out-of-fold scores.")
    rows = [{
        "alpha": float(alpha),
        "inner_mean_balanced_accuracy": float(np.mean(fold_ba[index])),
        "inner_std_balanced_accuracy": float(np.std(fold_ba[index], ddof=1)),
        "inner_pooled_balanced_accuracy": float(balanced_accuracy_score(y, oof_scores[index] > 0)),
        "inner_pooled_roc_auc": float(roc_auc_score(y, oof_scores[index])),
    } for index, alpha in enumerate(alphas)]
    selected = choose_alpha(rows)
    selected_index = list(alphas).index(selected)
    return selected, rows, oof_scores[selected_index]


def prediction_rows(ids, y, scores, **extra):
    return [{
        **extra,
        "participant_id": str(pid),
        "target_et": int(target),
        "decision_score_et": float(score),
        "predicted_et": int(score > 0.0),
    } for pid, target, score in zip(ids, y, scores, strict=True)]


def fit_full_model(X, y, config, alpha, seed):
    features = FeatureMap(config, seed)
    Z = features.fit_transform(X)
    return FittedModel(features, fit_ridge(Z, y, alpha))


def describe_values(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "values": values.tolist(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--n-jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mr-kernels", type=int, default=6250)
    parser.add_argument("--hydra-groups", type=int, default=64)
    parser.add_argument("--hydra-kernels", type=int, default=8)
    parser.add_argument("--hydra-device", choices=("auto", "cpu", "cuda"), default="cpu",
                        help="HYDRA convolution device; MultiROCKET and ridge remain on CPU.")
    parser.add_argument("--alphas", type=float, nargs="+", default=np.logspace(-3, 3, 7).tolist())
    parser.add_argument("--fit-final", action="store_true", help="Tune/refit all participants after CV; save joblib model.")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.seed > 2**32 - 1:
        parser.error("--seed must be an integer in [0, 2**32-1].")
    if args.n_jobs == 0 or args.n_jobs < -1 or args.batch_size < 1:
        parser.error("Use --n-jobs -1 or a positive integer, and a positive --batch-size.")
    if args.n_jobs > (os.cpu_count() or 1):
        parser.error("--n-jobs exceeds the available CPU count.")
    if args.mr_kernels < 84 or args.hydra_groups < 2 or args.hydra_groups % 2 or args.hydra_kernels < 1:
        parser.error("Need --mr-kernels >=84, a positive even --hydra-groups, and --hydra-kernels >=1.")
    if not args.alphas or any(not np.isfinite(a) or a <= 0 for a in args.alphas):
        parser.error("All ridge alpha candidates must be finite and strictly positive.")
    args.alphas = sorted(set(args.alphas))
    return args


def run(args):
    if version("aeon") != "1.6.0":
        raise RuntimeError("Install the versions in sota_pd_et/requirements.txt.")
    from .hydra_gpu import resolve_device
    import torch

    device = resolve_device(args.hydra_device)
    config = ModelConfig(args.mr_kernels, args.hydra_groups, args.hydra_kernels,
                         args.batch_size, args.n_jobs, args.hydra_device)
    print(f"HYDRA device: {device}; MultiROCKET/scalers/ridge: CPU", flush=True)
    print("Loading and SHA-256-verifying the current PD/ET cohort...", flush=True)
    cohort = load_cohort(args.data_root)
    X, y, ids = cohort.X, cohort.y, cohort.ids
    print(f"Participants: PD={sum(y == 0)}, ET={sum(y == 1)}; tensor={X.shape}", flush=True)
    plan = make_plan(y, ids, args.outer_folds, args.inner_folds, args.repeats, args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or PROJECT_ROOT / "artifacts" / "pd_et_mr_hydra" / stamp).expanduser().resolve()
    if output.is_relative_to(args.data_root.expanduser().resolve()):
        raise ValueError("Choose an output directory outside the source dataset.")
    # Do not mix predictions/configurations from different runs.
    output.mkdir(parents=True, exist_ok=False)
    versions = {p: version(p) for p in (
        "aeon", "numpy", "scipy", "pandas", "scikit-learn", "numba", "torch", "joblib", "threadpoolctl",
    )}
    manifest = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "acceleration": {
            "hydra_device_requested": args.hydra_device,
            "hydra_device_used": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch_cuda_version": torch.version.cuda,
            "multirocket_device": "cpu",
            "scalers_and_ridge_device": "cpu",
            "numerics": "CUDA uses the same fitted CPU kernels/channel indices; float32 "
            "convolutions and scatter reductions may differ numerically from CPU. "
            "TF32 is disabled for the CUDA convolution path.",
        },
        "config": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "output": str(output), "model": asdict(config)},
        "source_code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
                               for p in sorted(Path(__file__).parent.rglob("*.py"))},
        "dataset": cohort.provenance,
        "protocol": {
            "unit": "One participant, with all task/wrist/sensor/axis channels retained together.",
            "primary_metric": "Balanced accuracy from outer out-of-fold class predictions.",
            "alpha_selection": "Maximum mean inner-fold balanced accuracy; ties choose larger alpha.",
            "decision_rule": "ET if ridge decision score > 0; PD otherwise. Threshold fixed in advance.",
            "class_weight": "n_train / (2 * class_count), recalculated within every ridge fit.",
            "feature_fit": "HYDRA/MultiROCKET and both scalers refitted in each inner/outer training fold.",
            "probabilities": "Not produced. Decision scores are signed, uncalibrated ridge margins.",
            "representation_limit": "Published preprocessing includes task selection from the original study.",
        },
    }
    write_json(output / "manifest.json", manifest)
    write_csv(output / "cohort.csv", cohort.rows)
    write_json(output / "splits.json", plan)
    all_predictions, fold_metrics, repeat_metrics = [], [], []
    oof = np.full((args.repeats, len(y)), np.nan)
    coverage = np.zeros((args.repeats, len(y)), dtype=int)
    started = time.perf_counter()
    try:
        for split in plan:
            repeat, fold = split["repeat"], split["fold"]
            train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
            inner = [(np.array(s["train_positions"]), np.array(s["validation_positions"]))
                     for s in split["inner_splits"]]
            destination = output / f"repeat_{repeat:02d}_fold_{fold:02d}"
            destination.mkdir()
            print(f"Repeat {repeat}/{args.repeats}, outer fold {fold}/{args.outer_folds}", flush=True)
            fold_started = time.perf_counter()
            alpha, search, inner_scores = tune_alpha(X[train], y[train], config,
                                                     args.alphas, inner, split["model_seed"])
            write_csv(destination / "alpha_search.csv", search)
            write_csv(destination / "inner_oof_predictions.csv", prediction_rows(ids[train], y[train], inner_scores))
            print(f"    Selected alpha={alpha:g}; refitting outer training participants", flush=True)
            model = fit_full_model(X[train], y[train], config, alpha, split["model_seed"])
            scores = model.decision_function(X[test])
            result = {"repeat": repeat, "fold": fold, "selected_alpha": alpha,
                      "elapsed_seconds": time.perf_counter() - fold_started, **metrics(y[test], scores)}
            rows = prediction_rows(ids[test], y[test], scores, repeat=repeat, fold=fold)
            write_csv(destination / "test_predictions.csv", rows)
            write_json(destination / "metrics.json", result)
            fold_metrics.append(result)
            all_predictions.extend(rows)
            oof[repeat - 1, test] = scores
            coverage[repeat - 1, test] += 1
            print(f"    BA={result['balanced_accuracy']:.4f}, AUROC={result['roc_auc']:.4f}, "
                  f"ET recall={result['et_recall']:.4f}", flush=True)
            del model
            gc.collect()
        if not np.all(coverage == 1) or not np.isfinite(oof).all():
            raise RuntimeError("Each participant needs exactly one outer prediction per repeat.")
        for repeat in range(args.repeats):
            repeat_metrics.append({"repeat": repeat + 1, **metrics(y, oof[repeat])})
        summary = {
            "primary_metric": "balanced_accuracy",
            "positive_class": "Essential Tremor (ET=1); Parkinson's disease is PD=0.",
            "cohort_counts": cohort.provenance["condition_counts"],
            "repeat_metrics": repeat_metrics,
            "across_repeats": {key: describe_values([r[key] for r in repeat_metrics])
                               for key in PERFORMANCE_METRICS},
            "outer_fold_metrics": fold_metrics,
            "across_outer_folds_descriptive": {key: describe_values([r[key] for r in fold_metrics])
                                              for key in PERFORMANCE_METRICS},
            "always_predict_pd_baseline": metrics(y, -np.ones(len(y))),
            "average_precision_no_skill_reference": float(np.mean(y)),
            "interpretation": [
                "All reported model test predictions are from outer held-out participants.",
                "AUROC and AP use continuous margins. Scores are not probabilities.",
                "Repeat-level pooled AUROC/AP compare margins from separately fitted models; "
                "also inspect the per-fold ranking metrics.",
                "Fold/repeat standard deviations are descriptive, not confidence intervals; fits overlap.",
                "Repeating CV does not create more independent ET participants.",
                "This estimates the specified procedure on the published PADS representation; "
                "it establishes neither a PD/ET SOTA ranking nor external clinical validity.",
            ],
        }
        write_csv(output / "oof_predictions.csv", sorted(all_predictions,
                  key=lambda r: (r["repeat"], r["participant_id"])))
        write_csv(output / "fold_metrics.csv", [{k: v for k, v in r.items() if not isinstance(v, list)}
                                               for r in fold_metrics])
        write_json(output / "summary.json", summary)
        if args.fit_final:
            print("Tuning/refitting the optional final model on all participants...", flush=True)
            final_seed = derive_seed(args.seed, args.repeats, args.outer_folds)
            inner = make_inner_splits(y, args.inner_folds, final_seed)
            write_json(output / "final_inner_splits.json", [
                {"train_ids": ids[a].tolist(), "validation_ids": ids[b].tolist()} for a, b in inner
            ])
            alpha, search, final_inner_scores = tune_alpha(X, y, config, args.alphas, inner, final_seed)
            write_csv(output / "final_alpha_search.csv", search)
            write_csv(output / "final_inner_oof_predictions.csv", prediction_rows(ids, y, final_inner_scores))
            model = fit_full_model(X, y, config, alpha, final_seed)
            joblib.dump({
                "format_version": 1,
                "model": model,
                "input_shape": list(X.shape[1:]),
                "channel_names": cohort.provenance["channel_names"],
                "label_mapping": LABELS,
                "decision_threshold": 0.0,
                "selected_alpha": alpha,
                "training_ids": ids.tolist(),
                "package_versions": versions,
                "source_code_sha256": manifest["source_code_sha256"],
                "evaluation_summary": "summary.json",
                "note": "Final model trained on all participants. Its training performance is not evaluation.",
            }, output / "final_model.joblib", compress=3)
        manifest.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        print(f"\nSaved results: {output}", flush=True)
        for key in ("balanced_accuracy", "roc_auc", "et_recall", "pd_recall", "average_precision_et"):
            statistic = summary["across_repeats"][key]
            print(f"{key}: {statistic['mean']:.6f}", flush=True)
        return output
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        raise


def main(argv=None):
    args = parse_args(argv)
    # CV itself is sequential; parallelize each transform and bound BLAS oversubscription.
    workers = (os.cpu_count() or 1) if args.n_jobs == -1 else args.n_jobs
    with threadpool_limits(limits=workers):
        run(args)


if __name__ == "__main__":
    main()
