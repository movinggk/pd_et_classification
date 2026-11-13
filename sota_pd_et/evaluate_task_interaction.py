"""Matched three-condition neural ablations / MantisV2 / MultiROCKET-HYDRA.

This command starts a real PADS experiment ONLY when explicitly invoked. All
cohort-fitted objects are refitted within participant-level nested CV. Neural
regularization, epoch and optional decision-threshold selection use inner
validation folds only. Outer training is then refitted for a fixed epoch count.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from .data import DEFAULT_DATA_ROOT, PROJECT_ROOT, load_cohort, sha256_file
from .evaluate import (derive_seed, make_inner_splits, make_plan, metrics,
                       validate_split, write_csv, write_json)
from .model import FeatureMap, ModelConfig, fit_ridge
from .task_interaction import (CHANNEL_INDICES, SEGMENTS, TASKS, VARIANTS,
                               TaskStandardizer, TemporalCrossTaskNet, fit_trajectory,
                               predict_logits, resolve_torch_device, save_network,
                               select_task_segments, variant_config)
from .transformer import EncoderConfig, extract_features


BASELINES = ("mantis", "multirocket_hydra")
ALL_MODELS = tuple(VARIANTS) + BASELINES


def installed_versions():
    result = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "aeon", "mantis-tsfm",
                 "joblib", "threadpoolctl", "safetensors"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def scored_metrics(y, raw_scores, thresholds=0.0):
    """Classification uses raw>threshold; ranking always uses raw scores."""
    raw_scores = np.asarray(raw_scores, dtype=float)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), raw_scores.shape)
    if not np.isfinite(thresholds).all():
        raise ValueError("Thresholds must be finite.")
    result = metrics(y, raw_scores - thresholds)
    result["roc_auc"] = float(roc_auc_score(y, raw_scores))
    result["average_precision_et"] = float(average_precision_score(y, raw_scores))
    return result


def score_rows(ids, y, scores, threshold, **extra):
    return [{**extra, "participant_id": str(pid), "target_et": int(target),
             "raw_score_et": float(score), "decision_threshold": float(threshold),
             "decision_score_et": float(score - threshold), "predicted_et": int(score > threshold)}
            for pid, target, score in zip(ids, y, scores, strict=True)]


def choose_search(rows, kind):
    """Maximize mean inner-fold BA; deterministic, declared tie breaking."""
    best = max(row["inner_mean_balanced_accuracy"] for row in rows)
    tied = [r for r in rows if abs(r["inner_mean_balanced_accuracy"] - best) <= 1e-12]
    if kind == "neural":
        return min(tied, key=lambda r: (-r["weight_decay"], r["epochs"],
                                       abs(r["threshold"]), -r["threshold"]))
    return min(tied, key=lambda r: (-r["alpha"], abs(r["threshold"]), -r["threshold"]))


def search_rows(y, scores, inner_splits, thresholds, **candidate):
    if not np.isfinite(scores).all():
        raise RuntimeError("Missing or nonfinite inner held-out predictions.")
    return [{**candidate, "threshold": float(threshold),
             "inner_mean_balanced_accuracy": float(np.mean([
                 balanced_accuracy_score(y[valid], scores[valid] > threshold)
                 for _, valid in inner_splits])),
             "inner_pooled_balanced_accuracy": float(balanced_accuracy_score(y, scores > threshold)),
             "inner_pooled_roc_auc": float(roc_auc_score(y, scores))}
            for threshold in thresholds]


def tune_neural(X, y, config, inner_splits, args, seed):
    """Reuse a training trajectory across epoch candidates, never across folds."""
    oof = {(decay, epoch): np.full(len(y), np.nan)
           for decay in args.weight_decays for epoch in args.epochs}
    for fold, (train, valid) in enumerate(inner_splits):
        validate_split(train, valid, y)
        scaler = TaskStandardizer().fit(X[train])
        training, validation = scaler.transform(X[train]), scaler.transform(X[valid])
        for decay in args.weight_decays:
            print(f"    inner {fold + 1}/{len(inner_splits)}, weight_decay={decay:g}", flush=True)
            network, predictions, _ = fit_trajectory(
                training, y[train], config, epochs=max(args.epochs),
                learning_rate=args.learning_rate, weight_decay=decay,
                batch_size=args.batch_size, seed=derive_seed(seed, fold), device=args.device,
                validation=validation, checkpoint_epochs=args.epochs,
            )
            for epoch, scores in predictions.items():
                oof[decay, epoch][valid] = scores
            del network
        gc.collect()
    rows = [row for (decay, epoch), scores in oof.items()
            for row in search_rows(y, scores, inner_splits, args.thresholds,
                                   weight_decay=float(decay), epochs=int(epoch))]
    selected = choose_search(rows, "neural")
    return selected, rows, oof[selected["weight_decay"], selected["epochs"]]


def refit_neural(X, y, config, selected, args, seed):
    scaler = TaskStandardizer().fit(X)
    network, _, history = fit_trajectory(
        scaler.transform(X), y, config, epochs=selected["epochs"],
        learning_rate=args.learning_rate, weight_decay=selected["weight_decay"],
        batch_size=args.batch_size, seed=seed, device=args.device,
    )
    return network, scaler, history


def baseline_features(name, X, Z, train, valid, config, seed):
    if name == "mantis":
        feature_map = StandardScaler()
        training = feature_map.fit_transform(Z[train])
        validation = feature_map.transform(Z[valid])
    elif name == "multirocket_hydra":
        feature_map = FeatureMap(config, seed)
        training = feature_map.fit_transform(X[train])
        validation = feature_map.transform(X[valid])
    else:
        raise ValueError(f"Unknown baseline {name}.")
    return feature_map, np.ascontiguousarray(training, dtype=np.float64), np.ascontiguousarray(validation, dtype=np.float64)


def tune_baseline(name, X, Z, y, inner_splits, args, seed, config):
    oof = {alpha: np.full(len(y), np.nan) for alpha in args.alphas}
    for fold, (train, valid) in enumerate(inner_splits):
        validate_split(train, valid, y)
        print(f"    inner {fold + 1}/{len(inner_splits)}, {name} features", flush=True)
        feature_map, training, validation = baseline_features(
            name, X, Z, train, valid, config, derive_seed(seed, fold),
        )
        for alpha in args.alphas:
            classifier = fit_ridge(training, y[train], alpha)
            oof[alpha][valid] = classifier.decision_function(validation)
        del feature_map, training, validation, classifier
        gc.collect()
    rows = [row for alpha, scores in oof.items()
            for row in search_rows(y, scores, inner_splits, args.thresholds, alpha=float(alpha))]
    selected = choose_search(rows, "ridge")
    return selected, rows, oof[selected["alpha"]]


def paired_bootstrap(y, reference_prediction, other_prediction, *, seed, samples):
    """Stratified paired participant bootstrap, conditional on existing CV fits.

    Predictions are (repeats, participants). Each participant is resampled once
    as a cluster containing all repeats; repeated CV never multiplies sample n.
    The interval omits variability from retraining on a new dataset.
    """
    y = np.asarray(y)
    a, b = np.asarray(reference_prediction), np.asarray(other_prediction)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(y) or samples < 1:
        raise ValueError("Bootstrap requires aligned (repeats,participants) predictions.")
    delta = ((a == y[None]).astype(float) - (b == y[None]).astype(float)).mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = np.zeros(samples)
    observed = 0.0
    for target in (0, 1):
        people = np.flatnonzero(y == target)
        if not len(people):
            raise ValueError("Both diagnosis classes are required.")
        observed += float(delta[people].mean()) / 2
        draws += delta[rng.choice(people, size=(samples, len(people)), replace=True)].mean(1) / 2
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"reference_minus_other_balanced_accuracy": observed,
            "conditional_percentile_95_interval": [float(lower), float(upper)],
            "bootstrap_samples": samples, "independent_participants": len(y),
            "limitation": "Conditional on the saved fitted CV models; excludes retraining uncertainty. "
                          "Exploratory paired interval, not proof of superiority or external validity."}


def build_report(output, summaries, comparisons, manifest):
    lines = ["# Temporal / cross-task PADS experiment", "",
             "Generated from this run's outer held-out participant predictions only.", "",
             "ET=1 is positive; PD=0 is negative. PD specificity is PD recall.", "",
             "| Model | Mean repeat BA | ET sensitivity | PD specificity | Pooled AUROC | Mean fold AUROC |", 
             "|---|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        repeat = summary["repeat_metrics"]
        mean = lambda key: float(np.mean([r[key] for r in repeat]))
        lines.append(f"| {name} | {mean('balanced_accuracy'):.4f} | {mean('et_recall'):.4f} | "
                     f"{mean('pd_recall'):.4f} | {mean('roc_auc'):.4f} | "
                     f"{summary['mean_outer_fold_roc_auc']:.4f} |")
    lines.extend(["", "## Evaluation contract", "",
                  f"- Models specified before the run: {', '.join(manifest['config']['models'])}.",
                  f"- All methods use the same five segments: {', '.join(SEGMENTS)}.",
                  "- All task segments, wrists and windows from one participant stay in the same fold.",
                  "- Scaling, supervised training, epoch/regularization/threshold selection use training/inner folds only.",
                  "- Scores are uncalibrated logits or ridge margins, not clinical probabilities.",
                  "- Pooled AUROC/AP compare scores from independently fitted models; also inspect fold AUROC/AP.",
                  "- no_temporal removes ordered modeling BETWEEN windows; its local CNN still uses time within a window.",
                  "- concat_mlp can learn implicit interactions; no_interaction has an additive logit across tasks.",
                  "- These are exploratory analyses of a previously inspected cohort with only 28 ET participants.",
                  "- Official binaries use sample indices; actual raw timestamp gaps are not repaired.",
                  "- Existing 132-channel historical results use different inputs and are not directly matched controls.", ""])
    if comparisons:
        lines.extend(["## Conditional paired comparisons", "",
                      "Intervals resample participants, keeping methods and repeated predictions paired. "
                      "They condition on the saved CV fits and omit retraining uncertainty.", "",
                      "| Reference minus other | BA difference | Conditional 95% interval |", "|---|---:|---:|"])
        for name, result in comparisons.items():
            lo, hi = result["conditional_percentile_95_interval"]
            lines.append(f"| full − {name} | {result['reference_minus_other_balanced_accuracy']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", choices=ALL_MODELS, nargs="+", default=list(ALL_MODELS))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--reference-splits", type=Path,
                        help="Require an exact match to this saved plan; auto-check the existing baseline when present.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, nargs="+", default=[15, 30, 60])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.001, 0.01])
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--windows-per-segment", type=int, default=4)
    parser.add_argument("--interaction-rank", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0],
                        help="Common inner-only threshold candidates for ALL methods; default fixes zero.")
    parser.add_argument("--alphas", type=float, nargs="+", default=np.logspace(-3, 3, 7).tolist())
    parser.add_argument("--mr-kernels", type=int, default=6250)
    parser.add_argument("--hydra-groups", type=int, default=64)
    parser.add_argument("--hydra-kernels", type=int, default=8)
    parser.add_argument("--hydra-device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--fit-final", action="store_true", help="Separately tune/refit full model on the entire cohort.")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32 or args.threads < 1 or args.threads > (os.cpu_count() or 1):
        parser.error("Invalid seed or CPU thread count.")
    if min(args.outer_folds, args.inner_folds) < 2 or args.repeats < 1:
        parser.error("Use at least two outer/inner folds and one repeat.")
    if min(args.batch_size, args.encoder_batch_size, args.bootstrap_samples) < 1:
        parser.error("Batch sizes and bootstrap count must be positive.")
    if any(e < 1 for e in args.epochs) or any(not np.isfinite(w) or w < 0 for w in args.weight_decays):
        parser.error("Epochs must be positive and weight decays finite/nonnegative.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if any(not np.isfinite(t) for t in args.thresholds) or any(not np.isfinite(a) or a <= 0 for a in args.alphas):
        parser.error("Thresholds must be finite; ridge alphas finite/positive.")
    if args.mr_kernels < 84 or args.hydra_groups < 2 or args.hydra_groups % 2 or args.hydra_kernels < 1:
        parser.error("Need mr-kernels>=84, a positive even hydra-groups, and hydra-kernels>=1.")
    if len(set(args.models)) != len(args.models):
        parser.error("Do not repeat model names.")
    if args.fit_final and "full" not in args.models:
        parser.error("--fit-final requires full in --models.")
    args.epochs = sorted(set(args.epochs))
    args.weight_decays = sorted(set(args.weight_decays))
    args.thresholds = sorted(set(args.thresholds))
    args.alphas = sorted(set(args.alphas))
    try:
        variant_config("full", hidden_dim=args.hidden_dim, heads=args.heads,
                       temporal_layers=args.temporal_layers, windows_per_segment=args.windows_per_segment,
                       interaction_rank=args.interaction_rank, dropout=args.dropout)
    except ValueError as error:
        parser.error(str(error))
    return args


def run(args):
    started = time.perf_counter()
    args.device = resolve_torch_device(args.device)
    torch.set_num_threads(args.threads)
    print("Verifying official PADS data and selecting three task conditions...", flush=True)
    cohort = load_cohort(args.data_root)
    X, y, ids = select_task_segments(cohort.X), cohort.y, cohort.ids
    baseline_X = X.reshape(len(X), 60, X.shape[-1])
    plan = make_plan(y, ids, args.outer_folds, args.inner_folds, args.repeats, args.seed)
    reference = args.reference_splits
    automatic = PROJECT_ROOT / "artifacts" / "pads_mr_hydra" / "splits.json"
    if reference is None and automatic.is_file() and (args.outer_folds, args.inner_folds, args.repeats, args.seed) == (5, 3, 1, 20261017):
        reference = automatic
    if reference is not None and json.loads(reference.read_text()) != plan:
        raise ValueError("Reference outer/inner participant splits differ from this experiment.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or PROJECT_ROOT / "artifacts" / "pads_temporal_cross_task" / stamp).expanduser().resolve()
    protected = (args.data_root.expanduser().resolve(), Path(__file__).parent.resolve(),
                 PROJECT_ROOT / "artifacts" / "pads_mr_hydra", PROJECT_ROOT / "artifacts" / "pads_mantis_v2",
                 PROJECT_ROOT / "models")
    if any(output.is_relative_to(p.resolve()) for p in protected):
        raise ValueError("Choose a new result directory outside data, source, checkpoints and historical experiments.")
    output.mkdir(parents=True, exist_ok=False)
    network_configs = {name: variant_config(
        name, samples_per_segment=X.shape[-1], hidden_dim=args.hidden_dim, heads=args.heads,
        temporal_layers=args.temporal_layers, interaction_rank=args.interaction_rank,
        dropout=args.dropout, windows_per_segment=args.windows_per_segment,
    ) for name in args.models if name in VARIANTS}
    feature_config = ModelConfig(args.mr_kernels, args.hydra_groups, args.hydra_kernels,
                                 args.batch_size, args.threads, args.hydra_device)
    source_paths = sorted(Path(__file__).parent.rglob("*.py"))
    manifest = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "python": sys.version, "platform": platform.platform(),
        "package_versions": installed_versions(), "completed_models": [],
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "network_configs": {k: asdict(v) for k, v in network_configs.items()},
        "parameter_counts": {k: sum(p.numel() for p in TemporalCrossTaskNet(v).parameters())
                             for k, v in network_configs.items()},
        "dataset": cohort.provenance,
        "input_protocol": {"segment_names": list(SEGMENTS), "task_names": list(TASKS),
                           "selected_channel_indices": list(CHANNEL_INDICES), "shape": list(X.shape),
                           "raw_timestamp_repair": False,
                           "segment_boundary_handling": "CNN windows never cross boundaries; ordered encoder uses nominal positions including 48-sample crop gaps."},
        "reference_splits": {"path": str(reference), "sha256": sha256_file(reference)} if reference else None,
        "source_code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in source_paths},
        "selection": {"metric": "mean inner-fold balanced accuracy",
                      "neural": "weight decay and epoch; then stronger decay, fewer epochs on ties",
                      "baseline": "ridge alpha; stronger alpha on ties",
                      "threshold": "same candidate list for every model; ties prefer closest to zero, then higher",
                      "outer_refit": "fresh model/scalers on outer training participants; selected FIXED epochs",
                      "class_weights": "n_train/(2*n_class), recomputed inside each fit",
                      "score_semantics": "Uncalibrated logit or ridge margin; ET iff raw_score > selected_threshold"},
        "limitations": ["Exploratory, previously inspected single-center cohort; 28 ET cases.",
                        "Temporal learning concerns seconds within recorded tasks, not longitudinal disease progression.",
                        "Official preprocessing has task selection/cropping; raw timing gaps remain.",
                        "No clinical interpretation of learned latent dimensions or attention is established."],
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "splits.json", plan)
    write_csv(output / "cohort.csv", cohort.rows)
    with zipfile.ZipFile(output / "evaluated_sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, str(path.relative_to(PROJECT_ROOT)))
    summaries, all_predictions, comparisons = {}, {}, {}
    try:
        Z = None
        if "mantis" in args.models:
            Z, provenance = extract_features(
                baseline_X, EncoderConfig(device=args.device, batch_size=args.encoder_batch_size), output / "checkpoint",
            )
            np.save(output / "mantis_frozen_features.npy", Z, allow_pickle=False)
            provenance["feature_sha256"] = sha256_file(output / "mantis_frozen_features.npy")
            write_json(output / "mantis_encoder_provenance.json", provenance)
        for name in args.models:
            destination = output / name
            destination.mkdir()
            raw = np.full((args.repeats, len(y)), np.nan)
            thresholds = np.full_like(raw, np.nan)
            coverage = np.zeros_like(raw, dtype=int)
            rows, fold_metrics = [], []
            print(f"Model: {name}; device={args.device}", flush=True)
            for split in plan:
                repeat, fold = split["repeat"], split["fold"]
                fold_started = time.perf_counter()
                train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
                inner = [(np.array(s["train_positions"]), np.array(s["validation_positions"])) for s in split["inner_splits"]]
                folder = destination / f"repeat_{repeat:02d}_fold_{fold:02d}"
                folder.mkdir()
                print(f"  repeat {repeat}, outer {fold}: tuning on {len(train)} participants", flush=True)
                manifest["current"] = {"model": name, "repeat": repeat, "fold": fold}
                write_json(output / "manifest.json", manifest)
                with threadpool_limits(limits=args.threads):
                    if name in VARIANTS:
                        selected, search, inner_scores = tune_neural(
                            X[train], y[train], network_configs[name], inner, args, split["model_seed"],
                        )
                        network, scaler, history = refit_neural(
                            X[train], y[train], network_configs[name], selected, args, split["model_seed"],
                        )
                        # The outer test set first enters the neural path HERE.
                        scores = predict_logits(network, scaler.transform(X[test]), args.device, args.batch_size)
                        write_csv(folder / "training_history.csv", history)
                        del network, scaler
                    else:
                        selected, search, inner_scores = tune_baseline(
                            name, baseline_X[train], Z[train] if Z is not None else None,
                            y[train], inner, args, split["model_seed"], feature_config,
                        )
                        fmap, training, validation = baseline_features(
                            name, baseline_X, Z, train, test, feature_config, split["model_seed"],
                        )
                        classifier = fit_ridge(training, y[train], selected["alpha"])
                        scores = classifier.decision_function(validation)
                        del fmap, training, validation, classifier
                threshold = selected["threshold"]
                result = {"model": name, "repeat": repeat, "fold": fold,
                          "elapsed_seconds": time.perf_counter() - fold_started,
                          **scored_metrics(y[test], scores, threshold)}
                write_json(folder / "selected.json", selected)
                write_csv(folder / "inner_search.csv", search)
                write_csv(folder / "inner_oof_predictions.csv", score_rows(ids[train], y[train], inner_scores, threshold))
                outer_rows = score_rows(ids[test], y[test], scores, threshold, model=name, repeat=repeat, fold=fold)
                write_csv(folder / "test_predictions.csv", outer_rows)
                write_json(folder / "metrics.json", result)
                rows.extend(outer_rows)
                fold_metrics.append(result)
                raw[repeat - 1, test], thresholds[repeat - 1, test] = scores, threshold
                coverage[repeat - 1, test] += 1
                print(f"    outer BA={result['balanced_accuracy']:.4f}; ET={result['et_correct']}/{result['n_et']}", flush=True)
                gc.collect()
            if not np.all(coverage == 1) or not np.isfinite(raw).all() or not np.isfinite(thresholds).all():
                raise RuntimeError("Each participant must have exactly one finite outer prediction per repeat.")
            repeat_metrics = [{"repeat": r + 1, **scored_metrics(y, raw[r], thresholds[r])} for r in range(args.repeats)]
            summary = {"model": name, "repeat_metrics": repeat_metrics, "outer_fold_metrics": fold_metrics,
                       "mean_outer_fold_roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics])),
                       "mean_outer_fold_average_precision_et": float(np.mean([m["average_precision_et"] for m in fold_metrics])),
                       "mean_repeat_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in repeat_metrics])),
                       "always_pd": scored_metrics(y, -np.ones(len(y))),
                       "et_prevalence_ap_reference": float(y.mean())}
            write_csv(destination / "oof_predictions.csv", sorted(rows, key=lambda r: (r["repeat"], r["participant_id"])))
            write_csv(destination / "fold_metrics.csv", [{k: v for k, v in r.items() if not isinstance(v, list)} for r in fold_metrics])
            write_json(destination / "summary.json", summary)
            summaries[name], all_predictions[name] = summary, (raw > thresholds).astype(int)
            manifest["completed_models"].append(name)
            write_json(output / "manifest.json", manifest)
        if "full" in all_predictions:
            comparisons = {name: paired_bootstrap(y, all_predictions["full"], pred,
                                                  seed=args.seed, samples=args.bootstrap_samples)
                           for name, pred in all_predictions.items() if name != "full"}
        write_json(output / "paired_comparisons.json", comparisons)
        write_json(output / "summary.json", summaries)
        if args.fit_final:
            final_seed = derive_seed(args.seed, args.repeats, args.outer_folds)
            inner = make_inner_splits(y, args.inner_folds, final_seed)
            write_json(output / "final_inner_splits.json", [
                {"train_ids": ids[a].tolist(), "validation_ids": ids[b].tolist()} for a, b in inner])
            print("Separately tuning/refitting final full model on all participants...", flush=True)
            with threadpool_limits(limits=args.threads):
                selected, search, inner_scores = tune_neural(X, y, network_configs["full"], inner, args, final_seed)
                network, scaler, history = refit_neural(X, y, network_configs["full"], selected, args, final_seed)
            write_json(output / "final_selected.json", selected)
            write_csv(output / "final_inner_search.csv", search)
            write_csv(output / "final_training_history.csv", history)
            write_csv(output / "final_inner_oof_predictions.csv", score_rows(ids, y, inner_scores, selected["threshold"]))
            save_network(output / "final_model.pt", network, scaler, threshold=selected["threshold"], metadata={
                "training_ids": ids.tolist(), "selected": selected,
                "package_versions": manifest["package_versions"],
                "source_code_sha256": manifest["source_code_sha256"],
                "note": "Fitted on all participants; use outer CV, not training performance, for evaluation.",
            })
        build_report(output, summaries, comparisons, manifest)
        manifest.pop("current", None)
        manifest.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        print(f"Saved experiment: {output}", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                        elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        raise
    return output


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
