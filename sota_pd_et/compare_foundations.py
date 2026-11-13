"""Compare frozen and fine-tuned pretrained encoders on PADS PD versus ET.

Examples (from the project root):
  python -m sota_pd_et.compare_foundations --models all --mode frozen
  python -m sota_pd_et.compare_foundations --models tspulse --mode finetune

All methods share participant-level outer assignments. Frozen mode tunes a
class-balanced ridge head. Fine-tuned mode reserves checkpoint validation,
tunes LR within fitting participants, and restores its own best checkpoint.
Both modes select thresholds inside the outer training pool. No experiment
runs on import. Use --check-data for validation without weights or training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import sys
import time
import uuid

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import DEFAULT_DATA_ROOT, PROJECT_ROOT, load_cohort, sha256_file
from .evaluate import (
    PERFORMANCE_METRICS, derive_seed, describe_values, make_inner_splits, make_plan,
    metrics, prediction_rows, write_csv, write_json,
)
from .evaluate_transformer import scaled_features, validate_baseline
from .foundations.ridge import tune_alpha as tune_ridge
from .foundations.decision import (
    DECISION_RULES, DEFAULT_DECISION_RULE, apply_threshold, saved_raw_threshold,
    select_threshold, validation_fold_ids,
)
from .model import fit_ridge
from .foundations.registry import DEFAULT_MODELS, SPECS, specifications
from .foundations.progress import progress_bar, progress_message
from .foundations.validation import FINETUNE_PROTOCOL, checkpoint_split, make_checkpoint_plan
from .foundations.restart import restore_configuration, validate_restart, begin_restart, archive_incomplete
from .foundations.runtime import channel_override, checkpoint_override, override_mapping, model_runtime_settings


def _source_hashes():
    sources = [Path(__file__), Path(__file__).parent / "data.py", Path(__file__).parent / "evaluate.py",
               Path(__file__).parent / "evaluate_transformer.py", Path(__file__).parent / "model.py",
               Path(__file__).parent / "predict_foundation.py"]
    sources.extend((Path(__file__).parent / "foundations").rglob("*.py"))
    return {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sorted(sources)}


def create_run_directory(output_root):
    """Atomically allocate a new run without touching existing run contents."""
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"run_{stamp}_{uuid.uuid4().hex[:12]}"
        output = output_root / run_id
        try:
            output.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_id, output
    raise RuntimeError(f"Could not allocate a unique run directory under {output_root}.")


def _summary(y, oof, fold_results):
    repeat_results = [{"repeat": i + 1, **metrics(y, row)} for i, row in enumerate(oof)]
    return {
        "primary_metric": "balanced_accuracy", "positive_class": "Essential Tremor (ET=1); PD=0",
        "repeat_metrics": repeat_results,
        "across_repeats": {key: describe_values([r[key] for r in repeat_results]) for key in PERFORMANCE_METRICS},
        "outer_fold_metrics": fold_results,
        "across_outer_folds_descriptive": {key: describe_values([r[key] for r in fold_results]) for key in PERFORMANCE_METRICS},
        "always_predict_pd_baseline": metrics(y, -np.ones(len(y))),
        "average_precision_no_skill_reference": float(np.mean(y)),
        "interpretation": [
            "Only outer held-out participant predictions measure evaluation performance.",
            "Margins subtract the saved training-pool validation threshold; positive means ET. Zero is classified as PD.",
            "Threshold selection changes class decisions, not within-fold ranking. Raw-zero metrics are also saved per fold.",
            "Pooled AUROC/AP combine scores from different fits; inspect fold metrics too.",
            "Fold/repeat standard deviations are descriptive; overlapping training sets are not independent.",
            "Model ranking is exploratory; selecting a winner from this table requires separate confirmation.",
        ],
    }


def _write_results(destination, y, oof, coverage, predictions, folds):
    if not np.all(coverage == 1) or not np.isfinite(oof).all():
        raise RuntimeError("Every participant needs exactly one outer prediction per repeat.")
    predictions.sort(key=lambda r: (r["repeat"], r["participant_id"]))
    write_csv(destination / "oof_predictions.csv", predictions)
    write_csv(destination / "fold_metrics.csv", [{k: v for k, v in row.items() if not isinstance(v, list)} for row in folds])
    summary = _summary(y, oof, folds)
    write_json(destination / "summary.json", summary)
    return summary


def score_diagnostics(scores):
    """Describe the fixed-zero decision rule without changing any predictions."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected nonempty finite participant scores.")
    predicted_et = int(np.sum(scores > 0))
    return {
        "decision_threshold": 0., "participants": len(scores),
        "predicted_et": predicted_et, "predicted_pd": len(scores) - predicted_et,
        "single_predicted_class": "PD" if predicted_et == 0 else "ET" if predicted_et == len(scores) else None,
        "score_min": float(np.min(scores)), "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)), "score_std": float(np.std(scores)),
    }


def report_fold(model, mode, repeat, fold, result, scores, folder, progress, *, raw_scores=None, raw_threshold=0.):
    diagnostics = score_diagnostics(scores)
    diagnostics["raw_decision_threshold"] = float(raw_threshold)
    if raw_scores is not None:
        raw = score_diagnostics(raw_scores)
        diagnostics.update(raw_score_min=raw["score_min"], raw_score_max=raw["score_max"],
                           predicted_et_at_raw_zero=raw["predicted_et"])
    write_json(folder / "score_diagnostics.json", diagnostics)
    progress_message(
        f"  {model} {mode}: repeat {repeat} fold {fold}, BA={result['balanced_accuracy']:.4f}, "
        f"ET recall={result['et_recall']:.4f}, PD recall={result['pd_recall']:.4f}, "
        f"AUROC={result['roc_auc']:.4f}, predicted ET={diagnostics['predicted_et']}/{len(scores)}, "
        f"raw threshold={raw_threshold:.4g}, margins=[{diagnostics['score_min']:.4g}, {diagnostics['score_max']:.4g}]",
        enabled=progress,
    )
    if diagnostics["single_predicted_class"] is not None:
        progress_message(f"    All outer-test participants were predicted as {diagnostics['single_predicted_class']} "
                         "at the saved decision rule. Inspect threshold_selection.json and score_diagnostics.json.", enabled=progress)


def decision_prediction_rows(ids, y, raw_scores, selection, **extra):
    """Save both raw scores and shifted margins, with both thresholds explicit."""
    threshold = selection["raw_decision_threshold"]
    scores = apply_threshold(raw_scores, threshold)
    rows = prediction_rows(ids, y, scores, decision_threshold=0., raw_decision_threshold=threshold, **extra)
    for row, raw in zip(rows, raw_scores, strict=True):
        row["raw_score_et"] = float(raw)
    return scores, rows


def save_threshold_selection(folder, y, inner_scores, inner, rule):
    selection = select_threshold(y, inner_scores, validation_fold_ids(len(y), inner), rule=rule)
    write_json(folder / "threshold_selection.json", selection)
    return selection


def _final_metadata(cohort, encoder_options, mode, provenance, selection):
    from .foundations.encoders import package_versions
    return {
        "format_version": 1, "model_type": "foundation_comparison", "mode": mode,
        "encoder_options": asdict(encoder_options), "encoder_provenance": provenance,
        "input_shape": list(cohort.X.shape[1:]), "channel_names": cohort.provenance["channel_names"],
        "label_mapping": cohort.provenance["label_mapping"], "training_ids": cohort.ids.tolist(),
        "decision_threshold": 0., "raw_decision_threshold": saved_raw_threshold(selection),
        "decision_rule": selection["decision_rule"], "threshold_selection": selection,
        "package_versions": package_versions(),
        "note": "Use outer CV outputs to evaluate the training procedure. Neural checkpoints retain a reserved validation set.",
    }


def evaluate_frozen(cohort, plan, args, destination, encoder_options):
    from .foundations.encoders import extract_features, load_encoder
    X, y, ids = cohort.X, cohort.y, cohort.ids
    encoder = load_encoder(encoder_options, args.cache_dir)
    provenance = encoder.provenance
    progress = not args.no_progress
    progress_message(f"  Extracting {encoder_options.model} frozen features once...", enabled=progress)
    Z = extract_features(encoder, X, args.batch_size, progress=progress, progress_position=2)
    del encoder
    gc.collect()
    np.save(destination / "features.npy", Z, allow_pickle=False)
    write_json(destination / "encoder_provenance.json", provenance)
    write_json(destination / "feature_cache.json", {
        "shape": list(Z.shape), "dtype": str(Z.dtype), "sha256": sha256_file(destination / "features.npy"),
        "scope": "Fixed pretrained features only; no scaler or cohort adaptation cached across folds.",
    })
    predictions, folds = [], []
    oof, coverage = np.full((args.repeats, len(y)), np.nan), np.zeros((args.repeats, len(y)), int)
    for split in progress_bar(plan, desc=f"{encoder_options.model} frozen CV", unit="fold",
                              enabled=progress, position=1, leave=True):
        repeat, fold = split["repeat"], split["fold"]
        train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
        inner = [(np.array(s["train_positions"]), np.array(s["validation_positions"])) for s in split["inner_splits"]]
        folder = destination / f"repeat_{repeat:02d}_fold_{fold:02d}"
        folder.mkdir()
        with progress_bar(inner, desc="Ridge inner CV", unit="fold",
                          enabled=progress, position=2) as inner_bar:
            alpha, search, inner_scores = tune_ridge(Z[train], y[train], args.alphas, inner_bar,
                                                    decision_rule=args.decision_rule)
        selection = save_threshold_selection(folder, y[train], inner_scores, inner, args.decision_rule)
        scaler, training, testing = scaled_features(Z, train, test)
        classifier = fit_ridge(training, y[train], alpha)
        raw_scores = classifier.decision_function(testing)
        write_csv(folder / "alpha_search.csv", search)
        _, inner_rows = decision_prediction_rows(ids[train], y[train], inner_scores, selection)
        write_csv(folder / "inner_oof_predictions.csv", inner_rows)
        scores, rows = decision_prediction_rows(ids[test], y[test], raw_scores, selection, repeat=repeat, fold=fold)
        result = {"repeat": repeat, "fold": fold, "selected_alpha": alpha,
                  "decision_rule": args.decision_rule, "raw_decision_threshold": selection["raw_decision_threshold"],
                  **metrics(y[test], scores)}
        write_json(folder / "metrics_raw_zero.json", metrics(y[test], raw_scores))
        write_csv(folder / "test_predictions.csv", rows)
        write_json(folder / "metrics.json", result)
        if args.save_fold_models:
            joblib.dump({"scaler": scaler, "classifier": classifier, "threshold_selection": selection,
                         "raw_decision_threshold": selection["raw_decision_threshold"],
                         "decision_rule": args.decision_rule}, folder / "head.joblib", compress=3)
        predictions.extend(rows)
        folds.append(result)
        oof[repeat - 1, test] = scores
        coverage[repeat - 1, test] += 1
        report_fold(encoder_options.model, "frozen", repeat, fold, result, scores, folder, progress,
                    raw_scores=raw_scores, raw_threshold=selection["raw_decision_threshold"])
    summary = _write_results(destination, y, oof, coverage, predictions, folds)
    if args.fit_final:
        inner = make_inner_splits(y, args.inner_folds, derive_seed(args.seed, args.repeats, args.outer_folds))
        write_json(destination / "final_inner_splits.json", [{"train_ids": ids[a].tolist(), "validation_ids": ids[b].tolist()} for a, b in inner])
        with progress_bar(inner, desc="Final ridge inner CV", unit="fold",
                          enabled=progress, position=2) as inner_bar:
            alpha, search, scores = tune_ridge(Z, y, args.alphas, inner_bar, decision_rule=args.decision_rule)
        selection = select_threshold(y, scores, validation_fold_ids(len(y), inner), rule=args.decision_rule)
        write_json(destination / "final_threshold_selection.json", selection)
        write_csv(destination / "final_alpha_search.csv", search)
        _, rows = decision_prediction_rows(ids, y, scores, selection)
        write_csv(destination / "final_inner_oof_predictions.csv", rows)
        scaler = StandardScaler()
        features = np.ascontiguousarray(scaler.fit_transform(Z), dtype=np.float64)
        classifier = fit_ridge(features, y, alpha)
        bundle = _final_metadata(cohort, encoder_options, "frozen", provenance, selection)
        bundle.update(scaler=scaler, classifier=classifier, feature_dimension=Z.shape[1], selected_alpha=alpha)
        joblib.dump(bundle, destination / "final_model.joblib", compress=3)
    return summary


def select_finetune(X, y, ids, inner, encoder_options, args, seed, destination):
    """Tune LR within fitting participants; each inner model owns its threshold."""
    from .foundations.training import TrainOptions, fit_classifier
    settings = TrainOptions(
        head_learning_rate=args.head_learning_rate, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps, dropout=args.dropout, decision_rule=args.decision_rule,
    )
    inner = list(inner)
    validation_fold_ids(len(y), inner)
    candidates, details, predictions, fold_selections = [], [], {}, {}
    progress = not args.no_progress
    for index, learning_rate in enumerate(args.learning_rates):
        options = replace(settings, learning_rate=learning_rate)
        rows, best_epochs, scores, zero_scores, selections = [], [], [], [], []
        for fold, (train, valid) in enumerate(progress_bar(
            inner, desc=f"LR={learning_rate:g} inner CV", unit="fold", enabled=progress, position=2,
        )):
            _, result = fit_classifier(
                X[train], y[train], encoder_options, options, derive_seed(seed, fold), args.cache_dir,
                validation=(X[valid], y[valid]), retain_model=False, verbose=True,
                progress=progress, description=f"Inner {fold + 1}: LR={learning_rate:g}",
            )
            selection = result["best_threshold_selection"]
            best_epochs.append(result["best_epoch"])
            scores.append(result["best_validation_balanced_accuracy"])
            zero_scores.append(selection["inner_mean_balanced_accuracy_zero"])
            _, fold_rows = decision_prediction_rows(ids[valid], y[valid], result["best_validation_scores"],
                                                     selection, inner_fold=fold + 1)
            rows.extend(fold_rows)
            selections.append({"inner_fold": fold + 1, "best_epoch": result["best_epoch"],
                               "train_ids": ids[train].tolist(), "validation_ids": ids[valid].tolist(),
                               "threshold_selection": selection})
            details.extend({"candidate": index + 1, "learning_rate": learning_rate,
                            "head_learning_rate": result["train_options"]["head_learning_rate"],
                            "feature_dimension": result["feature_dimension"], "inner_fold": fold + 1,
                            **r} for r in result["history"])
            gc.collect()
        candidate = {
            "candidate": index + 1, "learning_rate": learning_rate, "decision_rule": args.decision_rule,
            "inner_mean_balanced_accuracy": float(np.mean(scores)),
            "inner_std_balanced_accuracy": float(np.std(scores, ddof=1)),
            "inner_mean_balanced_accuracy_zero": float(np.mean(zero_scores)),
            "best_inner_epochs": best_epochs,
            "threshold_scope": "Each inner model uses its own validation threshold; no cutoff transfers to a new model.",
            "head_learning_rate": result["train_options"]["head_learning_rate"],
            "head_learning_rate_policy": result["head_learning_rate_policy"],
            "feature_dimension": result["feature_dimension"],
        }
        candidates.append(candidate)
        predictions[index + 1], fold_selections[index + 1] = rows, selections
    best_score = max(r["inner_mean_balanced_accuracy"] for r in candidates)
    selected = min((r for r in candidates if abs(r["inner_mean_balanced_accuracy"] - best_score) <= 1e-12),
                   key=lambda r: r["learning_rate"])
    write_json(destination / "inner_search.json", candidates)
    write_json(destination / "selected.json", selected)
    write_json(destination / "inner_fold_selections.json", fold_selections[selected["candidate"]])
    write_csv(destination / "inner_training_history.csv", details)
    write_csv(destination / "inner_oof_predictions.csv", sorted(predictions[selected["candidate"]],
                                                               key=lambda r: r["participant_id"]))
    return replace(settings, learning_rate=selected["learning_rate"])


def _fit_checkpoint(cohort, split, encoder_options, args, destination, description):
    """Keep the outer/final validation set outside LR search and weight updates."""
    from .foundations.training import fit_classifier
    fit, valid = np.array(split["fit_indices"]), np.array(split["validation_indices"])
    inner = [(np.array(s["train_positions"]), np.array(s["validation_positions"])) for s in split["inner_splits"]]
    write_json(destination / "selection_split.json", split)
    options = select_finetune(cohort.X[fit], cohort.y[fit], cohort.ids[fit], inner,
                              encoder_options, args, split["split_seed"], destination)
    progress_message(f"  {description}: fitting={len(fit)}, reserved validation={len(valid)}; "
                     "restore the best validation checkpoint and calibrate that checkpoint.", enabled=not args.no_progress)
    model, training = fit_classifier(
        cohort.X[fit], cohort.y[fit], encoder_options, options, split["split_seed"], args.cache_dir,
        validation=(cohort.X[valid], cohort.y[valid]), retain_model=True, verbose=True,
        progress=not args.no_progress, description=description,
    )
    if not training["best_checkpoint_restored"]:
        raise RuntimeError("Refusing to evaluate a model whose best checkpoint was not restored.")
    selection = {**training["best_threshold_selection"],
                 "selection_scope": "Reserved validation participants excluded from fitting and LR search; scores from the restored best checkpoint. No outer-test selection.",
                 "training_protocol": FINETUNE_PROTOCOL, "checkpoint_epoch": training["best_epoch"],
                 "validation_ids": cohort.ids[valid].tolist()}
    write_json(destination / "threshold_selection.json", selection)
    _, rows = decision_prediction_rows(cohort.ids[valid], cohort.y[valid], training["best_validation_scores"], selection)
    write_csv(destination / "validation_predictions.csv", rows)
    write_csv(destination / "training_history.csv", training["history"])
    config = {key: training[key] for key in (
        "train_options", "requested_head_learning_rate", "head_learning_rate_policy", "feature_dimension",
        "training_class_counts", "training_class_weights", "best_epoch", "epochs_trained", "best_checkpoint_restored",
    )}
    config.update(training_protocol=FINETUNE_PROTOCOL, fit_ids=cohort.ids[fit].tolist(),
                  validation_ids=cohort.ids[valid].tolist(),
                  validation_class_counts={"pd": int(np.sum(cohort.y[valid] == 0)),
                                           "et": int(np.sum(cohort.y[valid] == 1))})
    write_json(destination / "training_configuration.json", config)
    write_json(destination / "encoder_provenance.json", training["encoder_provenance"])
    return model, options, training, selection


def _save_finetuned(model, cohort, encoder_options, options, destination, training, split, *, threshold_selection):
    import torch
    from .foundations.training import resolve_head_learning_rate
    if not training["best_checkpoint_restored"] or threshold_selection["checkpoint_epoch"] != training["best_epoch"]:
        raise ValueError("Only the restored checkpoint and its own threshold can be saved.")
    if threshold_selection["decision_rule"] != options.decision_rule:
        raise ValueError("Saved threshold selection and training decision rules differ.")
    metadata = _final_metadata(cohort, encoder_options, "finetune", model.encoder.provenance, threshold_selection)
    metadata["training_ids"] = split["fit_ids"]
    metadata["validation_ids"] = split["validation_ids"]
    metadata["selection_pool_ids"] = split["pool_ids"]
    resolved_options = replace(options, head_learning_rate=resolve_head_learning_rate(model.feature_dim, options.head_learning_rate))
    metadata.update(feature_dimension=model.feature_dim, train_options=asdict(resolved_options),
                    training_protocol=FINETUNE_PROTOCOL, best_epoch=training["best_epoch"],
                    epochs_trained=training["epochs_trained"], best_checkpoint_restored=True,
                    requested_head_learning_rate=options.head_learning_rate,
                    head_learning_rate_policy="auto_width_scaled" if options.head_learning_rate is None else "explicit",
                    note="Validation participants select the checkpoint and its threshold but do not enter weight fitting. Evaluate the procedure with outer CV.",
                    state_dict_file="model_state.pt")
    torch.save(model.state_dict(), destination / "model_state.pt")
    metadata["state_dict_sha256"] = sha256_file(destination / "model_state.pt")
    write_json(destination / "model.json", metadata)


def evaluate_finetuned(cohort, plan, args, destination, encoder_options):
    from .foundations.training import predict_scores
    X, y, ids = cohort.X, cohort.y, cohort.ids
    predictions, folds = [], []
    oof, coverage = np.full((args.repeats, len(y)), np.nan), np.zeros((args.repeats, len(y)), int)
    progress = not args.no_progress
    for split in progress_bar(plan, desc=f"{encoder_options.model} finetune CV", unit="fold",
                              enabled=progress, position=1, leave=True):
        repeat, fold, test = split["repeat"], split["fold"], np.array(split["test_indices"])
        folder = destination / f"repeat_{repeat:02d}_fold_{fold:02d}"
        folder.mkdir()
        model, options, training, selection = _fit_checkpoint(
            cohort, split, encoder_options, args, folder, f"Outer fold {fold}: validation early stopping")
        raw_scores = predict_scores(model, X[test], args.batch_size, progress=progress,
                                    description="Outer test prediction", progress_position=2)
        scores, rows = decision_prediction_rows(ids[test], y[test], raw_scores, selection, repeat=repeat, fold=fold)
        result = {"repeat": repeat, "fold": fold, "selected_learning_rate": options.learning_rate,
                  "training_protocol": FINETUNE_PROTOCOL, "best_epoch": training["best_epoch"],
                  "epochs_trained": training["epochs_trained"], "best_checkpoint_restored": True,
                  "backbone_gradient_verified": training["backbone_gradient_verified"],
                  "decision_rule": args.decision_rule, "raw_decision_threshold": selection["raw_decision_threshold"],
                  **metrics(y[test], scores)}
        write_json(folder / "metrics_raw_zero.json", metrics(y[test], raw_scores))
        write_json(folder / "metrics.json", result)
        write_csv(folder / "test_predictions.csv", rows)
        if args.save_fold_models:
            _save_finetuned(model, cohort, encoder_options, options, folder, training, split, threshold_selection=selection)
        predictions.extend(rows)
        folds.append(result)
        oof[repeat - 1, test] = scores
        coverage[repeat - 1, test] += 1
        del model
        gc.collect()
        report_fold(encoder_options.model, "finetune", repeat, fold, result, scores, folder, progress,
                    raw_scores=raw_scores, raw_threshold=selection["raw_decision_threshold"])
    summary = _write_results(destination, y, oof, coverage, predictions, folds)
    if args.fit_final:
        folder = destination / "final"
        folder.mkdir()
        seed = derive_seed(args.seed, args.repeats, args.outer_folds)
        split = checkpoint_split(y, ids, np.arange(len(y)), args.validation_fraction, args.inner_folds, seed)
        model, options, training, selection = _fit_checkpoint(cohort, split, encoder_options, args, folder,
                                                             "Final model: validation early stopping")
        _save_finetuned(model, cohort, encoder_options, options, folder, training, split, threshold_selection=selection)
        del model
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=("all", *SPECS), default=["all"])
    p.add_argument("--mode", choices=("frozen", "finetune", "both"), default="frozen")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output", type=Path, help="Parent results folder; every invocation creates a new uniquely identified run subfolder.")
    p.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "models" / "foundation_cache")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--batch-size", type=int, default=2, help="Participants per forward pass.")
    p.add_argument("--channel-batch-size", type=int, default=None,
                   help="Channels per forward/recomputed chunk. Default: TSPulse 128, Chronos-2 24, other models 8.")
    p.add_argument("--tirex-channel-batch-size", type=int, default=None,
                   help="Override channel batching for TiRex only; may also be changed on an incomplete-model restart.")
    p.add_argument("--tirex-gradient-checkpointing", choices=("on", "off"), default=None,
                   help="Override recomputation for TiRex only. Off uses more memory. Allowed on restart.")
    p.add_argument("--model-channel-batch-size", action="append", type=channel_override,
                   help="Per-model MODEL=N channel chunk, repeatable. Overrides the global chunk; allowed on restart.")
    p.add_argument("--model-gradient-checkpointing", action="append", type=checkpoint_override,
                   help="Per-model MODEL=on|off, repeatable. Off uses more activation memory; allowed on restart.")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--alphas", nargs="+", type=float, default=np.logspace(-3, 9, 13).tolist(),
                   help="Ridge penalties. Default: powers of ten from 1e-3 through 1e9.")
    p.add_argument("--decision-rule", choices=DECISION_RULES, default=DEFAULT_DECISION_RULE,
                   help="Default: select by balanced accuracy using inner OOF scores (frozen) or the restored checkpoint's reserved validation scores (finetune). 'zero' fixes the raw cutoff at 0.")
    p.add_argument("--learning-rates", nargs="+", type=float, default=[1e-5, 3e-5])
    p.add_argument("--head-learning-rate", type=float, default=None,
                   help="Explicit head AdamW rate. Default: 1e-3 * min(1, 256 / feature_dimension).")
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--validation-fraction", type=float, default=.2,
                   help="Reserve this fraction of each fine-tuning training pool for checkpoint selection and calibration. Default: 0.2.")
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--accumulation-steps", type=int, default=4)
    checkpointing = p.add_mutually_exclusive_group()
    checkpointing.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true",
                              help="Recompute channel activations to reduce fine-tuning memory.")
    checkpointing.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false",
                              help="Keep activations to reduce fine-tuning recomputation.")
    p.set_defaults(gradient_checkpointing=None)
    p.add_argument("--fit-final", action="store_true")
    p.add_argument("--save-fold-models", action="store_true")
    p.add_argument("--baseline", type=Path, help="Optionally verify exact MR-HYDRA or frozen Mantis cohort/splits.")
    p.add_argument("--continue-on-error", action="store_true", help="Record a failed model/mode and continue others; exit remains nonzero.")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars; retain plain-text status and epoch logs.")
    p.add_argument("--check-data", action="store_true", help="Verify data and nested splits, then exit without downloading weights or training.")
    p.add_argument("--restart-incomplete", type=Path,
                   help="Restart incomplete model/mode tasks in an interrupted run, retaining completed models and original settings.")
    args = p.parse_args(argv)
    try:
        args.model_channel_batch_size = override_mapping(args.model_channel_batch_size)
        args.model_gradient_checkpointing = override_mapping(args.model_gradient_checkpointing)
    except ValueError as error:
        p.error(str(error))
    for field, legacy in (("model_channel_batch_size", args.tirex_channel_batch_size),
                          ("model_gradient_checkpointing", args.tirex_gradient_checkpointing)):
        if legacy is not None:
            mapping = getattr(args, field)
            if "tirex" in mapping and mapping["tirex"] != legacy:
                p.error("Conflicting TiRex runtime overrides.")
            mapping["tirex"] = legacy
    if args.restart_incomplete is not None:
        supplied = sys.argv[1:] if argv is None else argv
        permitted = {"--restart-incomplete", "--no-progress", "--check-data",
                     "--tirex-channel-batch-size", "--tirex-gradient-checkpointing",
                     "--model-channel-batch-size", "--model-gradient-checkpointing"}
        if any(token.startswith("--") and token.split("=", 1)[0] not in permitted for token in supplied):
            p.error("Restart restores training settings; only presentation/data checks and per-model runtime overrides can accompany --restart-incomplete.")
        args = restore_configuration(args)
    if "all" in args.models:
        if args.models != ["all"]:
            p.error("Use --models all by itself, or list individual model names.")
        args.models = list(DEFAULT_MODELS)
    if len(set(args.models)) != len(args.models):
        p.error("Duplicate model names.")
    if not 0 <= args.seed <= 2**32 - 1:
        p.error("--seed must be in [0, 2**32-1].")
    if min(args.batch_size, args.threads, args.max_epochs, args.patience,
           args.accumulation_steps, args.repeats) < 1 or min(args.outer_folds, args.inner_folds) < 2:
        p.error("Counts must be positive; outer/inner fold counts must be at least two.")
    if args.channel_batch_size is not None and args.channel_batch_size < 1:
        p.error("--channel-batch-size must be positive.")
    if args.tirex_channel_batch_size is not None and args.tirex_channel_batch_size < 1:
        p.error("--tirex-channel-batch-size must be positive.")
    if not np.isfinite(args.validation_fraction) or not 0 < args.validation_fraction < 1:
        p.error("--validation-fraction must be finite and strictly between zero and one.")
    rates = [*args.alphas, *args.learning_rates]
    if args.head_learning_rate is not None:
        rates.append(args.head_learning_rate)
    if any(not np.isfinite(v) or v <= 0 for v in rates):
        p.error("Alphas and learning rates must be finite and strictly positive.")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        p.error("Invalid weight decay or dropout.")
    args.alphas, args.learning_rates = sorted(set(args.alphas)), sorted(set(args.learning_rates))
    return args


def run(args):
    from .foundations.encoders import (ADAPTER_VERSION, EncoderOptions, package_versions, resolve_device,
                                      resolve_channel_batch_size, resolve_gradient_checkpointing)
    from .foundations.training import seed_everything
    started = time.perf_counter()
    cohort = load_cohort(args.data_root)
    plan = make_plan(cohort.y, cohort.ids, args.outer_folds, args.inner_folds, args.repeats, args.seed)
    checkpoint_plan = (make_checkpoint_plan(cohort.y, cohort.ids, plan, args.validation_fraction, args.inner_folds)
                       if args.mode in ("finetune", "both") else None)
    if args.baseline:
        validate_baseline(args.baseline, cohort, plan, args.alphas)
    restarting = args.restart_incomplete is not None
    saved_manifest = (validate_restart(args.restart_incomplete, cohort, plan, checkpoint_plan) if restarting else None)
    if args.check_data:
        print(json.dumps({"participants": len(cohort.y), "counts": cohort.provenance["condition_counts"],
                          "shape": list(cohort.X.shape), "outer_folds_total": len(plan),
                          "models": args.models, "mode": args.mode,
                          "finetune_validation_splits": len(checkpoint_plan) if checkpoint_plan is not None else 0,
                          "training_executed": False}, indent=2))
        return None
    device = resolve_device(args.device)
    output_root = (args.output or PROJECT_ROOT / "artifacts" / "foundation_comparison").expanduser().resolve()
    protected = [args.data_root.expanduser().resolve(), Path(__file__).parent.resolve(), args.cache_dir.expanduser().resolve()]
    if args.baseline:
        protected.append(args.baseline.expanduser().resolve())
    if any(output_root.is_relative_to(p) for p in protected):
        raise ValueError("Choose an output parent outside source, dataset, checkpoint cache, and baseline.")
    if args.cache_dir.expanduser().resolve().is_relative_to(args.data_root.expanduser().resolve()):
        raise ValueError("The model cache must be outside the source dataset.")
    if restarting:
        output, run_id = args.restart_incomplete, saved_manifest["run_id"]
    else:
        run_id, output = create_run_directory(output_root)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(run_id=run_id, output_root=str(output_root), output_directory=str(output))
    manifest = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "output_root": str(output_root), "output_directory": str(output),
        "python": sys.version, "platform": platform.platform(), "package_versions": package_versions(),
        "config": config, "resolved_device": device, "dataset": cohort.provenance,
        "specifications": specifications(), "source_code_sha256": _source_hashes(), "runs": [],
        "protocol": {
            "unit": "One participant; all 132 task/wrist/sensor/axis series remain in the same split.",
            "frozen": "Pretrained eval-mode features; inner-only StandardScaler and balanced ridge alpha/threshold selection.",
            "finetune": "Fresh pretrained backbone per fit; balanced BCE, common LayerNorm/Dropout/Linear head.",
            "finetune_protocol": FINETUNE_PROTOCOL,
            "validation_fraction": args.validation_fraction,
            "head_learning_rate": "Automatic 1e-3*min(1,256/feature_dimension), unless explicitly overridden; actual rates recorded per fit.",
            "recipe_scope": "Shared-classifier backbone comparison, not reproduction of model-specific published classification recipes.",
            "adapter_version": ADAPTER_VERSION,
            "decision_rule": args.decision_rule,
            "decision_threshold": "Frozen: common inner-OOF threshold per alpha. Fine-tuned: threshold from the restored checkpoint's reserved-validation scores. Explicit zero fixes the raw cutoff at zero. No outer-test selection.",
            "early_stopping": "Reserve checkpoint validation before LR search. Tune LR by mean best inner-fold BA within fitting participants. Train the outer/final model on fitting participants, restore its earliest best reserved-validation checkpoint, and use that checkpoint's own validation threshold.",
            "deprecated_procedure": "Median-epoch full-pool refit with transferred inner-model threshold is removed; no legacy training selector.",
            "chronos_groups": "12 channels per task, separate across 11 tasks and all participants.",
            "time_adapters": "Mantis/TSPulse/VQShape: linear 976->512, align_corners=False. MOMENT: masked left-padding or endpoint-preserving resize. TiRex/Chronos/TiViT: native input length. No antialias filter.",
            "modes": "Frozen ridge and fine-tuned neural heads differ; compare backbones within each mode.",
        },
    }
    previous_elapsed = 0.
    if restarting:
        current_hashes = manifest["source_code_sha256"]
        manifest = saved_manifest
        previous_elapsed = manifest.get("elapsed_seconds", 0.)
        comparison = begin_restart(output, manifest, args, current_hashes)
    else:
        current_hashes = manifest["source_code_sha256"]
        comparison = []
    write_json(output / "manifest.json", manifest)
    progress_message(f"Run ID: {run_id}\nSaving results to: {output}", enabled=not args.no_progress)
    if not restarting:
        write_json(output / "splits.json", plan)
        if checkpoint_plan is not None:
            write_json(output / "finetune_splits.json", checkpoint_plan)
        write_csv(output / "cohort.csv", cohort.rows)
    failed = False
    modes = ["frozen", "finetune"] if args.mode == "both" else [args.mode]
    try:
        tasks = [(name, mode) for name in args.models for mode in modes]
        completed = {(r["model"], r["mode"]) for r in manifest["runs"] if r["status"] == "complete"}
        tasks = [task for task in tasks if task not in completed]
        if restarting:
            progress_message(f"Restarting incomplete tasks: {tasks}; keeping completed tasks: {sorted(completed)}.",
                             enabled=not args.no_progress)
        for name, mode in progress_bar(tasks, desc="Model / mode comparisons", unit="run",
                                       enabled=not args.no_progress, position=0, leave=True):
            destination = output / name / mode
            record = next((r for r in manifest["runs"] if (r["model"], r["mode"]) == (name, mode)), None)
            if restarting:
                archived = archive_incomplete(destination)
                if record is not None:
                    previous = {k: v for k, v in record.items() if k != "previous_attempts"}
                    previous["archived_directory"] = str(archived.relative_to(output)) if archived else None
                    record.setdefault("previous_attempts", []).append(previous)
            destination.mkdir(parents=True)
            if record is None:
                record = {"model": name, "mode": mode}
                manifest["runs"].append(record)
            record.update(status="running", directory=str(destination.relative_to(output)), source_code_sha256=current_hashes)
            record.pop("error", None)
            write_json(output / "manifest.json", manifest)
            seed_everything(args.seed)
            channel_batch, checkpointing = model_runtime_settings(args, name)
            options = EncoderOptions(name, device, resolve_channel_batch_size(name, channel_batch),
                                     resolve_gradient_checkpointing(name, checkpointing))
            record["encoder_options"] = asdict(options)
            write_json(output / "manifest.json", manifest)
            progress_message(f"\n{name}: {mode}, device={device}, channel batch={options.channel_batch_size}, "
                             f"gradient checkpointing={options.gradient_checkpointing}, adapter=v{options.adapter_version}",
                             enabled=not args.no_progress)
            try:
                function = evaluate_frozen if mode == "frozen" else evaluate_finetuned
                summary = function(cohort, plan if mode == "frozen" else checkpoint_plan, args, destination, options)
                record["status"] = "complete"
                row = {"model": name, "mode": mode}
                for key in PERFORMANCE_METRICS:
                    row[key + "_pooled_repeat_mean"] = summary["across_repeats"][key]["mean"]
                    row[key + "_fold_mean"] = summary["across_outer_folds_descriptive"][key]["mean"]
                comparison.append(row)
                write_csv(output / "comparison.csv", comparison)
            except Exception as exc:
                failed = True
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                write_json(destination / "failure.json", record)
                if not args.continue_on_error:
                    raise
                print(f"FAILED {name}/{mode}: {exc}", file=sys.stderr, flush=True)
            finally:
                write_json(output / "manifest.json", manifest)
                gc.collect()
        manifest["status"] = "partial_failure" if failed else "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=previous_elapsed + time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
    print(f"Comparison artifacts: {output}", flush=True)
    if failed:
        raise RuntimeError("Some requested models failed; inspect manifest.json and per-model failure.json.")
    return output


def main(argv=None):
    args = parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        return run(args)


if __name__ == "__main__":
    main()
