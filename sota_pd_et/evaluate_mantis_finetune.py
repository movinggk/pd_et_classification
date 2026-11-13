"""Fine-tune MantisV2 with participant-level nested CV, ONLY when invoked.

Default: the 132-channel frozen experiment's inputs, third-layer combined
tokens, last active Transformer block + linear head. Initial frozen features
fit normalization inside training folds; adapted features are never shared.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits
import torch
from tqdm.auto import tqdm

from .data import DEFAULT_DATA_ROOT, PROJECT_ROOT, load_cohort, sha256_file
from .evaluate import (PERFORMANCE_METRICS, derive_seed, make_inner_splits, make_plan, metrics,
                       prediction_rows, validate_split, write_csv, write_json)
from .evaluate_transformer import read_csv
from .mantis_finetune import (EMBEDDING_DIM, MODES, FinetuneConfig, TrainingConfig,
                              fit_finetune, predict_scores, save_finetuned, select_inputs)
from .task_interaction import CHANNEL_INDICES, resolve_torch_device
from .transformer import CHECKPOINT_HASHES, MODEL_ID, REVISION, EncoderConfig, extract_features


def installed_versions():
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "mantis-tsfm", "safetensors", "threadpoolctl"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def choose_candidate(rows):
    best = max(row["inner_mean_balanced_accuracy"] for row in rows)
    tied = [row for row in rows if abs(row["inner_mean_balanced_accuracy"] - best) <= 1e-12]
    return min(tied, key=lambda r: (-r["weight_decay"], r["epochs"], r["encoder_lr"]))


def planned_fits(plan, args):
    candidates = len(args.encoder_lrs) * len(args.weight_decays)
    count = sum(len(split["inner_splits"]) * candidates + 1 for split in plan)
    return count + (args.inner_folds * candidates + 1 if args.fit_final else 0)


def status(message, args):
    if args.progress:
        tqdm.write(message)
    else:
        print(message, flush=True)


def tune(X, y, Z0, config, base_training, checkpoint_dir, inner, args, seed, *, fit_progress=None, phase="CV"):
    oof = {(lr, decay, epoch): np.full(len(y), np.nan)
           for lr in args.encoder_lrs for decay in args.weight_decays for epoch in args.epochs}
    for fold, (train, valid) in enumerate(inner):
        validate_split(train, valid, y)
        for lr in args.encoder_lrs:
            for decay in args.weight_decays:
                training = replace(base_training, encoder_lr=lr, weight_decay=decay)
                label = f"{phase} inner {fold + 1}/{len(inner)}"
                if fit_progress is not None:
                    fit_progress.set_postfix_str(f"{label}; lr={lr:g}, wd={decay:g}")
                if not args.progress:
                    print(f"    inner {fold + 1}/{len(inner)}: encoder_lr={lr:g}, weight_decay={decay:g}", flush=True)
                model, predictions, _ = fit_finetune(
                    X[train], y[train], Z0[train], config, training, checkpoint_dir,
                    epochs=max(args.epochs), seed=derive_seed(seed, fold), device=args.device,
                    validation=X[valid], checkpoint_epochs=args.epochs,
                    progress=args.progress, progress_desc=label,
                )
                if fit_progress is not None:
                    fit_progress.update(1)
                for epoch in args.epochs:
                    oof[lr, decay, epoch][valid] = predictions[epoch]
                del model
        gc.collect()
    rows = []
    for (lr, decay, epoch), scores in oof.items():
        if not np.isfinite(scores).all():
            raise RuntimeError("Each inner participant must have a finite held-out logit.")
        fold_ba = [balanced_accuracy_score(y[valid], scores[valid] > 0) for _, valid in inner]
        rows.append({"encoder_lr": float(lr), "weight_decay": float(decay), "epochs": int(epoch),
                     "inner_mean_balanced_accuracy": float(np.mean(fold_ba)),
                     "inner_std_balanced_accuracy_descriptive": float(np.std(fold_ba, ddof=1)),
                     "inner_pooled_balanced_accuracy": float(balanced_accuracy_score(y, scores > 0)),
                     "inner_pooled_roc_auc": float(roc_auc_score(y, scores))})
    selected = choose_candidate(rows)
    return selected, rows, oof[selected["encoder_lr"], selected["weight_decay"], selected["epochs"]]


def refit(X, y, Z0, config, base_training, selected, checkpoint_dir, args, seed, *, fit_progress=None, phase="Refit"):
    training = replace(base_training, encoder_lr=selected["encoder_lr"], weight_decay=selected["weight_decay"])
    if fit_progress is not None:
        fit_progress.set_postfix_str(phase)
    model, _, history = fit_finetune(
        X, y, Z0, config, training, checkpoint_dir, epochs=selected["epochs"], seed=seed, device=args.device,
        progress=args.progress, progress_desc=phase,
    )
    if fit_progress is not None:
        fit_progress.update(1)
    return model, history


def validate_comparison(directory, cohort, plan, input_set):
    """Accept only a matching completed FROZEN Mantis result, not another fit.

    Supports the original 132-channel transformer run or the three-condition
    seven-model run (root or its mantis subdirectory). Checks data, input order,
    checkpoint, cohort, every split, and one score per held-out participant.
    """
    directory = Path(directory).expanduser().resolve()
    root = directory if (directory / "manifest.json").is_file() else directory.parent
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Comparison requires a completed baseline run.")
    if json.loads((root / "splits.json").read_text()) != plan:
        raise ValueError("Comparison participant splits differ.")
    for key in ("dataset", "checksum_list_sha256", "verified_source_sha256", "label_mapping",
                "participant_count", "condition_counts", "movement_shape", "channel_names"):
        if key not in cohort.provenance or manifest.get("dataset", {}).get(key) != cohort.provenance[key]:
            raise ValueError(f"Comparison dataset provenance differs: {key}.")
    expected_cohort = [(str(pid), int(y)) for pid, y in zip(cohort.ids, cohort.y, strict=True)]
    baseline_cohort = [(r["participant_id"], int(r["target_et"])) for r in read_csv(root / "cohort.csv")]
    if expected_cohort != baseline_cohort:
        raise ValueError("Comparison cohort/order differs.")
    if "input_protocol" in manifest:
        protocol = manifest["input_protocol"]
        if (input_set != "three_tasks" or protocol.get("selected_channel_indices") != list(CHANNEL_INDICES)
                or "mantis" not in manifest.get("completed_models", [])):
            raise ValueError("Comparison task/channel selection differs.")
        prediction_file = root / "mantis" / "oof_predictions.csv"
        encoder_file = root / "mantis_encoder_provenance.json"
    else:
        if input_set != "all":
            raise ValueError("The original comparison run uses all 132 channels.")
        prediction_file = root / "oof_predictions.csv"
        encoder_file = root / "encoder_provenance.json"
    encoder = json.loads(encoder_file.read_text())
    expected = {"model_id": MODEL_ID, "revision": REVISION, "sequence_length": 512,
                "return_transf_layer": 2, "output_token": "combined"}
    if (any(encoder.get("encoder", {}).get(k) != v for k, v in expected.items())
            or encoder.get("parameters_trainable") != 0
            or encoder.get("checkpoint", {}).get("sha256") != CHECKPOINT_HASHES
            or encoder.get("input_shape") != [len(cohort.y), 132 if input_set == "all" else 60, cohort.X.shape[-1]]):
        raise ValueError("Comparison is not the matching pinned frozen Mantis representation.")
    labels = dict(expected_cohort)
    expected_keys = {(s["repeat"], s["fold"], pid) for s in plan for pid in s["test_ids"]}
    predictions = {}
    for row in read_csv(prediction_file):
        key = (int(row["repeat"]), int(row["fold"]), row["participant_id"])
        if key not in expected_keys or key in predictions or int(row["target_et"]) != labels.get(key[2]):
            raise ValueError("Comparison predictions have duplicate/missing/wrong participants or labels.")
        raw = float(row.get("raw_score_et", row["decision_score_et"]))
        threshold = float(row.get("decision_threshold", 0))
        if not np.isfinite(raw) or threshold != 0 or int(row["predicted_et"]) != int(raw > threshold):
            raise ValueError("Comparison requires finite frozen scores and the same fixed zero threshold.")
        predictions[key] = raw
    if set(predictions) != expected_keys:
        raise ValueError("Comparison prediction coverage differs.")
    return predictions, {"directory": str(root), "input_set": input_set,
                         "prediction_sha256": sha256_file(prediction_file),
                         "manifest_sha256": sha256_file(root / "manifest.json"),
                         "validation": "Dataset, input channels, pretrained identity, cohort, splits and OOF coverage match."}


def make_comparison(rows, baseline):
    comparisons = []
    for repeat in sorted({r["repeat"] for r in rows}):
        subset = [r for r in rows if r["repeat"] == repeat]
        y = np.array([r["target_et"] for r in subset])
        before = np.array([baseline[r["repeat"], r["fold"], r["participant_id"]] for r in subset])
        after = np.array([r["decision_score_et"] for r in subset])
        a, b = metrics(y, before), metrics(y, after)
        comparisons.append({"repeat": repeat, "frozen": a, "finetuned": b,
                            "finetuned_minus_frozen": {k: b[k] - a[k] for k in PERFORMANCE_METRICS}})
    return {"repeat_comparisons": comparisons,
            "interpretation": "Descriptive paired comparison of different training procedures; no statistical superiority or external validity is established."}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-set", choices=("all", "three_tasks"), default="all")
    parser.add_argument("--mode", choices=MODES, default="last-block")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--reference-splits", type=Path)
    parser.add_argument("--compare-to", type=Path, help="Completed matching frozen-Mantis run, optionally a three-task run root.")
    parser.add_argument("--epochs", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--encoder-lrs", type=float, nargs="+", default=[1e-5, 3e-5])
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.01])
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2, help="Participants per microbatch.")
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--channel-batch-size", type=int, default=16, help="Independent channel records per encoder call.")
    parser.add_argument("--encoder-batch-size", type=int, default=128, help="Initial FROZEN feature extraction only.")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Show extraction, model-fit, epoch and participant progress bars (default on).")
    parser.add_argument("--save-outer-models", action="store_true")
    parser.add_argument("--fit-final", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32 or not 1 <= args.threads <= (os.cpu_count() or 1):
        parser.error("Invalid seed or CPU thread count.")
    if min(args.outer_folds, args.inner_folds) < 2 or args.repeats < 1:
        parser.error("Need at least two outer/inner folds and one repeat.")
    if min(args.epochs) < 1 or min(args.encoder_batch_size, args.channel_batch_size) < 1:
        parser.error("Epochs and encoder/channel batch sizes must be positive.")
    if any(not np.isfinite(v) or v <= 0 for v in args.encoder_lrs):
        parser.error("Encoder learning-rate candidates must be finite and positive.")
    if any(not np.isfinite(v) or v < 0 for v in args.weight_decays):
        parser.error("Weight decay candidates must be finite and nonnegative.")
    if args.mode != "head-only" and min(args.epochs) <= args.head_warmup_epochs:
        parser.error("Every candidate epoch must include at least one epoch after head-only warmup.")
    args.epochs = sorted(set(args.epochs))
    args.encoder_lrs = [0.0] if args.mode == "head-only" else sorted(set(args.encoder_lrs))
    args.weight_decays = sorted(set(args.weight_decays))
    try:
        FinetuneConfig(head_dropout=args.head_dropout, mode=args.mode, channel_batch_size=args.channel_batch_size)
        TrainingConfig(encoder_lr=args.encoder_lrs[0], head_lr=args.head_lr, weight_decay=args.weight_decays[0],
                       batch_size=args.batch_size, accumulation_steps=args.accumulation_steps, warmup_epochs=args.head_warmup_epochs)
    except ValueError as error:
        parser.error(str(error))
    return args


def run(args):
    started = time.perf_counter()
    args.device = resolve_torch_device(args.device)
    torch.set_num_threads(args.threads)
    cohort = load_cohort(args.data_root)
    X, y, ids = select_inputs(cohort.X, args.input_set), cohort.y, cohort.ids
    plan = make_plan(y, ids, args.outer_folds, args.inner_folds, args.repeats, args.seed)
    reference = args.reference_splits
    automatic = PROJECT_ROOT / "artifacts/pads_mantis_v2/splits.json"
    if reference is None and automatic.is_file() and (args.outer_folds, args.inner_folds, args.repeats, args.seed) == (5, 3, 1, 20261017):
        reference = automatic
    if reference is not None and json.loads(reference.read_text()) != plan:
        raise ValueError("Reference participant splits differ.")
    baseline, baseline_provenance = (validate_comparison(args.compare_to, cohort, plan, args.input_set)
                                     if args.compare_to is not None else (None, None))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or PROJECT_ROOT / "artifacts/pads_mantis_v2_finetuned" / stamp).expanduser().resolve()
    for protected in (args.data_root.expanduser().resolve(), Path(__file__).parent.resolve(), PROJECT_ROOT / "models"):
        if output.is_relative_to(protected.resolve()):
            raise ValueError("Choose an output outside data, source code and original pretrained checkpoints.")
    if any((parent / "manifest.json").is_file() for parent in output.parents):
        raise ValueError("Do not write inside an existing experiment.")
    output.mkdir(parents=True, exist_ok=False)
    config = FinetuneConfig(channels=X.shape[1], samples=X.shape[2], mode=args.mode,
                            head_dropout=args.head_dropout, channel_batch_size=args.channel_batch_size,
                            activation_checkpointing=args.activation_checkpointing)
    base_training = TrainingConfig(encoder_lr=args.encoder_lrs[0], head_lr=args.head_lr,
                                   weight_decay=args.weight_decays[0], batch_size=args.batch_size,
                                   accumulation_steps=args.accumulation_steps, warmup_epochs=args.head_warmup_epochs)
    sources = sorted(Path(__file__).parent.rglob("*.py"))
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": sys.argv, "python": sys.version, "platform": platform.platform(),
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "model": asdict(config), "training": asdict(base_training), "package_versions": installed_versions(),
                "dataset": cohort.provenance, "input_set": args.input_set, "input_shape": list(X.shape),
                "source_code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sources},
                "reference_splits": {"path": str(reference), "sha256": sha256_file(reference)} if reference else None,
                "comparison_baseline": baseline_provenance,
                "protocol": {"unit": "One participant; all channels/tasks stay in the same fold",
                             "initial_cache": "Externally pretrained frozen features ONLY; no adapted embeddings cached across folds",
                             "normalization": "Initial-feature means/std fitted on TRAINING participants only, fixed during adaptation",
                             "initialization": "Fresh pinned pretrained backbone and fresh head per inner/outer/final fit",
                             "selection": "Mean inner-fold BA; ties: stronger decay, fewer epochs, lower encoder LR",
                             "outer_refit": "Fresh model, fixed selected epochs; no outer-test early stopping",
                             "class_weight": "n_train/(2*n_class), recomputed in each fit",
                             "decision": "ET iff uncalibrated participant logit > 0; probabilities are not reported"}}
    write_json(output / "manifest.json", manifest)
    all_rows, fold_metrics = [], []
    oof = np.full((args.repeats, len(y)), np.nan)
    coverage = np.zeros_like(oof, dtype=int)
    fit_progress = None
    try:
        write_csv(output / "cohort.csv", cohort.rows)
        write_json(output / "splits.json", plan)
        with zipfile.ZipFile(output / "evaluated_sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for source in sources:
                archive.write(source, str(source.relative_to(PROJECT_ROOT)))
        checkpoint_dir = output / "initial_checkpoint"
        Z0, initial_provenance = extract_features(
            X, EncoderConfig(device=args.device, batch_size=args.encoder_batch_size), checkpoint_dir,
            progress=args.progress,
        )
        if Z0.shape != (len(y), X.shape[1] * EMBEDDING_DIM) or not np.isfinite(Z0).all():
            raise RuntimeError("Initial frozen features do not match participants/channels.")
        np.save(output / "initial_frozen_features.npy", Z0, allow_pickle=False)
        initial_provenance["feature_sha256"] = sha256_file(output / "initial_frozen_features.npy")
        write_json(output / "initial_encoder_provenance.json", initial_provenance)
        fit_progress = tqdm(total=planned_fits(plan, args), desc="Model fits", unit="fit",
                            position=0, dynamic_ncols=True, disable=not args.progress)
        for split in plan:
            repeat, fold = split["repeat"], split["fold"]
            train, test = np.array(split["train_indices"]), np.array(split["test_indices"])
            inner = [(np.array(s["train_positions"]), np.array(s["validation_positions"])) for s in split["inner_splits"]]
            destination = output / f"repeat_{repeat:02d}_fold_{fold:02d}"
            destination.mkdir()
            manifest["current"] = {"repeat": repeat, "fold": fold}
            write_json(output / "manifest.json", manifest)
            status(f"Repeat {repeat}, outer fold {fold}: {args.mode}, {len(train)} training participants", args)
            fold_started = time.perf_counter()
            with threadpool_limits(limits=args.threads):
                selected, search, inner_scores = tune(X[train], y[train], Z0[train], config, base_training, checkpoint_dir,
                                                      inner, args, split["model_seed"], fit_progress=fit_progress,
                                                      phase=f"R{repeat}/F{fold}")
                model, history = refit(X[train], y[train], Z0[train], config, base_training, selected, checkpoint_dir, args,
                                       split["model_seed"], fit_progress=fit_progress, phase=f"R{repeat}/F{fold} refit")
                scores = predict_scores(model, X[test], device=args.device, batch_size=args.batch_size,
                                        progress=args.progress, progress_desc=f"R{repeat}/F{fold} test", progress_position=1)
            manifest["parameter_counts"] = model.parameter_counts()
            result = {"repeat": repeat, "fold": fold, "elapsed_seconds": time.perf_counter() - fold_started,
                      **metrics(y[test], scores)}
            rows = prediction_rows(ids[test], y[test], scores, repeat=repeat, fold=fold)
            write_json(destination / "selected.json", selected)
            write_csv(destination / "inner_search.csv", search)
            write_csv(destination / "inner_oof_predictions.csv", prediction_rows(ids[train], y[train], inner_scores))
            write_csv(destination / "training_history.csv", history)
            write_csv(destination / "test_predictions.csv", rows)
            write_json(destination / "metrics.json", result)
            if args.save_outer_models:
                save_finetuned(destination / "model.pt", model, input_set=args.input_set,
                               metadata={"training_ids": ids[train].tolist(), "selected": selected,
                                         "package_versions": manifest["package_versions"], "source_code_sha256": manifest["source_code_sha256"]})
            fold_metrics.append(result); all_rows.extend(rows)
            oof[repeat - 1, test] = scores; coverage[repeat - 1, test] += 1
            status(f"  outer BA={result['balanced_accuracy']:.4f}, ET={result['et_correct']}/{result['n_et']}", args)
            del model
            gc.collect()
        if not np.all(coverage == 1) or not np.isfinite(oof).all():
            raise RuntimeError("Each participant requires exactly one finite outer prediction per repeat.")
        repeat_metrics = [{"repeat": r + 1, **metrics(y, oof[r])} for r in range(args.repeats)]
        summary = {"primary_metric": "balanced_accuracy", "positive_class": "ET=1; PD=0", "input_set": args.input_set,
                   "repeat_metrics": repeat_metrics, "outer_fold_metrics": fold_metrics,
                   "mean_outer_fold_roc_auc": float(np.mean([r["roc_auc"] for r in fold_metrics])),
                   "mean_outer_fold_average_precision_et": float(np.mean([r["average_precision_et"] for r in fold_metrics])),
                   "always_pd_baseline": metrics(y, -np.ones(len(y))),
                   "interpretation": ["Only outer held-out participants contribute to evaluation.",
                                      "Initial head-only warmup epochs count toward the selected epoch budget.",
                                      "Pooled AUROC/AP mix logits from separate fits; inspect fold-level metrics.",
                                      "Fold/repeat variation is descriptive. Repeats do not increase independent ET sample size.",
                                      "Follow-up on an inspected cohort is exploratory; no external clinical validity is established."]}
        write_json(output / "summary.json", summary)
        write_csv(output / "oof_predictions.csv", sorted(all_rows, key=lambda r: (r["repeat"], r["participant_id"])))
        write_csv(output / "fold_metrics.csv", [{k: v for k, v in r.items() if not isinstance(v, list)} for r in fold_metrics])
        if baseline is not None:
            write_json(output / "frozen_comparison.json", make_comparison(all_rows, baseline))
        if args.fit_final:
            final_seed = derive_seed(args.seed, args.repeats, args.outer_folds)
            inner = make_inner_splits(y, args.inner_folds, final_seed)
            write_json(output / "final_inner_splits.json", [{"train_ids": ids[a].tolist(), "validation_ids": ids[b].tolist()} for a, b in inner])
            with threadpool_limits(limits=args.threads):
                selected, search, final_scores = tune(X, y, Z0, config, base_training, checkpoint_dir, inner, args, final_seed,
                                                      fit_progress=fit_progress, phase="Final selection")
                model, history = refit(X, y, Z0, config, base_training, selected, checkpoint_dir, args, final_seed,
                                       fit_progress=fit_progress, phase="Final refit")
            write_json(output / "final_selected.json", selected)
            write_csv(output / "final_inner_search.csv", search)
            write_csv(output / "final_inner_oof_predictions.csv", prediction_rows(ids, y, final_scores))
            write_csv(output / "final_training_history.csv", history)
            save_finetuned(output / "final_model.pt", model, input_set=args.input_set,
                           metadata={"training_ids": ids.tolist(), "selected": selected,
                                     "package_versions": manifest["package_versions"], "source_code_sha256": manifest["source_code_sha256"],
                                     "note": "Full-cohort model; training performance is not held-out evaluation."})
        lines = ["# Fine-tuned MantisV2: participant-level PADS evaluation", "",
                 f"Input: {args.input_set}; mode: {args.mode}; encoder: pinned layer 2 / combined tokens.", "",
                 "| Repeat | BA | ET sensitivity | PD specificity | Pooled AUROC | ET AP |",
                 "|---|---:|---:|---:|---:|---:|"]
        for r in repeat_metrics:
            lines.append(f"| {r['repeat']} | {r['balanced_accuracy']:.4f} | {r['et_recall']:.4f} | {r['pd_recall']:.4f} | {r['roc_auc']:.4f} | {r['average_precision_et']:.4f} |")
        lines += ["", f"Mean outer-fold AUROC: {summary['mean_outer_fold_roc_auc']:.4f}.", "",
                  "Scores are uncalibrated logits. The zero threshold was fixed before evaluation.",
                  "Initial frozen features fit training-only normalization; fine-tuned features were not reused between folds.",
                  "The optional full-cohort final model has no new independent test set.",
                  "This is an exploratory single-cohort comparison, not external clinical validation."]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")
        manifest.pop("current", None)
        manifest.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        status(f"Saved fine-tuning run: {output}", args)
    except BaseException as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error)}, elapsed_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if fit_progress is not None:
            fit_progress.close()
    return output


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
