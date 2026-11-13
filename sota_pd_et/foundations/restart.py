"""Restart incomplete model/mode tasks without touching completed results."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .validation import FINETUNE_PROTOCOL


def restore_configuration(args):
    directory = args.restart_incomplete.expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") not in ("failed", "partial_failure", "interrupted"):
        raise ValueError("Restart requires a failed/interrupted run; a running or completed run is not restarted.")
    config = manifest["config"]
    if config["mode"] in ("finetune", "both") and manifest["protocol"].get("finetune_protocol") != FINETUNE_PROTOCOL:
        raise ValueError("Cannot restart a run using a deprecated fine-tuning protocol.")
    if manifest["protocol"].get("adapter_version") != 3:
        raise ValueError("Restart requires adapter version 3.")
    no_progress, check_data = args.no_progress, args.check_data
    runtime = {key: getattr(args, key) for key in ("tirex_channel_batch_size", "tirex_gradient_checkpointing")
               if getattr(args, key) is not None}
    per_model = {key: getattr(args, key).copy() for key in
                 ("model_channel_batch_size", "model_gradient_checkpointing")}
    for key in vars(args):
        if key in config and key not in ("restart_incomplete", "no_progress", "check_data"):
            value = config[key]
            if key in ("data_root", "output", "cache_dir", "baseline") and value is not None:
                value = Path(value)
            setattr(args, key, value)
    args.restart_incomplete = directory
    args.no_progress = no_progress or config.get("no_progress", False)
    args.check_data = check_data
    for key, value in runtime.items():
        setattr(args, key, value)
    for key, value in per_model.items():
        setattr(args, key, {**config.get(key, {}), **value})
    return args


def validate_restart(directory, cohort, plan, checkpoint_plan):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") not in ("failed", "partial_failure", "interrupted"):
        raise ValueError("Run status changed; refusing to restart an active or completed run.")
    if manifest["dataset"] != cohort.provenance:
        raise ValueError("Dataset provenance differs from the interrupted run.")
    if json.loads((directory / "splits.json").read_text()) != plan:
        raise ValueError("Outer/frozen-inner splits differ from the interrupted run.")
    if checkpoint_plan is not None and json.loads((directory / "finetune_splits.json").read_text()) != checkpoint_plan:
        raise ValueError("Fine-tuning fitting/validation splits differ from the interrupted run.")
    for record in manifest["runs"]:
        if record["status"] == "complete":
            expected = directory / record["model"] / record["mode"] / "summary.json"
            if not expected.is_file():
                raise ValueError(f"Completed model has no summary: {expected}")
    return manifest


def begin_restart(directory, manifest, args, source_hashes):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = directory / "restart_history" / f"manifest_before_{stamp}.json"
    backup.parent.mkdir(exist_ok=True)
    backup.write_bytes((directory / "manifest.json").read_bytes())
    manifest.setdefault("restart_history", []).append({
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_manifest": str(backup.relative_to(directory)),
        "source_code_sha256": source_hashes,
        "effective_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "tirex_runtime_overrides": {"channel_batch_size": args.tirex_channel_batch_size,
                                   "gradient_checkpointing": args.tirex_gradient_checkpointing},
        "model_runtime_overrides": {"channel_batch_size": args.model_channel_batch_size,
                                   "gradient_checkpointing": args.model_gradient_checkpointing},
        "policy": "Completed model/mode directories untouched; each incomplete model restarts from its first fold.",
    })
    manifest["status"] = "running"
    for key in ("error", "finished_at_utc"):
        manifest.pop(key, None)
    rows = []
    table = directory / "comparison.csv"
    if table.exists():
        with table.open() as stream:
            rows = list(csv.DictReader(stream))
    completed = {(r["model"], r["mode"]) for r in manifest["runs"] if r["status"] == "complete"}
    rows = [row for row in rows if (row["model"], row["mode"]) in completed]
    return rows


def archive_incomplete(directory):
    if not directory.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = directory.with_name(f"{directory.name}_interrupted_{stamp}_{uuid.uuid4().hex[:6]}")
    directory.rename(archived)
    return archived
