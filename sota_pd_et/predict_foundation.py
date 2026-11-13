"""Predict from a final frozen-head bundle or a fine-tuned model.json.

Input is a .npy array of already-preprocessed (N,132,976) participant records.
Only load your own joblib artifacts. Neural states use weights_only=True.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .data import PROJECT_ROOT, sha256_file
from .foundations.encoders import ADAPTER_VERSION, EncoderOptions, extract_features, load_encoder, resolve_device
from .foundations.training import FineTunedClassifier, predict_scores
from .foundations.decision import apply_threshold, saved_raw_threshold


def predict(model_path, X, *, device="cpu", batch_size=2, channel_batch_size=None, cache_dir=None,
            progress=False, return_details=False):
    """Return ET margins after the saved raw threshold has been subtracted.

    ``return_details=True`` additionally exposes raw scores and both thresholds
    for CSV output. Current adapter-3 artifacts saved before threshold selection
    was added use their original raw threshold of zero.
    """
    model_path = Path(model_path)
    is_neural = model_path.suffix == ".json"
    metadata = json.loads(model_path.read_text()) if is_neural else joblib.load(model_path)
    if metadata.get("format_version") != 1 or metadata.get("model_type") != "foundation_comparison":
        raise ValueError("Unsupported model artifact.")
    saved_options = dict(metadata["encoder_options"])
    version = saved_options.pop("adapter_version", None)
    if type(version) is not int or version != ADAPTER_VERSION:
        raise ValueError("Only adapter version 3 model artifacts are supported. Retrain this model with the current adapter.")
    threshold = saved_raw_threshold(metadata)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or len(X) < 1 or list(X.shape[1:]) != metadata["input_shape"] or not np.isfinite(X).all():
        raise ValueError(f"Expected finite participant arrays of shape (N,{metadata['input_shape']}).")
    options = EncoderOptions(**{**saved_options, "device": resolve_device(device),
                                "channel_batch_size": channel_batch_size or metadata["encoder_options"]["channel_batch_size"]})
    encoder = load_encoder(options, cache_dir)
    if encoder.provenance.get("checkpoint_sha256") != metadata["encoder_provenance"].get("checkpoint_sha256"):
        raise ValueError("Pretrained checkpoint hashes differ from those recorded at training time.")
    if is_neural:
        if metadata["mode"] != "finetune":
            raise ValueError("Expected fine-tuned model metadata.")
        state_path = (model_path.parent / metadata["state_dict_file"]).resolve()
        if not state_path.is_relative_to(model_path.parent.resolve()):
            raise ValueError("State file escapes the model directory.")
        if sha256_file(state_path) != metadata["state_dict_sha256"]:
            raise ValueError("Fine-tuned state hash mismatch.")
        model = FineTunedClassifier(encoder, metadata["feature_dimension"], metadata["train_options"]["dropout"])
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True), strict=True)
        model.to(options.device)
        raw = predict_scores(model, X, batch_size, progress=progress)
    else:
        if metadata["mode"] != "frozen":
            raise ValueError("Expected a frozen-head bundle.")
        Z = extract_features(encoder, X, batch_size, progress=progress)
        if Z.shape[1] != metadata["feature_dimension"]:
            raise ValueError("Feature dimension differs from training.")
        raw = metadata["classifier"].decision_function(metadata["scaler"].transform(Z))
    scores = apply_threshold(raw, threshold)
    if return_details:
        return {"decision_score_et": scores, "raw_score_et": np.asarray(raw, dtype=np.float64),
                "raw_decision_threshold": float(threshold), "decision_threshold": 0.}
    return scores


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--channel-batch-size", type=int, default=None, help="Override the saved encoder channel-batch size.")
    p.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "models" / "foundation_cache")
    p.add_argument("--no-progress", action="store_true", help="Disable prediction progress bars.")
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Output exists; choose a new file.")
    if args.batch_size < 1 or (args.channel_batch_size is not None and args.channel_batch_size < 1):
        p.error("Batch sizes must be positive.")
    result = predict(args.model, np.load(args.inputs, allow_pickle=False), device=args.device,
                     batch_size=args.batch_size, channel_batch_size=args.channel_batch_size,
                     cache_dir=args.cache_dir, progress=not args.no_progress, return_details=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_index", "raw_score_et", "raw_decision_threshold",
                                                    "decision_score_et", "decision_threshold", "predicted_et", "predicted_condition"])
        writer.writeheader()
        for index, score in enumerate(result["decision_score_et"]):
            writer.writerow({"case_index": index, "raw_score_et": float(result["raw_score_et"][index]),
                             "raw_decision_threshold": result["raw_decision_threshold"], "decision_threshold": 0.,
                             "decision_score_et": float(score), "predicted_et": int(score > 0),
                             "predicted_condition": "Essential Tremor" if score > 0 else "Parkinson's"})
    print(args.output.resolve())


if __name__ == "__main__":
    main()
