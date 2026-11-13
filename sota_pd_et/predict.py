"""Predict from a saved final model and already-preprocessed participant arrays."""

from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help=".npy array of shape (N, 132, 976).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hydra-device", choices=("auto", "cpu", "cuda"), default=None,
                        help="Override saved HYDRA runtime policy; no retraining is performed.")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("The output file already exists; choose a new path.")
    # Load only your own model artifacts: joblib uses Python pickle serialization.
    bundle = joblib.load(args.model)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported model bundle format.")
    for package, expected in bundle["package_versions"].items():
        if version(package) != expected:
            raise RuntimeError(f"Restore the training environment: {package} must be {expected}.")
    X = np.load(args.inputs, allow_pickle=False)
    if X.ndim != 3 or list(X.shape[1:]) != bundle["input_shape"] or len(X) == 0:
        raise ValueError(f"Expected (N, {bundle['input_shape']}) with N>=1; got {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("Input contains nonfinite samples.")
    if args.hydra_device is not None:
        bundle["model"].features.set_hydra_device(args.hydra_device)
    scores = bundle["model"].decision_function(X)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "case_index", "decision_score_et", "predicted_et", "predicted_condition",
        ])
        writer.writeheader()
        for index, score in enumerate(scores):
            predicted = int(score > bundle["decision_threshold"])
            writer.writerow({"case_index": index, "decision_score_et": float(score),
                             "predicted_et": predicted,
                             "predicted_condition": "Essential Tremor" if predicted else "Parkinson's"})
    print(args.output.resolve())


if __name__ == "__main__":
    main()
