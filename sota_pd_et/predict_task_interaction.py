"""Predict using a user-trained final temporal/cross-task model bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .task_interaction import load_network, predict_logits, resolve_torch_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Official-preprocessing .npy: (N,132,976), selected (N,60,976), or (N,5,12,976).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    if args.output.exists():
        parser.error("The output file already exists; choose a new path.")
    device = resolve_torch_device(args.device)
    network, scaler, payload = load_network(args.model, device)
    X = np.load(args.inputs, allow_pickle=False)
    scores = predict_logits(network, scaler.transform(X), device, args.batch_size)
    threshold = payload["threshold"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "case_index", "raw_score_et", "decision_threshold", "decision_score_et",
            "predicted_et", "predicted_condition",
        ])
        writer.writeheader()
        for i, score in enumerate(scores):
            predicted = int(score > threshold)
            writer.writerow({"case_index": i, "raw_score_et": float(score),
                             "decision_threshold": threshold, "decision_score_et": float(score - threshold),
                             "predicted_et": predicted,
                             "predicted_condition": "Essential Tremor" if predicted else "Parkinson's"})
    return args.output


if __name__ == "__main__":
    print(main().resolve())
