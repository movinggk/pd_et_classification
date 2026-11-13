"""Inference from a self-contained fine-tuned MantisV2 bundle; no downloads."""

import argparse
import csv
from pathlib import Path

import numpy as np

from .mantis_finetune import load_finetuned, predict_scores, select_inputs
from .task_interaction import resolve_torch_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Official-preprocessed participant .npy array.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.output.exists():
        parser.error("Use a positive batch size and a new output filename.")
    device = resolve_torch_device(args.device)
    model, payload = load_finetuned(args.model, device)
    X = select_inputs(np.load(args.inputs, allow_pickle=False), payload["input_set"])
    scores = predict_scores(model, X, device=device, batch_size=args.batch_size, progress=args.progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_index", "decision_score_et", "predicted_et", "predicted_condition"])
        writer.writeheader()
        for i, score in enumerate(scores):
            writer.writerow({"case_index": i, "decision_score_et": float(score), "predicted_et": int(score > 0),
                             "predicted_condition": "Essential Tremor" if score > 0 else "Parkinson's"})
    return args.output


if __name__ == "__main__":
    print(main().resolve())
