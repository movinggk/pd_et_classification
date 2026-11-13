"""Exercise the shared decision rule through every actual small architecture.

All inputs and checkpoint states are generated locally. No PADS experiment or
released-model accuracy evaluation is performed. Artificial score shifts test
the cutoff contract, not the predictive quality of these random backbones.
"""

import contextlib
import csv
from dataclasses import asdict
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
import torch

from sota_pd_et import compare_foundations as comparison
from sota_pd_et import predict_foundation as inference
from sota_pd_et.data import Cohort, sha256_file
from sota_pd_et.foundations import encoders, ridge, training
from sota_pd_et.foundations.decision import DEFAULT_DECISION_RULE, saved_raw_threshold
from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.foundations.registry import SPECS
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


MODEL_COVERAGE = {}


def stable_tiny_loader(options, cache_dir=None):
    """Reproduce fixed small checkpoint weights without changing training RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(841)
        return tiny_encoder(options, cache_dir)


def imbalanced_fixture():
    # Twelve channels also exercise Chronos's production group size of 12.
    y = np.tile([0, 0, 0, 1], 6)
    X = np.random.default_rng(431).normal(size=(len(y), 12, 64)).astype(np.float32)
    time = np.linspace(0, 2 * np.pi, 64)
    X += (.4 * y[:, None, None] * np.sin(3 * time)[None, None]).astype(np.float32)
    ids = np.array([f"synthetic_{i:02d}" for i in range(len(y))])
    rows = [{"participant_id": str(pid), "target_et": int(t)} for pid, t in zip(ids, y)]
    return Cohort(X, y, ids, rows, {
        "condition_counts": {"Parkinson's": 18, "Essential Tremor": 6},
        "channel_names": [f"synthetic_channel_{i}" for i in range(12)],
        "label_mapping": {"Parkinson's": 0, "Essential Tremor": 1},
    })


class FoundationAllModelDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_reusable_defaults_match_the_comparison_runner(self):
        self.assertEqual(DEFAULT_DECISION_RULE, "inner-balanced")
        self.assertEqual(training.TrainOptions().decision_rule, DEFAULT_DECISION_RULE)
        self.assertEqual(inspect.signature(ridge.tune_alpha).parameters["decision_rule"].default,
                         DEFAULT_DECISION_RULE)
        for name in SPECS:
            with self.subTest(model=name):
                self.assertEqual(comparison.parse_args(["--models", name]).decision_rule, DEFAULT_DECISION_RULE)

    def test_missing_threshold_is_rejected_before_loading_every_model(self):
        X = imbalanced_fixture().X
        with tempfile.TemporaryDirectory() as temporary:
            for name in SPECS:
                for mode, extension in (("frozen", ".joblib"), ("finetune", ".json")):
                    with self.subTest(model=name, mode=mode):
                        metadata = {
                            "format_version": 1, "model_type": "foundation_comparison", "mode": mode,
                            "encoder_options": asdict(EncoderOptions(name)), "input_shape": [12, 64],
                            "decision_rule": "inner-balanced", "decision_threshold": 0.,
                        }
                        path = Path(temporary) / (name + extension)
                        if extension == ".json":
                            path.write_text(json.dumps(metadata))
                        else:
                            joblib.dump(metadata, path)
                        with mock.patch.object(inference, "load_encoder") as loader, \
                             self.assertRaisesRegex(ValueError, "requires raw_decision_threshold"):
                            inference.predict(path, X)
                        loader.assert_not_called()
                        MODEL_COVERAGE.setdefault(name, {})[mode + "_missing_threshold_rejected"] = True
        # Earlier adapter-3 fixed-zero artifacts remain well-defined. A tuned
        # artifact with a missing threshold is a distinct, invalid case.
        self.assertEqual(saved_raw_threshold({}), 0.)
        self.assertEqual(saved_raw_threshold({"decision_rule": "zero", "decision_threshold": 0.}), 0.)
        for value in (None, "-0.5", True, [0.], float("nan"), float("inf")):
            with self.subTest(invalid_threshold=value), self.assertRaises(ValueError):
                saved_raw_threshold({"decision_rule": "inner-balanced", "raw_decision_threshold": value})

    def test_improving_all_negative_validation_scores_select_the_best_epoch_for_every_model(self):
        cohort = imbalanced_fixture()
        train, valid = np.arange(8), np.arange(8, 16)
        weak = np.full(len(valid), -.5)
        good = np.where(cohort.y[valid], -.2, -.8)
        for name in SPECS:
            with self.subTest(model=name), \
                 mock.patch.object(training, "predict_scores", side_effect=[weak, good]):
                _, result = training.fit_classifier(
                    cohort.X[train], cohort.y[train], EncoderOptions(name, channel_batch_size=8),
                    training.TrainOptions(max_epochs=2, patience=2, batch_size=2, accumulation_steps=2),
                    71, None, validation=(cohort.X[valid], cohort.y[valid]), loader=stable_tiny_loader,
                )
                self.assertTrue(result["backbone_gradient_verified"])
                self.assertEqual(result["best_epoch"], 2)
                self.assertEqual(result["best_validation_balanced_accuracy"], 1.)
                self.assertEqual(result["history"][-1]["validation_predicted_et_zero"], 0)
                self.assertEqual(result["history"][-1]["validation_predicted_et"], 2)
                self.assertEqual(result["training_class_counts"], {"pd": 6, "et": 2})
                self.assertEqual(result["training_class_weights"], {"pd": 2 / 3, "et": 2.})
                MODEL_COVERAGE.setdefault(name, {})["all_negative_inner_epoch_selection"] = True

    def _check_saved_model_after_negative_shift(self, path, X):
        """Move every raw score below zero and preserve the learned decisions."""
        original = inference.predict(path, X, return_details=True)
        if path.suffix == ".json":
            metadata = json.loads(path.read_text())
            # Replay the exact validation participants recorded beside the
            # saved checkpoint, before any artificial score modification.
            positions = [int(pid.rsplit("_", 1)[1]) for pid in metadata["validation_ids"]]
            with (path.parent / "validation_predictions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            np.testing.assert_allclose(original["raw_score_et"][positions],
                                       [float(row["raw_score_et"]) for row in rows], rtol=1e-5, atol=2e-6)
        shift = max(1., float(np.max(original["raw_score_et"])) + 1.,
                    original["raw_decision_threshold"] + 1.)
        if path.suffix == ".json":
            metadata = json.loads(path.read_text())
            state = torch.load(path.parent / metadata["state_dict_file"], map_location="cpu", weights_only=True)
            state["head.2.bias"] -= shift
            state_path = path.parent / "negative_shift_state.pt"
            torch.save(state, state_path)
            metadata.update(state_dict_file=state_path.name, state_dict_sha256=sha256_file(state_path))
            shifted_path = path.parent / "negative_shift_model.json"
        else:
            metadata = joblib.load(path)
            metadata["classifier"].intercept_ -= shift
            shifted_path = path.parent / "negative_shift_model.joblib"
        metadata["raw_decision_threshold"] -= shift
        metadata["threshold_selection"] = {**metadata["threshold_selection"],
                                            "raw_decision_threshold": metadata["raw_decision_threshold"]}
        if path.suffix == ".json":
            shifted_path.write_text(json.dumps(metadata))
        else:
            joblib.dump(metadata, shifted_path)
        shifted = inference.predict(shifted_path, X, return_details=True)
        self.assertTrue(np.all(shifted["raw_score_et"] < 0))
        np.testing.assert_allclose(shifted["decision_score_et"], original["decision_score_et"],
                                   rtol=1e-5, atol=2e-6)
        # Account for float32 neural output rounding only at an exact tie.
        away_from_ties = np.abs(original["decision_score_et"]) > 2e-6
        np.testing.assert_array_equal(shifted["decision_score_et"][away_from_ties] > 0,
                                      original["decision_score_et"][away_from_ties] > 0)
        return {
            "all_shifted_raw_scores_negative": True,
            "saved_threshold_preserves_margins": True,
            "saved_validation_scores_replayed": path.suffix == ".json",
            "predicted_et_before_shift": int(np.sum(original["decision_score_et"] > 0)),
            "raw_zero_predicted_et_after_shift": 0,
            "synthetic_score_shift": shift,
        }

    def test_all_eight_real_architectures_use_saved_thresholds_in_both_nested_modes(self):
        cohort = imbalanced_fixture()
        for name in SPECS:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = comparison.parse_args([
                    "--models", name, "--mode", "both", "--device", "cpu", "--threads", "2", "--no-progress",
                    "--outer-folds", "2", "--inner-folds", "2", "--max-epochs", "1", "--batch-size", "2",
                    "--learning-rates", "0.00001", "--alphas", ".1", "1000", "--fit-final", "--save-fold-models",
                    "--output", str(root / "results"), "--cache-dir", str(root / "cache"),
                ])
                with mock.patch.object(comparison, "load_cohort", return_value=cohort), \
                     mock.patch.object(encoders, "load_encoder", side_effect=stable_tiny_loader), \
                     mock.patch.object(training, "load_encoder", side_effect=stable_tiny_loader), \
                     mock.patch.object(inference, "load_encoder", side_effect=stable_tiny_loader), \
                     contextlib.redirect_stdout(io.StringIO()):
                    output = comparison.run(args)
                    fine_plan = json.loads((output / "finetune_splits.json").read_text())
                    for mode in ("frozen", "finetune"):
                        destination = output / name / mode
                        folders = sorted(destination.glob("repeat_*_fold_*"))
                        self.assertEqual(len(folders), 2)
                        for folder in folders:
                            selection = json.loads((folder / "threshold_selection.json").read_text())
                            result = json.loads((folder / "metrics.json").read_text())
                            self.assertEqual(selection["inner_participants"], 12 if mode == "frozen" else 3)
                            self.assertEqual(selection["decision_rule"], "inner-balanced")
                            self.assertEqual(result["raw_decision_threshold"], selection["raw_decision_threshold"])
                            self.assertTrue((folder / "metrics_raw_zero.json").is_file())
                            if mode == "finetune":
                                config = json.loads((folder / "training_configuration.json").read_text())
                                self.assertEqual(config["train_options"]["decision_rule"], "inner-balanced")
                                self.assertEqual(config["training_class_counts"], {"pd": 7, "et": 2})
                                self.assertTrue(config["best_checkpoint_restored"])
                                self.assertEqual(selection["checkpoint_epoch"], config["best_epoch"])
                                fold_split = fine_plan[result["fold"] - 1]
                                self.assertEqual(config["fit_ids"], fold_split["fit_ids"])
                                self.assertEqual(config["validation_ids"], fold_split["validation_ids"])
                        path = destination / ("final_model.joblib" if mode == "frozen" else "final/model.json")
                        details = self._check_saved_model_after_negative_shift(path, cohort.X)
                        MODEL_COVERAGE.setdefault(name, {})[mode] = {"outer_folds_completed": len(folders), **details}


if __name__ == "__main__":
    unittest.main()
