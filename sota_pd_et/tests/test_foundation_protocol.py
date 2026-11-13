"""Participant leakage, nested fine-tuning selection, and saved inference tests."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
import torch
from torch import nn

from sota_pd_et import compare_foundations as comparison
from sota_pd_et import predict_foundation as inference
from sota_pd_et.data import Cohort
from sota_pd_et.foundations import encoders, training
from sota_pd_et.foundations.encoders import EncoderOptions


class ToyEncoder(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.backbone = nn.Linear(4, 6)
        self.options = options
        self.provenance = {"test_only": True, "checkpoint_sha256": {"fixture": "fixed"}}

    def forward(self, x):
        return self.backbone(x.mean(-1))


def toy_loader(options, cache_dir=None):
    # A checkpoint loader must return the same pretrained weights on every
    # call, while leaving the training RNG stream alone.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(441)
        return ToyEncoder(options)


def fixture():
    y = np.tile([0, 1], 8)
    X = np.random.default_rng(33).normal(size=(len(y), 4, 64)).astype(np.float32)
    # Row identities can be recovered exactly during fit/predict instrumentation.
    X[:, 0, 0] = np.arange(len(y))
    ids = np.array([f"p{i:03d}" for i in range(len(y))])
    rows = [{"participant_id": pid, "target_et": int(t), "condition": "ET" if t else "PD"} for pid, t in zip(ids, y)]
    return Cohort(X, y, ids, rows, {
        "condition_counts": {"PD": 8, "ET": 8}, "channel_names": [f"c{i}" for i in range(4)],
        "label_mapping": {"Parkinson's": 0, "Essential Tremor": 1},
    })


class FoundationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_modes_use_identical_plan_and_outer_test_never_enters_finetune(self):
        cohort = fixture()
        plan = comparison.make_plan(cohort.y, cohort.ids, 2, 2, 1, 20261017)
        fit_calls = []
        real_fit = training.fit_classifier

        def tracked(X, y, *args, **kwargs):
            train_people = set(X[:, 0, 0].astype(int))
            valid = kwargs.get("validation")
            valid_people = set(valid[0][:, 0, 0].astype(int)) if valid is not None else set()
            self.assertFalse(train_people & valid_people)
            fit_calls.append((train_people, valid_people))
            return real_fit(X, y, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = comparison.parse_args([
                "--models", "tspulse", "--mode", "both", "--device", "cpu", "--no-progress",
                "--outer-folds", "2", "--inner-folds", "2", "--max-epochs", "1",
                "--learning-rates", "0.001", "--head-learning-rate", "0.001",
                "--alphas", ".1", "1", "--fit-final", "--save-fold-models",
                "--output", str(root / "results"), "--cache-dir", str(root / "cache"),
            ])
            with mock.patch.object(comparison, "load_cohort", return_value=cohort), \
                 mock.patch.object(encoders, "load_encoder", side_effect=toy_loader), \
                 mock.patch.object(training, "load_encoder", side_effect=toy_loader), \
                 mock.patch.object(training, "fit_classifier", side_effect=tracked), \
                 contextlib.redirect_stdout(io.StringIO()):
                output = comparison.run(args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(output.parent, (root / "results").resolve())
            self.assertEqual(manifest["run_id"], output.name)
            self.assertEqual(manifest["output_directory"], str(output))
            self.assertEqual(json.loads((output / "splits.json").read_text()), plan)
            self.assertEqual(len(manifest["runs"]), 2)
            for run in manifest["runs"]:
                self.assertEqual(run["encoder_options"]["channel_batch_size"], 128)
                self.assertFalse(run["encoder_options"]["gradient_checkpointing"])
            fine_plan = json.loads((output / "finetune_splits.json").read_text())
            # Each outer fold tunes LR within fitting participants, then uses
            # its reserved validation set for the actual retained checkpoint.
            for index, split in enumerate(fine_plan):
                outer_test = set(split["test_indices"])
                fitting, validation = set(split["fit_indices"]), set(split["validation_indices"])
                calls = fit_calls[3 * index:3 * index + 3]
                for train_people, valid_people in calls:
                    self.assertFalse((train_people | valid_people) & outer_test)
                for train_people, valid_people in calls[:-1]:
                    self.assertLessEqual(train_people | valid_people, fitting)
                    self.assertFalse((train_people | valid_people) & validation)
                self.assertEqual(calls[-1], (fitting, validation))
            for mode in ("frozen", "finetune"):
                path = output / "tspulse" / mode
                with (path / "oof_predictions.csv").open() as stream:
                    import csv
                    rows = list(csv.DictReader(stream))
                self.assertCountEqual([r["participant_id"] for r in rows], cohort.ids)
                self.assertEqual(len(rows), len(cohort.y))
                for row in rows:
                    raw, threshold = float(row["raw_score_et"]), float(row["raw_decision_threshold"])
                    self.assertAlmostEqual(float(row["decision_score_et"]), raw - threshold, places=14)
                    self.assertEqual(int(row["predicted_et"]), int(raw > threshold))
                    self.assertEqual(float(row["decision_threshold"]), 0.)
                for split in plan:
                    folder = path / f"repeat_{split['repeat']:02d}_fold_{split['fold']:02d}"
                    selection = json.loads((folder / "threshold_selection.json").read_text())
                    expected = (len(split["train_indices"]) if mode == "frozen"
                                else len(fine_plan[split["fold"] - 1]["validation_indices"]))
                    self.assertEqual(selection["inner_participants"], expected)
                    self.assertEqual(selection["decision_rule"], "inner-balanced")
                    self.assertTrue((folder / "metrics_raw_zero.json").is_file())
            frozen = output / "tspulse/frozen/final_model.joblib"
            neural = output / "tspulse/finetune/final/model.json"
            with mock.patch.object(inference, "load_encoder", side_effect=toy_loader):
                scores_frozen = inference.predict(frozen, cohort.X)
                scores_neural = inference.predict(neural, cohort.X)
            self.assertEqual(scores_frozen.shape, cohort.y.shape)
            self.assertEqual(scores_neural.shape, cohort.y.shape)
            self.assertTrue(np.isfinite(scores_neural).all())
            # A saved nonzero threshold must affect both inference formats
            # exactly once, including models restored in a fresh encoder.
            shifted_frozen = frozen.parent / "shifted_threshold.joblib"
            bundle = joblib.load(frozen)
            bundle["raw_decision_threshold"] += .7
            joblib.dump(bundle, shifted_frozen)
            shifted_neural = neural.parent / "shifted_threshold.json"
            meta = json.loads(neural.read_text())
            meta["raw_decision_threshold"] += .7
            shifted_neural.write_text(json.dumps(meta))
            with mock.patch.object(inference, "load_encoder", side_effect=toy_loader):
                np.testing.assert_allclose(inference.predict(shifted_frozen, cohort.X), scores_frozen - .7, atol=1e-12)
                np.testing.assert_allclose(inference.predict(shifted_neural, cohort.X), scores_neural - .7, atol=1e-12)
                details = inference.predict(neural, cohort.X, return_details=True)
            np.testing.assert_array_equal(details["decision_score_et"], scores_neural)
            np.testing.assert_allclose(details["raw_score_et"] - details["raw_decision_threshold"], scores_neural, atol=1e-12)
            fold_meta = json.loads((output / "tspulse/finetune/repeat_01_fold_01/model.json").read_text())
            self.assertEqual(fold_meta["training_ids"], fine_plan[0]["fit_ids"])
            self.assertEqual(fold_meta["validation_ids"], fine_plan[0]["validation_ids"])
            self.assertTrue(fold_meta["best_checkpoint_restored"])
            self.assertNotIn("refit_epochs", fold_meta)
            self.assertEqual(fold_meta["best_epoch"], fold_meta["threshold_selection"]["checkpoint_epoch"])
            self.assertEqual(fold_meta["train_options"]["head_learning_rate"], .001)
            self.assertEqual(fold_meta["head_learning_rate_policy"], "explicit")
            # Unsupported feature definitions are rejected before restoring
            # weights, even though their head tensor shapes might still match.
            for version in (None, 1, 2):
                with self.subTest(unsupported_neural_version=version):
                    unsupported_meta = json.loads(neural.read_text())
                    if version is None:
                        unsupported_meta["encoder_options"].pop("adapter_version")
                    else:
                        unsupported_meta["encoder_options"]["adapter_version"] = version
                    unsupported_neural = neural.parent / "unsupported_model.json"
                    unsupported_neural.write_text(json.dumps(unsupported_meta))
                    with mock.patch.object(inference, "load_encoder") as loader, \
                         self.assertRaisesRegex(ValueError, "adapter version 3"):
                        inference.predict(unsupported_neural, cohort.X)
                    loader.assert_not_called()
            # Tampered neural state is rejected before loading its tensors.
            state_path = neural.parent / "model_state.pt"
            state_path.write_bytes(state_path.read_bytes() + b"tampered")
            with mock.patch.object(inference, "load_encoder", side_effect=toy_loader), \
                 self.assertRaisesRegex(ValueError, "hash mismatch"):
                inference.predict(neural, cohort.X)

    def test_early_stopping_returns_best_epoch_predictions(self):
        cohort = fixture()
        train, valid = np.arange(8), np.arange(8, 16)
        correct = np.where(cohort.y[valid], 1., -1.)
        wrong = -correct
        options = training.TrainOptions(max_epochs=3, patience=1, batch_size=4)
        with mock.patch.object(training, "predict_scores", side_effect=[correct, wrong]):
            _, result = training.fit_classifier(
                cohort.X[train], cohort.y[train], EncoderOptions("tspulse"), options, 4, None,
                validation=(cohort.X[valid], cohort.y[valid]), loader=toy_loader,
            )
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["epochs_trained"], 2)
        np.testing.assert_array_equal(result["best_validation_scores"], correct)

    def test_every_finetune_fit_starts_from_fresh_weights_and_is_reproducible(self):
        cohort = fixture()
        options = training.TrainOptions(max_epochs=1, batch_size=4)
        models = []
        for _ in range(2):
            model, result = training.fit_classifier(cohort.X, cohort.y, EncoderOptions("tspulse"),
                                                     options, 71, None, epochs=1, loader=toy_loader)
            models.append(model)
            self.assertTrue(result["backbone_gradient_verified"])
        self.assertIsNot(models[0].encoder, models[1].encoder)
        for left, right in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(left, right)

    def test_detached_backbone_is_rejected(self):
        class Detached(ToyEncoder):
            def forward(self, x):
                return super().forward(x).detach()
        cohort = fixture()
        with self.assertRaisesRegex(RuntimeError, "backbone gradients"):
            training.fit_classifier(cohort.X, cohort.y, EncoderOptions("tspulse"), training.TrainOptions(),
                                    1, None, epochs=1, loader=lambda opts, cache: Detached(opts))

    def test_parser_rejects_ambiguous_or_invalid_runs(self):
        for argv in (["--models", "all", "tspulse"], ["--learning-rates", "nan"],
                     ["--validation-fraction", "0"], ["--validation-fraction", "1"], ["--validation-fraction", "nan"],
                     ["--outer-folds", "1"], ["--batch-size", "0"], ["--dropout", "1"],
                     ["--decision-rule", "unknown"],
                     ["--gradient-checkpointing", "--no-gradient-checkpointing"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                comparison.parse_args(argv)
        self.assertIsNone(comparison.parse_args([]).gradient_checkpointing)
        self.assertTrue(comparison.parse_args(["--gradient-checkpointing"]).gradient_checkpointing)
        self.assertFalse(comparison.parse_args(["--no-gradient-checkpointing"]).gradient_checkpointing)

    def test_check_data_does_not_load_weights_or_train(self):
        with mock.patch.object(comparison, "load_cohort", return_value=fixture()), \
             mock.patch.object(encoders, "load_encoder") as load, \
             contextlib.redirect_stdout(io.StringIO()):
            comparison.run(comparison.parse_args(["--check-data", "--outer-folds", "2", "--inner-folds", "2"]))
        load.assert_not_called()

    def test_repeated_runs_share_parent_and_preserve_previous_results(self):
        cohort = fixture()
        scores = np.where(cohort.y, 1., -1.)
        summary = comparison._summary(cohort.y, scores[None], [comparison.metrics(cohort.y, scores)])
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "foundations_comparison"
            parent.mkdir()
            previous_manifest = parent / "manifest.json"
            previous_manifest.write_text('{"previous_run": true}\n')
            previous_predictions = parent / "tspulse/frozen/repeat_01_fold_01/test_predictions.csv"
            previous_predictions.parent.mkdir(parents=True)
            previous_predictions.write_text("previous results\n")
            args = comparison.parse_args([
                "--models", "tspulse", "--mode", "frozen", "--device", "cpu", "--no-progress",
                "--outer-folds", "2", "--inner-folds", "2", "--output", str(parent),
                "--cache-dir", str(Path(temporary) / "cache"),
            ])
            with mock.patch.object(comparison, "load_cohort", return_value=cohort), \
                 mock.patch.object(comparison, "evaluate_frozen", return_value=summary), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = comparison.run(args)
                first_manifest = (first / "manifest.json").read_bytes()
                second = comparison.run(args)
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, parent.resolve())
            self.assertEqual(second.parent, parent.resolve())
            self.assertEqual((first / "manifest.json").read_bytes(), first_manifest)
            self.assertEqual(previous_manifest.read_text(), '{"previous_run": true}\n')
            self.assertEqual(previous_predictions.read_text(), "previous results\n")
            for output in (first, second):
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["run_id"], output.name)
                self.assertEqual(manifest["output_root"], str(parent.resolve()))
                self.assertTrue((output / "comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
