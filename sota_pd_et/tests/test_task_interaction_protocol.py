"""Only fabricated participants: nested selection, baseline isolation and output."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from sota_pd_et.data import Cohort
from sota_pd_et import evaluate_task_interaction as evaluation
from sota_pd_et.task_interaction import TaskStandardizer, load_network, predict_logits, select_task_segments, variant_config


class FakeFeatureMap:
    """Test double checking disjoint input people, not an imitation of HYDRA."""
    fitted = []

    def __init__(self, config, seed):
        self.seed = seed

    def fit_transform(self, X):
        self.ids = set(X[:, 0, 0].tolist())
        self.fitted.append(self.ids)
        return X[:, :3].mean(-1).astype(float)

    def transform(self, X):
        if self.ids & set(X[:, 0, 0].tolist()):
            raise AssertionError("Feature map received a training participant in its validation input.")
        return X[:, :3].mean(-1).astype(float)


class TaskProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def cohort(self):
        rng = np.random.default_rng(51)
        y = np.r_[np.zeros(12, dtype=int), np.ones(4, dtype=int)]
        ids = np.array([f"{i:03d}" for i in range(len(y))])
        X = rng.normal(size=(len(y), 132, 32)).astype(np.float32)
        X[:, 0, 0] = np.arange(len(y)) + 100
        rows = [{"participant_id": str(pid), "target_et": int(target),
                 "condition": "Essential Tremor" if target else "Parkinson's"}
                for pid, target in zip(ids, y, strict=True)]
        return Cohort(X, y, ids, rows, {"dataset": "SYNTHETIC TEST ONLY", "condition_counts": {"PD": 12, "ET": 4}})

    def args(self, root, models=None, extra=()):
        return evaluation.parse_args([
            "--data-root", str(root / "unused-data"), "--output", str(root / "results"),
            "--models", *(models or list(evaluation.ALL_MODELS)),
            "--outer-folds", "2", "--inner-folds", "2", "--epochs", "1", "2",
            "--weight-decays", "0.01", "--alphas", "0.1", "1", "--thresholds", "-0.1", "0", "0.1",
            "--hidden-dim", "8", "--heads", "2", "--batch-size", "4", "--threads", "1",
            "--device", "cpu", "--bootstrap-samples", "30", *extra,
        ])

    def test_neural_inner_scalers_see_only_their_training_participants(self):
        cohort = self.cohort()
        X = select_task_segments(cohort.X)
        inner = evaluation.make_inner_splits(cohort.y, 2, 13)
        fitted = []
        original_fit = TaskStandardizer.fit

        def fit(scaler, values):
            fitted.append(set(values[:, 0, 0, 0].tolist()))
            return original_fit(scaler, values)

        def fake_fit(training, y, config, **kwargs):
            # Training-only centering; no cohort/global scaler is reused.
            np.testing.assert_allclose(training.mean(axis=(0, 1, 3)), 0, atol=2e-7)
            scores = {e: np.zeros(len(kwargs["validation"])) for e in kwargs["checkpoint_epochs"]}
            return object(), scores, []

        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary), ["full"])
            config = variant_config("full", samples_per_segment=32, hidden_dim=8, heads=2)
            with mock.patch.object(TaskStandardizer, "fit", fit), \
                    mock.patch.object(evaluation, "fit_trajectory", side_effect=fake_fit):
                selected, _, _ = evaluation.tune_neural(X, cohort.y, config, inner, args, 17)
        self.assertEqual(fitted, [set(X[a, 0, 0, 0].tolist()) for a, _ in inner])
        self.assertEqual(selected["epochs"], 1)
        self.assertEqual(selected["threshold"], 0)

    def test_threshold_affects_classification_but_ranking_uses_raw_scores(self):
        y = np.array([0, 0, 1, 1])
        raw = np.array([-0.9, 0.8, 0.5, 0.6])
        threshold = np.array([-2, -2, 2, 2])
        result = evaluation.scored_metrics(y, raw, threshold)
        self.assertEqual(result["roc_auc"], roc_auc_score(y, raw))
        self.assertEqual(result["balanced_accuracy"], 0)

    def test_paired_bootstrap_keeps_all_repeats_of_one_person_together(self):
        y = np.tile([0, 1], 8)
        a = y[None].copy()
        b = a.copy()
        b[0, :5] = 1 - b[0, :5]
        first = evaluation.paired_bootstrap(y, a, b, seed=7, samples=100)
        second = evaluation.paired_bootstrap(y, np.repeat(a, 3, 0), np.repeat(b, 3, 0), seed=7, samples=100)
        self.assertEqual(first, second)
        self.assertEqual(first["independent_participants"], 16)

    def test_selection_ties_prefer_regularization_fewer_epochs_and_zero_threshold(self):
        rows = [{"inner_mean_balanced_accuracy": 0.7, "weight_decay": d, "epochs": e, "threshold": t}
                for d in [0.001, 0.01] for e in [1, 2] for t in [-0.1, 0, 0.1]]
        selected = evaluation.choose_search(rows, "neural")
        self.assertEqual((selected["weight_decay"], selected["epochs"], selected["threshold"]), (0.01, 1, 0))

    def test_complete_synthetic_suite_writes_only_held_out_predictions_and_final_model(self):
        cohort = self.cohort()
        Z = np.random.default_rng(61).normal(size=(16, 8)).astype(np.float32)
        FakeFeatureMap.fitted = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root, extra=["--fit-final"])
            with mock.patch.object(evaluation, "load_cohort", return_value=cohort), \
                    mock.patch.object(evaluation, "extract_features", return_value=(Z, {"synthetic": True})) as extract, \
                    mock.patch.object(evaluation, "FeatureMap", FakeFeatureMap), contextlib.redirect_stdout(io.StringIO()):
                output = evaluation.run(args)
            extract.assert_called_once()
            self.assertEqual(tuple(extract.call_args.args[0].shape), (16, 60, 32))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_models"], list(evaluation.ALL_MODELS))
            plan = json.loads((output / "splits.json").read_text())
            import csv
            for name in evaluation.ALL_MODELS:
                with (output / name / "oof_predictions.csv").open() as stream:
                    predictions = list(csv.DictReader(stream))
                self.assertCountEqual([r["participant_id"] for r in predictions], cohort.ids.tolist())
                for split in plan:
                    rows = [r for r in predictions if int(r["fold"]) == split["fold"]]
                    self.assertCountEqual([r["participant_id"] for r in rows], split["test_ids"])
                    self.assertFalse(set(split["train_ids"]) & {r["participant_id"] for r in rows})
                    for row in rows:
                        self.assertEqual(int(row["predicted_et"]), int(float(row["raw_score_et"]) > float(row["decision_threshold"])))
            # Two outer folds: two inner maps plus one outer map in each.
            self.assertEqual(len(FakeFeatureMap.fitted), 6)
            network, scaler, payload = load_network(output / "final_model.pt")
            score = predict_logits(network, scaler.transform(cohort.X), batch_size=4)
            self.assertTrue(np.isfinite(score).all())
            self.assertEqual(set(payload["metadata"]["training_ids"]), set(cohort.ids))
            self.assertTrue((output / "REPORT.md").is_file())
            self.assertTrue((output / "paired_comparisons.json").is_file())
            with mock.patch.object(evaluation, "load_cohort", return_value=cohort), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                evaluation.run(args)

    def test_failure_is_recorded_without_claiming_completed_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root, ["full"])
            with mock.patch.object(evaluation, "load_cohort", return_value=self.cohort()), \
                    mock.patch.object(evaluation, "tune_neural", side_effect=RuntimeError("synthetic failure")), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                evaluation.run(args)
            manifest = json.loads((root / "results" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_models"], [])
            self.assertFalse((root / "results" / "summary.json").exists())

    def test_explicit_mismatched_reference_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "wrong.json").write_text("[]")
            args = self.args(root, ["full"], ["--reference-splits", str(root / "wrong.json")])
            with mock.patch.object(evaluation, "load_cohort", return_value=self.cohort()), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "splits differ"):
                evaluation.run(args)
            self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
