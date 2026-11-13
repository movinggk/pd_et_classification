"""Focused leakage, baseline-matching and synthetic evaluator checks."""

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
from sklearn.preprocessing import StandardScaler as RealStandardScaler

from sota_pd_et.data import Cohort
from sota_pd_et import evaluate_transformer as evaluation


class TransformerProtocolTests(unittest.TestCase):
    def fixture(self, root):
        y = np.r_[np.zeros(12, dtype=int), np.ones(4, dtype=int)]
        ids = np.array([f"{i:03d}" for i in range(len(y))])
        X = np.random.default_rng(17).normal(size=(len(y), 2, 32)).astype(np.float32)
        provenance = {
            "dataset": "synthetic", "checksum_list_sha256": "synthetic-checksum",
            "verified_source_sha256": {"synthetic.bin": "synthetic-source"},
            "label_mapping": {"Parkinson's": 0, "Essential Tremor": 1},
            "participant_count": len(y), "condition_counts": {"Parkinson's": 12, "Essential Tremor": 4},
            "movement_shape": list(X.shape), "channel_names": ["synthetic_a", "synthetic_b"],
        }
        cohort = Cohort(X, y, ids, [
            {"participant_id": pid, "condition": "Essential Tremor" if target else "Parkinson's",
             "target_et": int(target), "pd_with_et_comment": False}
            for pid, target in zip(ids, y, strict=True)
        ], provenance)
        plan = evaluation.make_plan(y, ids, 2, 2, 1, 20261017)
        baseline = root / "baseline"
        baseline.mkdir()
        evaluation.write_json(baseline / "manifest.json", {
            "status": "complete", "dataset": provenance, "config": {"alphas": [0.1, 1.0]},
        })
        evaluation.write_json(baseline / "splits.json", plan)
        evaluation.write_json(baseline / "summary.json", {})
        evaluation.write_csv(baseline / "cohort.csv", cohort.rows)
        predictions = []
        for split in plan:
            test = np.array(split["test_indices"])
            predictions.extend(evaluation.prediction_rows(
                ids[test], y[test], np.where(y[test], 1.0, -1.0),
                repeat=split["repeat"], fold=split["fold"],
            ))
        evaluation.write_csv(baseline / "oof_predictions.csv", predictions)
        return cohort, plan, baseline

    def test_inner_scalers_only_see_training_participants(self):
        Z = np.column_stack((np.arange(16), np.arange(16) ** 2)).astype(np.float32)
        y = np.tile([0, 1], 8)
        inner = evaluation.make_inner_splits(y, 4, 17)
        fitted = []
        test = self

        class TrackingScaler:
            def fit_transform(self, rows):
                self.seen = set(rows[:, 0].tolist())
                fitted.append(self.seen)
                self.scaler = RealStandardScaler()
                return self.scaler.fit_transform(rows)

            def transform(self, rows):
                test.assertFalse(self.seen & set(rows[:, 0].tolist()))
                return self.scaler.transform(rows)

        with mock.patch.object(evaluation, "StandardScaler", TrackingScaler):
            alpha, search, scores = evaluation.tune_alpha(Z, y, [0.1, 1.0], inner)
        self.assertEqual(len(fitted), len(inner))
        self.assertTrue(all(len(people) == 12 for people in fitted))
        self.assertIn(alpha, [0.1, 1.0])
        self.assertEqual(len(search), 2)
        self.assertTrue(np.isfinite(scores).all())

    def test_baseline_rejects_label_hash_and_split_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort, plan, baseline = self.fixture(Path(temporary))
            predictions, _ = evaluation.validate_baseline(baseline, cohort, plan)
            self.assertEqual(len(predictions), len(cohort.y))
            changed = json.loads(json.dumps(plan))
            changed[0]["model_seed"] += 1
            with self.assertRaisesRegex(ValueError, "assignments differ"):
                evaluation.validate_baseline(baseline, cohort, changed)
            manifest_path = baseline / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["dataset"]["verified_source_sha256"]["synthetic.bin"] = "changed"
            evaluation.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                evaluation.validate_baseline(baseline, cohort, plan)
            manifest["dataset"] = cohort.provenance
            evaluation.write_json(manifest_path, manifest)
            rows = evaluation.read_csv(baseline / "oof_predictions.csv")
            rows[0]["target_et"] = 1 - int(rows[0]["target_et"])
            evaluation.write_csv(baseline / "oof_predictions.csv", rows)
            with self.assertRaisesRegex(ValueError, "target differs"):
                evaluation.validate_baseline(baseline, cohort, plan)

    def test_baseline_requires_the_same_alpha_candidates_when_supplied(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort, plan, baseline = self.fixture(Path(temporary))
            _, provenance = evaluation.validate_baseline(baseline, cohort, plan, [0.1, 1.0])
            self.assertEqual(provenance["verified_alpha_candidates"], [0.1, 1.0])
            with self.assertRaisesRegex(ValueError, "alpha candidates differ"):
                evaluation.validate_baseline(baseline, cohort, plan, [0.1, 1.0, 10.0])

    def test_complete_synthetic_run_extracts_once_and_serializes_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort, _, baseline = self.fixture(root)
            output = root / "new-results"
            args = evaluation.parse_args([
                "--data-root", str(root / "unused-data"), "--output", str(output),
                "--baseline", str(baseline), "--outer-folds", "2", "--inner-folds", "2",
                "--alphas", "0.1", "1", "--fit-final",
            ])
            Z = np.random.default_rng(9).normal(size=(len(cohort.y), 12)).astype(np.float32)
            with mock.patch.object(evaluation, "load_cohort", return_value=cohort), \
                    mock.patch.object(evaluation, "extract_features", return_value=(Z, {"synthetic": True})) as extract, \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.run(args)
            extract.assert_called_once()
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "complete")
            rows = evaluation.read_csv(output / "oof_predictions.csv")
            self.assertCountEqual([r["participant_id"] for r in rows], cohort.ids.tolist())
            comparison = json.loads((output / "baseline_comparison.json").read_text())
            self.assertEqual(comparison["repeat_comparisons"][0]["baseline"]["balanced_accuracy"], 1.0)
            bundle = joblib.load(output / "final_model.joblib")
            self.assertEqual(bundle["model_type"], "frozen_transformer_ridge")
            scores = bundle["classifier"].decision_function(bundle["scaler"].transform(Z))
            self.assertEqual(scores.shape, cohort.y.shape)
            self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
