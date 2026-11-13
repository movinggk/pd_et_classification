"""Tests use small synthetic fixtures, never train on the real PADS cohort."""

from __future__ import annotations

import csv
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from sota_pd_et.data import CHANNEL_NAMES, MOVEMENT_SHAPE, load_cohort, sha256_file


HAS_SKLEARN = all(find_spec(p) is not None for p in ("sklearn", "joblib", "threadpoolctl"))
HAS_MODEL_DEPS = HAS_SKLEARN and all(find_spec(p) is not None for p in ("aeon", "torch"))


def make_fixture(root):
    (root / "preprocessed/movement").mkdir(parents=True)
    (root / "patients").mkdir()
    (root / "scripts/utils").mkdir(parents=True)
    (root / "scripts/run_preprocessing.py").write_text("# synthetic fixture\n")
    (root / "scripts/utils/data_handling.py").write_text("# synthetic fixture\n")
    rows = [
        {"id": "004", "condition": "Parkinson's", "label": "1", "disease_comment": "essential tremor history"},
        {"id": "066", "condition": "Essential Tremor", "label": "2", "disease_comment": "-"},
        {"id": "002", "condition": "Other Movement Disorders", "label": "2", "disease_comment": "-"},
    ]
    with (root / "preprocessed/file_list.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows[:2]:
        (root / f"patients/patient_{row['id']}.json").write_text(json.dumps(row))
        np.full(MOVEMENT_SHAPE, 0.25, dtype="<f4").tofile(
            root / f"preprocessed/movement/{row['id']}_ml.bin")
    update_checksums(root)


def update_checksums(root):
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (root / "SHA256SUMS.txt").write_text("".join(
        f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files
    ))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        make_fixture(self.root)

    def test_condition_strings_define_binary_target(self):
        cohort = load_cohort(self.root)
        self.assertEqual(cohort.ids.tolist(), ["004", "066"])
        self.assertEqual(cohort.y.tolist(), [0, 1])
        self.assertEqual(cohort.X.shape, (2, 132, 976))
        self.assertEqual(cohort.provenance["pd_with_et_comment_ids"], ["004"])
        self.assertEqual(len(CHANNEL_NAMES), 132)
        self.assertEqual(CHANNEL_NAMES[0], "Relaxed1_LeftWrist_Accelerometer_X")
        self.assertEqual(CHANNEL_NAMES[-1], "Entrainment2_RightWrist_Gyroscope_Z")

    def test_checksum_tampering_is_rejected(self):
        path = self.root / "preprocessed/movement/066_ml.bin"
        with path.open("r+b") as stream:
            stream.write(b"bad!")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_cohort(self.root)

    def test_nonfinite_samples_are_rejected_after_valid_checksum(self):
        path = self.root / "preprocessed/movement/066_ml.bin"
        values = np.fromfile(path, dtype="<f4")
        values[0] = np.nan
        values.tofile(path)
        update_checksums(self.root)
        with self.assertRaisesRegex(ValueError, "Nonfinite movement samples"):
            load_cohort(self.root)

    def test_patient_table_disagreement_is_rejected(self):
        path = self.root / "patients/patient_066.json"
        row = json.loads(path.read_text())
        row["condition"] = "Parkinson's"
        path.write_text(json.dumps(row))
        update_checksums(self.root)
        with self.assertRaisesRegex(ValueError, "table/JSON disagreement"):
            load_cohort(self.root)


@unittest.skipUnless(HAS_SKLEARN, "Install sota_pd_et/requirements.txt for protocol tests.")
class ProtocolTests(unittest.TestCase):
    def test_every_person_has_one_test_prediction_per_repeat_and_nested_splits(self):
        from sota_pd_et.evaluate import make_plan

        y = np.r_[np.zeros(80, dtype=int), np.ones(8, dtype=int)]
        ids = np.array([f"{i:03d}" for i in range(len(y))])
        plan = make_plan(y, ids, outer_folds=4, inner_folds=3, repeats=2, seed=17)
        self.assertEqual(plan, make_plan(y, ids, 4, 3, 2, 17))
        for repeat in (1, 2):
            test_ids = [p for s in plan if s["repeat"] == repeat for p in s["test_ids"]]
            self.assertCountEqual(test_ids, ids.tolist())
        for split in plan:
            train, test = set(split["train_ids"]), set(split["test_ids"])
            self.assertFalse(train & test)
            inner_valid = []
            for inner in split["inner_splits"]:
                a, b = set(inner["train_ids"]), set(inner["validation_ids"])
                self.assertFalse(a & b)
                self.assertEqual(a | b, train)
                self.assertFalse((a | b) & test)
                inner_valid.extend(inner["validation_ids"])
            self.assertCountEqual(inner_valid, split["train_ids"])

    def test_impossible_inner_stratification_is_rejected(self):
        from sota_pd_et.evaluate import make_plan

        y = np.r_[np.zeros(20, dtype=int), np.ones(4, dtype=int)]
        with self.assertRaisesRegex(ValueError, "Inner fold count"):
            make_plan(y, np.arange(len(y)), 4, 4, 1, 17)

    def test_metric_orientation_and_fixed_zero_threshold(self):
        from sota_pd_et.evaluate import metrics

        result = metrics(np.array([0, 0, 1, 1]), np.array([-2.0, 0.0, 0.0, 2.0]))
        self.assertEqual(result["confusion_matrix_true_rows_predicted_columns_pd_et"], [[2, 0], [1, 1]])
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["et_recall"], 0.5)
        self.assertAlmostEqual(result["pd_recall"], 1.0)
        self.assertAlmostEqual(result["roc_auc"], 0.875)

    def test_ties_choose_stronger_regularization(self):
        from sota_pd_et.evaluate import choose_alpha

        rows = [{"alpha": a, "inner_mean_balanced_accuracy": score}
                for a, score in [(0.1, 0.8), (1.0, 0.7), (10.0, 0.8)]]
        self.assertEqual(choose_alpha(rows), 10.0)

    def test_feature_maps_are_refitted_without_inner_validation_people(self):
        import sota_pd_et.evaluate as evaluation
        from sota_pd_et.model import ModelConfig

        X = np.zeros((16, 1, 16), dtype=np.float32)
        X[:, 0, 0] = np.arange(16)
        y = np.tile([0, 1], 8)
        fitted_ids = []
        test_case = self

        class TrackingMap:
            def __init__(self, config, seed):
                pass

            def fit_transform(self, values):
                self.people = set(values[:, 0, 0].tolist())
                fitted_ids.append(self.people)
                return values[:, :, 0]

            def transform(self, values):
                test_case.assertFalse(self.people & set(values[:, 0, 0].tolist()))
                return values[:, :, 0]

        class FixedMargin:
            def decision_function(self, features):
                return 2 * (features[:, 0] % 2) - 1

        inner = evaluation.make_inner_splits(y, 4, 17)
        with mock.patch.object(evaluation, "FeatureMap", TrackingMap), \
                mock.patch.object(evaluation, "fit_ridge", return_value=FixedMargin()):
            alpha, _, scores = evaluation.tune_alpha(X, y, ModelConfig(), [0.1, 10.0], inner, 17)
        self.assertEqual(len(fitted_ids), 4)
        self.assertTrue(all(len(people) == 12 for people in fitted_ids))
        self.assertEqual(alpha, 10.0)
        np.testing.assert_array_equal(scores > 0, y)

    def test_ridge_score_orientation(self):
        from sota_pd_et.model import fit_ridge

        Z = np.array([[-3.0, 0.0], [-2.0, 1.0], [2.0, -1.0], [3.0, 0.0]])
        y = np.array([0, 0, 1, 1])
        classifier = fit_ridge(Z, y, 1.0)
        np.testing.assert_array_equal(classifier.decision_function(Z) > 0, y)


@unittest.skipUnless(HAS_MODEL_DEPS, "Install full dependencies for the synthetic model/serialization test.")
class ModelSmokeTests(unittest.TestCase):
    def test_batched_real_features_and_serialization(self):
        import joblib
        from sota_pd_et.evaluate import fit_full_model
        from sota_pd_et.model import ModelConfig

        rng = np.random.default_rng(7)
        X = rng.normal(size=(12, 3, 64)).astype(np.float32)
        y = np.tile([0, 1], 6)
        config = ModelConfig(mr_kernels=84, hydra_groups=4, hydra_kernels=2,
                             batch_size=3, n_jobs=1)
        fitted = fit_full_model(X[:8], y[:8], config, alpha=1.0, seed=17)
        before = fitted.decision_function(X[8:])
        self.assertEqual(before.shape, (4,))
        self.assertTrue(np.isfinite(before).all())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.joblib"
            joblib.dump(fitted, path)
            after = joblib.load(path).decision_function(X[8:])
        np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-12)

    def test_complete_synthetic_evaluation_and_prediction_entry_point(self):
        import contextlib
        import io
        import joblib
        import sota_pd_et.evaluate as evaluation
        from sota_pd_et.data import Cohort
        from sota_pd_et.predict import main as predict

        rng = np.random.default_rng(19)
        X = rng.normal(size=(16, 3, 64)).astype(np.float32)
        y = np.r_[np.zeros(12, dtype=int), np.ones(4, dtype=int)]
        ids = np.array([f"{i:03d}" for i in range(16)])
        cohort = Cohort(X, y, ids, [
            {"participant_id": pid, "condition": "Essential Tremor" if target else "Parkinson's",
             "target_et": int(target), "pd_with_et_comment": False}
            for pid, target in zip(ids, y, strict=True)
        ], {"condition_counts": {"Parkinson's": 12, "Essential Tremor": 4},
            "channel_names": ["synthetic_x", "synthetic_y", "synthetic_z"]})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results"
            args = evaluation.parse_args([
                "--data-root", str(root / "unused_source"), "--output", str(output),
                "--outer-folds", "2", "--inner-folds", "2", "--repeats", "1",
                "--n-jobs", "1", "--batch-size", "3", "--mr-kernels", "84",
                "--hydra-groups", "4", "--hydra-kernels", "2", "--alphas", "0.1", "1",
                "--fit-final",
            ])
            with mock.patch.object(evaluation, "load_cohort", return_value=cohort), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.run(args)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "complete")
            with (output / "oof_predictions.csv").open(newline="") as stream:
                predictions = list(csv.DictReader(stream))
            self.assertCountEqual([row["participant_id"] for row in predictions], ids.tolist())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["repeat_metrics"][0]["n_pd"], 12)
            self.assertEqual(summary["repeat_metrics"][0]["n_et"], 4)
            self.assertAlmostEqual(summary["always_predict_pd_baseline"]["balanced_accuracy"], 0.5)
            np.save(root / "new.npy", X[:2])
            with contextlib.redirect_stdout(io.StringIO()):
                predict(["--model", str(output / "final_model.joblib"),
                         "--inputs", str(root / "new.npy"), "--output", str(root / "predictions.csv")])
            with (root / "predictions.csv").open(newline="") as stream:
                actual = [float(row["decision_score_et"]) for row in csv.DictReader(stream)]
            expected = joblib.load(output / "final_model.joblib")["model"].decision_function(X[:2])
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
