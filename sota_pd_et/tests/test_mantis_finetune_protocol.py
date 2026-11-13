"""Fabricated participants only: nested fitting, matching and output checks."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from tqdm.auto import tqdm as real_tqdm

from sota_pd_et.data import Cohort
from sota_pd_et import evaluate_mantis_finetune as evaluation
from sota_pd_et import mantis_finetune as fine
from sota_pd_et.tests.test_mantis_finetune import TinyEncoder, tiny_model


class FinetuneProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        y = np.r_[np.zeros(12, dtype=int), np.ones(4, dtype=int)]
        ids = np.array([f"{i:03d}" for i in range(len(y))])
        X = np.random.default_rng(31).normal(size=(16, 132, 32)).astype(np.float32)
        X[:, 0, 0] = np.arange(16) + 100
        rows = [{"participant_id": str(pid), "target_et": int(target),
                 "condition": "Essential Tremor" if target else "Parkinson's"}
                for pid, target in zip(ids, y, strict=True)]
        provenance = {"dataset": "SYNTHETIC ONLY", "checksum_list_sha256": "synthetic-list",
                      "verified_source_sha256": {"synthetic.bin": "synthetic-hash"},
                      "label_mapping": {"Parkinson's": 0, "Essential Tremor": 1},
                      "participant_count": 16, "condition_counts": {"Parkinson's": 12, "Essential Tremor": 4},
                      "movement_shape": list(X.shape), "channel_names": [str(i) for i in range(132)]}
        cohort = Cohort(X, y, ids, rows, provenance)
        Z = np.random.default_rng(32).normal(size=(16, 132 * 512)).astype(np.float32)
        Z[:, 0] = np.arange(16) + 200
        plan = evaluation.make_plan(y, ids, 2, 2, 1, 20261017)
        return cohort, Z, plan

    def args(self, root, extra=()):
        return evaluation.parse_args([
            "--data-root", str(root / "unused-data"), "--output", str(root / "results"),
            "--device", "cpu", "--threads", "1", "--outer-folds", "2", "--inner-folds", "2",
            "--epochs", "1", "--encoder-lrs", "0.00001", "--weight-decays", "0.01",
            "--head-warmup-epochs", "0", "--batch-size", "4", "--accumulation-steps", "1",
            "--channel-batch-size", "256", "--no-progress", *extra,
        ])

    def baseline(self, root, cohort, plan, three_tasks=False):
        directory = root / "baseline"
        directory.mkdir()
        manifest = {"status": "complete", "dataset": cohort.provenance}
        if three_tasks:
            manifest.update(input_protocol={"selected_channel_indices": list(evaluation.CHANNEL_INDICES)}, completed_models=["mantis"])
            (directory / "mantis").mkdir()
        evaluation.write_json(directory / "manifest.json", manifest)
        evaluation.write_json(directory / "splits.json", plan)
        evaluation.write_csv(directory / "cohort.csv", cohort.rows)
        provenance = {"encoder": {"model_id": evaluation.MODEL_ID, "revision": evaluation.REVISION,
                                   "sequence_length": 512, "return_transf_layer": 2, "output_token": "combined"},
                      "parameters_trainable": 0, "checkpoint": {"sha256": dict(evaluation.CHECKPOINT_HASHES)},
                      "input_shape": [16, 60 if three_tasks else 132, 32]}
        evaluation.write_json(directory / ("mantis_encoder_provenance.json" if three_tasks else "encoder_provenance.json"), provenance)
        rows = []
        for split in plan:
            index = np.array(split["test_indices"])
            rows += evaluation.prediction_rows(cohort.ids[index], cohort.y[index], np.where(cohort.y[index], 1.0, -1.0),
                                               repeat=1, fold=split["fold"])
        evaluation.write_csv(directory / ("mantis/oof_predictions.csv" if three_tasks else "oof_predictions.csv"), rows)
        return directory

    def test_inner_fits_use_only_training_signals_labels_and_feature_statistics(self):
        cohort, Z, _ = self.fixture()
        inner = evaluation.make_inner_splits(cohort.y, 2, 17)
        calls = []

        def fake_fit(X, y, Z0, *args, **kwargs):
            train, valid = inner[len(calls)]
            np.testing.assert_array_equal(X, cohort.X[train])
            np.testing.assert_array_equal(y, cohort.y[train])
            np.testing.assert_array_equal(Z0, Z[train])
            np.testing.assert_array_equal(kwargs["validation"], cohort.X[valid])
            calls.append(True)
            return object(), {1: np.zeros(len(valid))}, []

        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            with mock.patch.object(evaluation, "fit_finetune", side_effect=fake_fit), contextlib.redirect_stdout(io.StringIO()):
                _, _, scores = evaluation.tune(cohort.X, cohort.y, Z, fine.FinetuneConfig(samples=32),
                    fine.TrainingConfig(warmup_epochs=0), None, inner, args, 5)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(scores), len(cohort.y))

    def test_candidate_selection_is_inner_mean_then_decay_epoch_and_lr(self):
        rows = [{"inner_mean_balanced_accuracy": 0.6, "weight_decay": w, "epochs": e, "encoder_lr": lr}
                for w in [0.001, 0.01] for e in [3, 5] for lr in [1e-5, 3e-5]]
        chosen = evaluation.choose_candidate(rows)
        self.assertEqual((chosen["weight_decay"], chosen["epochs"], chosen["encoder_lr"]), (0.01, 3, 1e-5))

    def test_both_baseline_formats_match_and_wrong_inputs_scores_or_splits_fail(self):
        cohort, _, plan = self.fixture()
        for three in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = self.baseline(root, cohort, plan, three)
                input_set = "three_tasks" if three else "all"
                predictions, provenance = evaluation.validate_comparison(path, cohort, plan, input_set)
                self.assertEqual(len(predictions), 16)
                self.assertEqual(provenance["input_set"], input_set)
                if three:
                    again, _ = evaluation.validate_comparison(path / "mantis", cohort, plan, input_set)
                    self.assertEqual(again, predictions)
                with self.assertRaisesRegex(ValueError, "channels|channel|132"):
                    evaluation.validate_comparison(path, cohort, plan, "all" if three else "three_tasks")
                changed = json.loads(json.dumps(plan)); changed[0]["model_seed"] += 1
                with self.assertRaisesRegex(ValueError, "splits"):
                    evaluation.validate_comparison(path, cohort, changed, input_set)
                rows_path = path / ("mantis/oof_predictions.csv" if three else "oof_predictions.csv")
                rows = evaluation.read_csv(rows_path); rows.append(rows[0])
                evaluation.write_csv(rows_path, rows)
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    evaluation.validate_comparison(path, cohort, plan, input_set)

    def test_full_synthetic_nested_run_uses_fresh_models_and_roundtrips_final_bundle(self):
        cohort, Z, plan = self.fixture()
        constructions = []

        def factory(config, checkpoint_dir, training_features, device="cpu"):
            constructions.append(set(training_features[:, 0].tolist()))
            return tiny_model(config, checkpoint_dir, training_features, device)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.baseline(root, cohort, plan)
            args = self.args(root, ["--fit-final", "--save-outer-models", "--compare-to", str(baseline)])
            args.progress = True
            progress_stream, progress_bars = io.StringIO(), []

            def record_progress(*values, **kwargs):
                bar = real_tqdm(*values, **kwargs, file=progress_stream, mininterval=0)
                progress_bars.append(bar)
                return bar

            record_progress.write = lambda message: real_tqdm.write(message, file=progress_stream)
            with mock.patch.object(evaluation, "load_cohort", return_value=cohort), \
                    mock.patch.object(evaluation, "extract_features", return_value=(Z, {"synthetic": True})) as extract, \
                    mock.patch.object(fine, "create_model", side_effect=factory), \
                    mock.patch.object(evaluation, "tqdm", side_effect=record_progress) as overall_progress, \
                    contextlib.redirect_stderr(progress_stream), contextlib.redirect_stdout(io.StringIO()):
                overall_progress.write.side_effect = record_progress.write
                output = evaluation.run(args)
            extract.assert_called_once()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(constructions), 9)  # 2*(2 inner + 1 outer) + (2 final inner + 1 full fit).
            self.assertEqual(len(progress_bars), 1)
            self.assertEqual(progress_bars[0].total, len(constructions))
            self.assertEqual(progress_bars[0].n, len(constructions))
            self.assertTrue(progress_bars[0].disable)  # close() completed on exit.
            self.assertIn("Model fits", progress_stream.getvalue())
            expected = []
            for split in plan:
                outer_train = np.array(split["train_indices"])
                for inner in split["inner_splits"]:
                    expected.append(set(Z[outer_train[np.array(inner["train_positions"])] ,0].tolist()))
                expected.append(set(Z[outer_train, 0].tolist()))
            self.assertEqual(constructions[:6], expected)
            rows = evaluation.read_csv(output / "oof_predictions.csv")
            self.assertCountEqual([r["participant_id"] for r in rows], cohort.ids.tolist())
            for split in plan:
                group = [r for r in rows if int(r["fold"]) == split["fold"]]
                self.assertCountEqual([r["participant_id"] for r in group], split["test_ids"])
                self.assertFalse({r["participant_id"] for r in group} & set(split["train_ids"]))
            compare = json.loads((output / "frozen_comparison.json").read_text())
            self.assertEqual(compare["repeat_comparisons"][0]["frozen"]["balanced_accuracy"], 1.0)
            with mock.patch.object(fine, "build_uninitialized_encoder", side_effect=lambda _: TinyEncoder()):
                model, payload = fine.load_finetuned(output / "final_model.pt")
            self.assertEqual(set(payload["metadata"]["training_ids"]), set(cohort.ids))
            self.assertTrue(np.isfinite(fine.predict_scores(model, cohort.X)).all())
            self.assertTrue((output / "REPORT.md").is_file())
            with mock.patch.object(evaluation, "load_cohort", return_value=cohort), self.assertRaises(FileExistsError):
                evaluation.run(args)

    def test_three_task_selection_is_applied_before_frozen_extraction(self):
        cohort, _, _ = self.fixture()
        Z = np.random.default_rng(36).normal(size=(16, 60 * 512)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary), ["--input-set", "three_tasks", "--mode", "head-only"])
            with mock.patch.object(evaluation, "load_cohort", return_value=cohort), \
                    mock.patch.object(evaluation, "extract_features", return_value=(Z, {"synthetic": True})) as extract, \
                    mock.patch.object(fine, "create_model", side_effect=tiny_model), contextlib.redirect_stdout(io.StringIO()):
                output = evaluation.run(args)
            np.testing.assert_array_equal(extract.call_args.args[0], cohort.X[:, :60])
            self.assertEqual(json.loads((output / "manifest.json").read_text())["input_shape"], [16, 60, 32])

    def test_training_failure_is_recorded_and_never_marked_complete(self):
        cohort, Z, _ = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = self.args(root)
            with mock.patch.object(evaluation, "load_cohort", return_value=cohort), \
                    mock.patch.object(evaluation, "extract_features", return_value=(Z, {"synthetic": True})), \
                    mock.patch.object(evaluation, "fit_finetune", side_effect=RuntimeError("synthetic failure")), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                evaluation.run(args)
            manifest = json.loads((root / "results/manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse((root / "results/summary.json").exists())

    def test_progress_cli_is_enabled_by_default_and_can_be_disabled(self):
        self.assertTrue(evaluation.parse_args([]).progress)
        self.assertFalse(evaluation.parse_args(["--no-progress"]).progress)
        self.assertTrue(evaluation.parse_args(["--no-progress", "--progress"]).progress)

    def test_invalid_cli_cannot_start_training(self):
        for arguments in (["--epochs", "1"], ["--encoder-lrs", "0"], ["--head-lr", "nan"],
                          ["--accumulation-steps", "0"], ["--inner-folds", "1"], ["--weight-decays", "-1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.parse_args(arguments)


if __name__ == "__main__":
    unittest.main()
