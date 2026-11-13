"""Verify checkpoint restoration and separation of every validation role."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from sota_pd_et import compare_foundations as comparison
from sota_pd_et.foundations import encoders, training
from sota_pd_et.foundations.decision import select_threshold
from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.foundations.registry import SPECS
from sota_pd_et.foundations.validation import FINETUNE_PROTOCOL, checkpoint_split, make_checkpoint_plan
from sota_pd_et.tests.test_foundation_all_model_decisions import imbalanced_fixture, stable_tiny_loader
from sota_pd_et.tests.test_foundation_protocol import fixture, toy_loader


CHECKPOINT_COVERAGE = {}


class FoundationCheckpointValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_stratified_checkpoint_and_lr_splits_are_disjoint_and_reproducible(self):
        cohort = imbalanced_fixture()
        outer = comparison.make_plan(cohort.y, cohort.ids, 2, 2, 1, 71)
        plan = make_checkpoint_plan(cohort.y, cohort.ids, outer, .2, 2)
        self.assertEqual(plan, make_checkpoint_plan(cohort.y, cohort.ids, outer, .2, 2))
        for split in plan:
            fit, valid, test = map(set, (split["fit_indices"], split["validation_indices"], split["test_indices"]))
            self.assertFalse(fit & valid or fit & test or valid & test)
            self.assertEqual(fit | valid, set(split["pool_indices"]))
            for group in (fit, valid):
                self.assertEqual(set(cohort.y[list(group)]), {0, 1})
            for inner in split["inner_splits"]:
                self.assertLessEqual(set(inner["train_ids"]) | set(inner["validation_ids"]), set(split["fit_ids"]))
                self.assertFalse((set(inner["train_ids"]) | set(inner["validation_ids"])) & set(split["validation_ids"]))

    def test_insufficient_classes_and_invalid_validation_fractions_fail_explicitly(self):
        y, ids = np.array([0, 0, 0, 0, 1, 1]), np.array([str(i) for i in range(6)])
        with self.assertRaisesRegex(ValueError, "Cannot reserve stratified checkpoint validation"):
            checkpoint_split(y, ids, np.arange(6), .3, 2, 1)
        for fraction in (0., 1., -.1, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                checkpoint_split(y, ids, np.arange(6), fraction, 2, 1)

    def _restore_after_metric_decline(self, name, device):
        cohort = imbalanced_fixture()
        states = []
        native_predict = training.predict_scores
        native_select = training.select_threshold

        def observed_predict(model, *args, **kwargs):
            scores = native_predict(model, *args, **kwargs)
            states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
            return scores

        def declining_metric(*args, **kwargs):
            selection = native_select(*args, **kwargs)
            # Control only the selection metric. Actual validation forwards,
            # state snapshots, thresholds, and gradients remain unmodified.
            selection["inner_mean_balanced_accuracy"] = .9 if len(states) == 1 else .5
            return selection

        with mock.patch.object(training, "predict_scores", side_effect=observed_predict), \
             mock.patch.object(training, "select_threshold", side_effect=declining_metric):
            model, result = training.fit_classifier(
                cohort.X[:8], cohort.y[:8], EncoderOptions(name, device=device, channel_batch_size=8),
                training.TrainOptions(max_epochs=3, patience=1, batch_size=2, accumulation_steps=2),
                71, None, validation=(cohort.X[8:16], cohort.y[8:16]), loader=stable_tiny_loader,
            )
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["epochs_trained"], 2)
        self.assertTrue(result["best_checkpoint_restored"])
        self.assertFalse(model.training)
        self.assertTrue(any(not torch.equal(states[0][key], states[1][key]) for key in states[0]))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), states[0][key], rtol=0, atol=0)
        replay = native_predict(model, cohort.X[8:16], 2)
        np.testing.assert_allclose(replay, result["best_validation_scores"], rtol=1e-5, atol=2e-6)
        selection = select_threshold(cohort.y[8:16], replay)
        self.assertAlmostEqual(selection["raw_decision_threshold"],
                               result["best_threshold_selection"]["raw_decision_threshold"], places=5)
        return result

    def test_all_eight_models_restore_best_weights_and_reproduce_their_validation_scores(self):
        for name in SPECS:
            with self.subTest(model=name):
                self._restore_after_metric_decline(name, "cpu")
                CHECKPOINT_COVERAGE[name] = {"restored_best_after_later_updates": True,
                                             "selected_validation_scores_replayed": True}

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_tspulse_mps_checkpoint_restores_through_cpu_mapped_storage(self):
        self._restore_after_metric_decline("tspulse", "mps")
        CHECKPOINT_COVERAGE.setdefault("tspulse", {})["mps_checkpoint_restore"] = True

    def test_discarded_inner_models_do_not_write_unused_checkpoints(self):
        cohort = fixture()
        with mock.patch.object(training.torch, "save") as save:
            model, result = training.fit_classifier(
                cohort.X[:8], cohort.y[:8], EncoderOptions("tspulse"),
                training.TrainOptions(max_epochs=2, batch_size=4), 71, None,
                validation=(cohort.X[8:], cohort.y[8:]), loader=toy_loader, retain_model=False,
            )
        self.assertIsNone(model)
        self.assertFalse(result["best_checkpoint_restored"])
        self.assertTrue(np.isfinite(result["best_validation_scores"]).all())
        save.assert_not_called()

    def test_outer_test_labels_cannot_change_lr_checkpoint_or_its_own_threshold(self):
        cohort = fixture()
        plan = comparison.make_plan(cohort.y, cohort.ids, 2, 2, 1, 71)
        changed = cohort.y.copy()
        changed[plan[0]["test_indices"]] = 1 - changed[plan[0]["test_indices"]]
        alternative = replace(cohort, y=changed)
        outputs, calls = [], []
        native_fit = training.fit_classifier

        def observed_fit(*args, **kwargs):
            self.assertIsNotNone(kwargs.get("validation"))
            self.assertIsNone(kwargs.get("epochs"))
            calls.append(kwargs["retain_model"])
            return native_fit(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            for index, data in enumerate((cohort, alternative)):
                args = comparison.parse_args([
                    "--models", "tspulse", "--mode", "finetune", "--device", "cpu", "--no-progress",
                    "--seed", "71", "--outer-folds", "2", "--inner-folds", "2", "--max-epochs", "2",
                    "--learning-rates", ".0001", "--save-fold-models",
                    "--output", str(Path(temporary) / str(index)), "--cache-dir", str(Path(temporary) / "cache"),
                ])
                with mock.patch.object(comparison, "load_cohort", return_value=data), \
                     mock.patch.object(comparison, "make_plan", return_value=plan), \
                     mock.patch.object(encoders, "load_encoder", side_effect=toy_loader), \
                     mock.patch.object(training, "load_encoder", side_effect=toy_loader), \
                     mock.patch.object(training, "fit_classifier", side_effect=observed_fit), \
                     contextlib.redirect_stdout(io.StringIO()):
                    output = comparison.run(args)
                fold = output / "tspulse/finetune/repeat_01_fold_01"
                meta = json.loads((fold / "model.json").read_text())
                self.assertEqual(meta["training_protocol"], FINETUNE_PROTOCOL)
                self.assertNotIn("refit_epochs", meta)
                outputs.append((json.loads((fold / "selected.json").read_text()),
                                json.loads((fold / "threshold_selection.json").read_text()),
                                torch.load(fold / "model_state.pt", map_location="cpu", weights_only=True)))
        self.assertEqual(calls, [False, False, True] * 4)
        self.assertEqual(outputs[0][:2], outputs[1][:2])
        for key in outputs[0][2]:
            torch.testing.assert_close(outputs[0][2][key], outputs[1][2][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
