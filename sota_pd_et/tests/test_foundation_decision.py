"""Inner-only threshold selection, all-negative scores, and inference contracts."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits

from sota_pd_et import compare_foundations as comparison
from sota_pd_et.foundations import encoders, ridge, training
from sota_pd_et.foundations.decision import apply_threshold, select_threshold, validation_fold_ids
from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.tests.test_foundation_protocol import fixture, toy_loader


class FoundationDecisionTests(unittest.TestCase):
    def test_negative_scores_can_rank_et_without_crossing_zero(self):
        y = np.r_[np.zeros(18, int), np.ones(2, int)]
        raw = np.r_[np.linspace(-1.2, -.7, 18), [-.4, -.2]]
        selection = select_threshold(y, raw)
        self.assertEqual(balanced_accuracy_score(y, raw > 0), .5)
        self.assertLess(selection["raw_decision_threshold"], 0)
        margins = apply_threshold(raw, selection["raw_decision_threshold"])
        np.testing.assert_array_equal(margins > 0, y)
        self.assertEqual(roc_auc_score(y, margins), roc_auc_score(y, raw))
        self.assertEqual(selection["inner_mean_balanced_accuracy"], 1.)
        self.assertEqual(selection["inner_predicted_et_zero"], 0)

    def test_constant_scores_keep_zero_and_do_not_force_et(self):
        y = np.r_[np.zeros(9, int), 1]
        for score in (-3., 0., 2.):
            with self.subTest(score=score):
                selection = select_threshold(y, np.full(len(y), score))
                self.assertEqual(selection["raw_decision_threshold"], 0.)
                self.assertEqual(selection["inner_mean_balanced_accuracy"], .5)
                self.assertEqual(selection["inner_predicted_et"], len(y) if score > 0 else 0)

    def test_one_common_threshold_maximizes_mean_fold_ba_including_ties(self):
        rng = np.random.default_rng(917)
        y = np.r_[np.zeros(15, int), np.ones(3, int), np.zeros(4, int), np.ones(2, int)]
        folds = np.r_[np.zeros(18, int), np.ones(6, int)]
        for _ in range(6):
            scores = rng.integers(-8, 4, len(y)) / 4
            selection = select_threshold(y, scores, folds)
            candidates = np.r_[np.nextafter(scores.min(), -np.inf), np.unique(scores), 0.]
            expected = max(np.mean([
                balanced_accuracy_score(y[folds == f], scores[folds == f] > t)
                for f in (0, 1)
            ]) for t in candidates)
            self.assertAlmostEqual(selection["inner_mean_balanced_accuracy"], expected, places=14)
            self.assertAlmostEqual(selection["inner_mean_balanced_accuracy"], np.mean([
                balanced_accuracy_score(y[folds == f], scores[folds == f] > selection["raw_decision_threshold"])
                for f in (0, 1)
            ]), places=14)

    def test_zero_rule_and_strict_boundary_are_preserved(self):
        y, raw = np.array([0, 0, 1, 1]), np.array([-1., -.8, -.3, -.1])
        selection = select_threshold(y, raw, rule="zero")
        self.assertEqual(selection["raw_decision_threshold"], 0.)
        np.testing.assert_array_equal(apply_threshold(raw, 0.), raw)
        # Subtract in float64: a float32 score next to the learned threshold
        # must not lose its distinction by first rounding the threshold.
        threshold = float(raw[2])
        margins = apply_threshold(np.r_[raw, threshold], threshold)
        self.assertEqual(margins[-1], 0.)
        self.assertFalse(margins[-1] > 0)

    def test_adjacent_floats_do_not_split_score_ties_or_round_cutoff_up(self):
        lower = -1.
        upper = np.nextafter(lower, np.inf)
        raw = np.array([lower, lower, upper, upper])
        y = np.array([0, 0, 1, 1])
        selection = select_threshold(y, raw)
        self.assertEqual(selection["raw_decision_threshold"], lower)
        np.testing.assert_array_equal(apply_threshold(raw, lower) > 0, y)

    def test_invalid_scores_labels_and_validation_coverage_are_rejected(self):
        y = np.array([0, 0, 1, 1])
        for raw in ([0, 1], [0, 0, 1, np.nan], [0, 0, 1, np.inf]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                select_threshold(y, raw)
        with self.assertRaises(ValueError):
            select_threshold(y, np.arange(4), fold_ids=[0, 0, 1, 1])
        with self.assertRaises(ValueError):
            select_threshold(y, np.arange(4), rule="unknown")
        for splits in ([([], [0, 1]), ([], [1, 2, 3])], [([], [0, 1, 2])], [([], [0, 0, 1, 2, 3])]):
            with self.subTest(splits=splits), self.assertRaises(ValueError):
                validation_fold_ids(4, splits)

    def test_ridge_selects_alpha_under_the_selected_threshold_rule(self):
        y = np.tile([0, 0, 0, 1], 4)
        Z = np.random.default_rng(4).normal(size=(len(y), 3))
        inner = comparison.make_inner_splits(y, 2, 13)
        outputs = [np.stack((np.where(y[v], -.2, -.8), np.full(len(v), -.4))) for _, v in inner]
        with mock.patch.object(ridge, "ridge_grid_scores", side_effect=outputs):
            corrected = ridge.tune_alpha(Z, y, [1., 10.], inner)
        with mock.patch.object(ridge, "ridge_grid_scores", side_effect=outputs):
            zero = ridge.tune_alpha(Z, y, [1., 10.], inner, decision_rule="zero")
        self.assertEqual(corrected[0], 1.)
        self.assertEqual(zero[0], 10.)
        self.assertEqual(corrected[1][0]["inner_mean_balanced_accuracy"], 1.)
        self.assertEqual(corrected[1][0]["inner_mean_balanced_accuracy_zero"], .5)

    def test_finetune_epoch_selection_recognizes_improving_negative_scores(self):
        cohort = fixture()
        train, valid = np.arange(8), np.arange(8, 16)
        weak = np.full(len(valid), -.5)
        good = np.where(cohort.y[valid], -.2, -.8)
        options = training.TrainOptions(max_epochs=2, patience=2, batch_size=4, decision_rule="inner-balanced")
        with mock.patch.object(training, "predict_scores", side_effect=[weak, good]):
            _, result = training.fit_classifier(
                cohort.X[train], cohort.y[train], EncoderOptions("tspulse"), options, 3, None,
                validation=(cohort.X[valid], cohort.y[valid]), loader=toy_loader,
            )
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["best_validation_balanced_accuracy"], 1.)
        self.assertEqual(result["history"][-1]["validation_predicted_et_zero"], 0)
        self.assertEqual(result["history"][-1]["validation_balanced_accuracy_zero"], .5)
        np.testing.assert_array_equal(result["best_validation_scores"], good)

    def test_outer_labels_cannot_change_their_own_alpha_or_threshold(self):
        cohort = fixture()
        plan = comparison.make_plan(cohort.y, cohort.ids, 2, 2, 1, 71)
        altered_y = cohort.y.copy()
        altered_y[plan[0]["test_indices"]] = 1 - altered_y[plan[0]["test_indices"]]
        altered = replace(cohort, y=altered_y)
        Z = np.random.default_rng(1).normal(size=(len(cohort.y), 30)).astype(np.float32)
        args = comparison.parse_args(["--models", "tspulse", "--outer-folds", "2", "--inner-folds", "2",
                                      "--alphas", "1", "10", "--no-progress"])
        selections, alphas = [], []
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=2):
            for index, data in enumerate((cohort, altered)):
                destination = Path(temporary) / str(index)
                destination.mkdir()
                with mock.patch.object(encoders, "load_encoder", return_value=SimpleNamespace(provenance={})), \
                     mock.patch.object(encoders, "extract_features", return_value=Z), \
                     contextlib.redirect_stdout(io.StringIO()):
                    comparison.evaluate_frozen(data, plan, args, destination, EncoderOptions("tspulse"))
                fold = destination / "repeat_01_fold_01"
                selections.append(json.loads((fold / "threshold_selection.json").read_text()))
                alphas.append(json.loads((fold / "metrics.json").read_text())["selected_alpha"])
                self.assertTrue((fold / "metrics_raw_zero.json").is_file())
        self.assertEqual(selections[0], selections[1])
        self.assertEqual(alphas[0], alphas[1])

    def test_default_uses_inner_selection_and_broader_ridge_grid(self):
        args = comparison.parse_args([])
        self.assertEqual(args.decision_rule, "inner-balanced")
        self.assertEqual(args.alphas, np.logspace(-3, 9, 13).tolist())
        self.assertEqual(comparison.parse_args(["--decision-rule", "zero"]).decision_rule, "zero")


if __name__ == "__main__":
    unittest.main()
