"""Numerical and protocol equivalence for the shared ridge alpha search."""

import unittest
from unittest import mock

import numpy as np
from threadpoolctl import threadpool_limits

from sota_pd_et.evaluate import make_inner_splits
from sota_pd_et.evaluate_transformer import tune_alpha as reference_tune
from sota_pd_et.foundations import ridge
from sota_pd_et.model import fit_ridge


class FoundationRidgeTests(unittest.TestCase):
    def test_grid_matches_sklearn_for_imbalanced_primal_and_dual_problems(self):
        rng = np.random.default_rng(183)
        alphas = np.logspace(-3, 9, 13)
        for features in (9, 180):
            with self.subTest(features=features), threadpool_limits(limits=2):
                X = rng.normal(size=(48, features)) + rng.normal(size=features)
                V = rng.normal(size=(13, features)) + 3
                y = np.r_[np.zeros(41, dtype=int), np.ones(7, dtype=int)]
                original_X, original_V = X.copy(), V.copy()
                actual = ridge.ridge_grid_scores(X, y, V, alphas)
                expected = np.stack([fit_ridge(X, y, a).decision_function(V) for a in alphas])
                np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)
                np.testing.assert_array_equal(actual > 0, expected > 0)
                np.testing.assert_array_equal(X, original_X)
                np.testing.assert_array_equal(V, original_V)

    def test_rank_deficiency_and_zero_margin_keep_reference_decisions(self):
        rng = np.random.default_rng(91)
        base = rng.normal(size=(24, 2))
        X = np.column_stack((base, base, np.ones((24, 2))))
        V = np.column_stack((np.zeros((3, 4)), np.ones((3, 2))))
        y = np.tile([0, 1], 12)
        alphas = [1e-3, 1., 1000.]
        with threadpool_limits(limits=2):
            actual = ridge.ridge_grid_scores(X, y, V, alphas)
            expected = np.stack([fit_ridge(X, y, a).decision_function(V) for a in alphas])
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
            zeros = np.zeros((24, 50))
            scores = ridge.ridge_grid_scores(zeros, y, zeros[:3], alphas)
            np.testing.assert_array_equal(scores, np.zeros((3, 3)))

    def test_numerically_small_alpha_uses_reference_solver(self):
        rng = np.random.default_rng(22)
        X = rng.normal(size=(16, 30))
        y = np.tile([0, 1], 8)
        with mock.patch.object(ridge, "_reference_scores", wraps=ridge._reference_scores) as reference:
            ridge.ridge_grid_scores(X, y, X[:3], [1e-18, 1.])
        reference.assert_called_once()

    def test_search_selects_same_alpha_metrics_and_oof_scores(self):
        rng = np.random.default_rng(37)
        Z = rng.normal(size=(60, 300)).astype(np.float32)
        y = np.r_[np.zeros(48, int), np.ones(12, int)]
        inner = make_inner_splits(y, 3, 172)
        alphas = np.logspace(-3, 3, 7).tolist()
        with threadpool_limits(limits=2):
            expected = reference_tune(Z, y, alphas, inner)
            actual = ridge.tune_alpha(Z, y, alphas, inner, decision_rule="zero")
        self.assertEqual(actual[:2], expected[:2])
        np.testing.assert_allclose(actual[2], expected[2], rtol=2e-9, atol=2e-9)


if __name__ == "__main__":
    unittest.main()
