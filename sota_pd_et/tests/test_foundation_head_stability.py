"""Regression checks for coherent Adam steps in very wide participant heads."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from sota_pd_et import compare_foundations as comparison
from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.foundations.training import (
    FineTunedClassifier, TrainOptions, fit_classifier, predict_scores,
    resolve_head_learning_rate, seed_everything,
)


class WideSyntheticEncoder(nn.Module):
    def __init__(self, options, width):
        super().__init__()
        self.options = options
        self.width = width
        self.backbone = nn.Parameter(torch.ones(()))
        self.provenance = {"synthetic_test_only": True}

    def feature_dimension(self, channels):
        return self.width

    def forward(self, values):
        return self.backbone * values[:, 0]


class FoundationHeadStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_coherent_first_adam_step_remains_small_at_real_adapter_widths(self):
        for width in (19008, 67584):
            with self.subTest(width=width):
                changes = []
                for requested in (None, 1e-3):
                    seed_everything(71)
                    model = FineTunedClassifier(
                        WideSyntheticEncoder(EncoderOptions("tspulse"), width), width, dropout=0.,
                    )
                    with torch.no_grad():
                        model.head[-1].weight.zero_()
                        model.head[-1].bias.fill_(1.)
                    values = torch.linspace(-1., 1., width).reshape(1, 1, width)
                    optimizer = torch.optim.AdamW(
                        model.head.parameters(), lr=resolve_head_learning_rate(width, requested),
                        weight_decay=.01,
                    )
                    before = model(values).detach()
                    nn.functional.binary_cross_entropy_with_logits(model(values), torch.zeros(1)).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    changes.append(float((model(values).detach() - before).abs()))
                # The explicit old rate can change the score by tens in one
                # update despite gradient clipping. The default stays bounded.
                self.assertLess(changes[0], .3)
                self.assertGreater(changes[1], 10.)

    def test_default_head_learns_a_wide_imbalanced_held_out_signal(self):
        width = 19008
        rng = np.random.default_rng(583)
        common = rng.normal(0., 2., size=width)
        direction = rng.choice([-1., 1.], size=width)

        def dataset(pd, et):
            labels = np.r_[np.zeros(pd, dtype=int), np.ones(et, dtype=int)]
            values = common + .5 * (2 * labels[:, None] - 1) * direction
            values += rng.normal(0., .2, size=values.shape)
            return values.astype(np.float32)[:, None], labels

        training, labels = dataset(40, 8)
        validation, targets = dataset(20, 4)
        model, result = fit_classifier(
            training, labels, EncoderOptions("tspulse"),
            TrainOptions(learning_rate=1e-7, dropout=.1, batch_size=2, accumulation_steps=4),
            71, None, epochs=4,
            loader=lambda options, _: WideSyntheticEncoder(options, width),
        )
        scores = predict_scores(model, validation, batch_size=8)
        self.assertGreaterEqual(balanced_accuracy_score(targets, scores > 0), .95)
        self.assertLess(result["history"][-1]["training_weighted_bce"],
                        result["history"][0]["training_weighted_bce"])
        self.assertTrue(result["backbone_gradient_verified"])
        self.assertEqual(result["head_learning_rate_policy"], "auto_width_scaled")
        self.assertIsNone(result["requested_head_learning_rate"])
        self.assertLess(result["train_options"]["head_learning_rate"], 1e-4)
        self.assertEqual(result["training_class_counts"], {"pd": 40, "et": 8})

    def test_explicit_rates_are_preserved_and_invalid_rates_are_rejected(self):
        self.assertEqual(resolve_head_learning_rate(67584, 1e-3), 1e-3)
        self.assertIsNone(comparison.parse_args([]).head_learning_rate)
        self.assertEqual(comparison.parse_args(["--head-learning-rate", "0.001"]).head_learning_rate, .001)
        for rate in (0., -1., float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                TrainOptions(head_learning_rate=rate)
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                resolve_head_learning_rate(67584, rate)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                comparison.parse_args(["--head-learning-rate", str(rate)])

    def test_score_diagnostics_expose_single_class_without_changing_scores(self):
        targets = np.array([0, 0, 0, 1])
        for scores, label, count in ((np.array([1., 2., 3., 4.]), "ET", 4),
                                      (np.array([-4., -3., -2., 0.]), "PD", 0)):
            with self.subTest(label=label):
                original = scores.copy()
                result = comparison.metrics(targets, scores)
                diagnostics = comparison.score_diagnostics(scores)
                self.assertEqual(result["balanced_accuracy"], .5)
                self.assertEqual(diagnostics["single_predicted_class"], label)
                self.assertEqual(diagnostics["predicted_et"], count)
                self.assertEqual(diagnostics["decision_threshold"], 0.)
                stream = io.StringIO()
                with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(stream):
                    comparison.report_fold("tspulse", "finetune", 1, 1, result, scores, Path(folder), False)
                    self.assertTrue((Path(folder) / "score_diagnostics.json").is_file())
                self.assertIn(f"predicted ET={count}/4", stream.getvalue())
                self.assertIn("ET recall=", stream.getvalue())
                self.assertIn("AUROC=", stream.getvalue())
                np.testing.assert_array_equal(scores, original)


if __name__ == "__main__":
    unittest.main()
