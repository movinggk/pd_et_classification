"""Real-architecture train/validate/train checks with generated participants."""

import unittest

import numpy as np
import torch

from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.foundations.registry import SPECS
from sota_pd_et.foundations.training import TrainOptions, fit_classifier, predict_scores
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


class FoundationTrainingLifecycleTests(unittest.TestCase):
    def test_all_eight_models_resume_training_after_validation(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(486)
        training = rng.normal(size=(8, 4, 64)).astype(np.float32)
        validation = rng.normal(size=(4, 4, 64)).astype(np.float32)
        labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        targets = np.array([0, 0, 1, 1])
        for name in SPECS:
            with self.subTest(model=name):
                model, result = fit_classifier(
                    training, labels,
                    EncoderOptions(name, channel_batch_size=8, chronos_group_size=2),
                    TrainOptions(max_epochs=2, patience=3, batch_size=2, accumulation_steps=2),
                    71, None, validation=(validation, targets), loader=tiny_encoder,
                )
                self.assertEqual(result["epochs_trained"], 2)
                self.assertTrue(result["backbone_gradient_verified"])
                self.assertTrue(all(np.isfinite(row["training_weighted_bce"]) for row in result["history"]))
                self.assertTrue(np.isfinite(predict_scores(model, validation)).all())


if __name__ == "__main__":
    unittest.main()
