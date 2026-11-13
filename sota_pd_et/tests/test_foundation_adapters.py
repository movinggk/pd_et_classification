"""Actual upstream forward/backward passes; no pretrained weights or datasets."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import unittest

import numpy as np
import torch

from sota_pd_et.foundations.encoders import EncoderOptions, extract_features
from sota_pd_et.foundations.registry import SPECS
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


class FoundationAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.X = np.random.default_rng(81).normal(size=(3, 4, 64)).astype(np.float32)

    def test_each_real_architecture_has_finite_independent_frozen_features(self):
        for name in SPECS:
            with self.subTest(model=name):
                torch.manual_seed(17)
                options = EncoderOptions(name, channel_batch_size=4, chronos_group_size=2)
                encoder = tiny_encoder(options)
                Z = extract_features(encoder, self.X, batch_size=2)
                first = extract_features(encoder, self.X[:1], batch_size=1)
                changed = self.X.copy()
                changed[1:] *= 100
                mixed = extract_features(encoder, changed, batch_size=3)
                self.assertEqual(Z.shape[0], 3)
                self.assertGreater(Z.shape[1], 0)
                self.assertEqual(Z.shape[1], encoder.feature_dimension(self.X.shape[1]))
                self.assertTrue(np.isfinite(Z).all())
                np.testing.assert_allclose(Z[:1], first, rtol=1e-4, atol=2e-5)
                np.testing.assert_allclose(Z[:1], mixed[:1], rtol=1e-4, atol=2e-5)
                self.assertFalse(any(p.requires_grad for p in encoder.parameters()))

    def test_each_real_architecture_updates_backbone_with_checkpointing(self):
        for name in SPECS:
            with self.subTest(model=name):
                torch.manual_seed(19)
                encoder = tiny_encoder(EncoderOptions(name, channel_batch_size=2, chronos_group_size=2))
                encoder.train().requires_grad_(True)
                before = {k: p.detach().clone() for k, p in encoder.backbone.named_parameters()}
                optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
                output = encoder(torch.tensor(self.X[:2]))
                coefficients = torch.linspace(-1., 1., output.shape[-1])
                (output * coefficients).mean().backward()
                active = [(k, p) for k, p in encoder.backbone.named_parameters() if p.grad is not None and torch.count_nonzero(p.grad)]
                self.assertTrue(active, f"No backbone gradient for {name}")
                self.assertTrue(all(torch.isfinite(p.grad).all() for _, p in active))
                optimizer.step()
                self.assertTrue(any(not torch.equal(before[k], p) for k, p in active), f"No weight update for {name}")

    def test_chronos_groups_do_not_mix_task_segments(self):
        encoder = tiny_encoder(EncoderOptions("chronos2", chronos_group_size=2, channel_batch_size=8))
        encoder.eval()
        with torch.no_grad():
            original = encoder(torch.tensor(self.X[:1])).reshape(1, 4, -1)
            changed = torch.tensor(self.X[:1])
            changed[:, 2:] *= -20
            result = encoder(changed).reshape(1, 4, -1)
        torch.testing.assert_close(original[:, :2], result[:, :2])


if __name__ == "__main__":
    unittest.main()
