"""Check participant/channel ordering and stateless frozen encoder extraction."""

from dataclasses import replace
from pathlib import Path
import io
import unittest
from unittest import mock

import numpy as np
import torch
from tqdm.auto import tqdm as real_tqdm

from sota_pd_et.data import PROJECT_ROOT
from sota_pd_et.transformer import EncoderConfig, _encode_batches, load_encoder
from sota_pd_et import transformer


class AdapterTests(unittest.TestCase):
    def test_channel_identity_and_batch_size_do_not_change_features(self):
        class TraceEncoder(torch.nn.Module):
            def forward(self, values):
                return values[:, 0, :]

        X = np.arange(3 * 4 * 976, dtype=np.float32).reshape(3, 4, 976)
        config = EncoderConfig(batch_size=5)
        Z = _encode_batches(X, TraceEncoder(), config, progress=False)
        expected = torch.nn.functional.interpolate(
            torch.from_numpy(X), size=512, mode="linear", align_corners=False,
        ).numpy().reshape(3, 4 * 512)
        np.testing.assert_array_equal(Z, expected)
        for batch_size in (1, 7):
            np.testing.assert_array_equal(
                Z, _encode_batches(X, TraceEncoder(), replace(config, batch_size=batch_size), progress=False),
            )

    def test_invalid_inputs_and_unpinned_configuration_rejected(self):
        for value in (np.empty((0, 2, 976)), np.full((1, 1, 976), np.nan), np.zeros((2, 976))):
            with self.assertRaisesRegex(ValueError, "Expected finite"):
                _encode_batches(value, None, EncoderConfig(), progress=False)
        with self.assertRaisesRegex(ValueError, "pinned"):
            EncoderConfig(revision="main")

    def test_feature_progress_counts_independent_channels_and_partial_batch(self):
        class TraceEncoder(torch.nn.Module):
            def forward(self, values):
                return values[:, 0]

        X = np.arange(2 * 3 * 976, dtype=np.float32).reshape(2, 3, 976)
        stream, bars = io.StringIO(), []

        def record_bar(*args, **kwargs):
            bar = real_tqdm(*args, **kwargs, file=stream)
            bars.append(bar)
            return bar

        with mock.patch.object(transformer, "tqdm", side_effect=record_bar):
            actual = _encode_batches(X, TraceEncoder(), EncoderConfig(batch_size=4), progress=True)
        expected = _encode_batches(X, TraceEncoder(), EncoderConfig(batch_size=4), progress=False)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual((bars[0].n, bars[0].total), (6, 6))
        self.assertTrue(bars[0].disable)
        self.assertIn("Initial MantisV2 features", stream.getvalue())

    @unittest.skipUnless((PROJECT_ROOT / "models/mantis_v2/model.safetensors").is_file(),
                         "Pinned checkpoint not downloaded locally.")
    def test_actual_checkpoint_frozen_finite_and_independent_of_other_people(self):
        torch.set_num_threads(2)
        config = EncoderConfig(batch_size=4)
        network, _ = load_encoder(config, PROJECT_ROOT / "models/mantis_v2")
        self.assertFalse(network.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in network.parameters()))
        rng = np.random.default_rng(7)
        X = rng.normal(size=(2, 2, 976)).astype(np.float32)
        X[0, 0] = 0.0  # A constant record must also give finite features.
        together = _encode_batches(X, network, config, progress=False)
        np.testing.assert_allclose(
            together[:1], _encode_batches(X[:1], network, replace(config, batch_size=1), progress=False),
            rtol=1e-5, atol=1e-5,
        )
        X[1] *= 1e4
        np.testing.assert_array_equal(together[:1], _encode_batches(X, network, config, progress=False)[:1])


if __name__ == "__main__":
    unittest.main()
