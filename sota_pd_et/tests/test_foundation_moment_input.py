"""Author-supported MOMENT sizing, masks, and native embedding checks."""

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from sota_pd_et import predict_foundation
from sota_pd_et.foundations.encoders import Encoder, EncoderOptions, extract_features
from sota_pd_et.foundations.moment_input import prepare_moment_input
from sota_pd_et.model import fit_ridge
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


class FoundationMomentInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_long_resampling_matches_author_endpoint_inclusive_scipy_grid(self):
        values = torch.randn(3, 976, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(714))
        actual, mask = prepare_moment_input(values)
        reference = interp1d(np.linspace(0., 1., 976), values.numpy())(np.linspace(0., 1., 512))
        np.testing.assert_allclose(actual[:, 0].numpy(), reference, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(actual[:, 0, 0], values[:, 0], rtol=0., atol=0.)
        torch.testing.assert_close(actual[:, 0, -1], values[:, -1], rtol=0., atol=0.)
        self.assertTrue(bool(torch.all(mask == 1)))

    def test_short_inputs_retain_samples_and_mask_left_padding(self):
        values = torch.randn(2, 61, generator=torch.Generator().manual_seed(193), requires_grad=True)
        actual, mask = prepare_moment_input(values)
        self.assertEqual(tuple(actual.shape), (2, 1, 512))
        self.assertTrue(bool(torch.all(actual[:, :, :451] == 0)))
        self.assertTrue(bool(torch.all(mask[:, :451] == 0)))
        self.assertTrue(bool(torch.all(mask[:, 451:] == 1)))
        torch.testing.assert_close(actual[:, 0, 451:], values, rtol=0., atol=0.)
        actual.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0., atol=0.)

    def test_exact_target_length_is_unchanged_without_a_second_normalization(self):
        values = torch.linspace(-150., 250., 512).repeat(2, 1)
        actual, mask = prepare_moment_input(values)
        self.assertEqual(actual.data_ptr(), values.data_ptr())
        torch.testing.assert_close(actual[:, 0], values, rtol=0., atol=0.)
        self.assertTrue(bool(torch.all(mask == 1)))

    def test_native_masked_mean_pooling_ignores_padded_patch_outputs(self):
        torch.manual_seed(91)
        backbone = tiny_encoder(EncoderOptions("moment")).backbone.eval()
        values = torch.randn(2, 61, generator=torch.Generator().manual_seed(117))
        inputs, mask = prepare_moment_input(values, backbone.seq_len, backbone.patch_len)
        from momentfm.utils.masking import Masking

        patch_mask = Masking.convert_seq_to_patch_view(mask, backbone.patch_len)
        self.assertEqual(int(patch_mask[0].sum()), 7)
        with torch.no_grad():
            hidden = backbone(x_enc=inputs, input_mask=mask, reduction="none").embeddings[:, 0]
            weights = patch_mask.unsqueeze(-1)
            reference = (hidden * weights).sum(1) / weights.sum(1)
            actual = backbone(x_enc=inputs, input_mask=mask, reduction="mean").embeddings
        torch.testing.assert_close(actual, reference, rtol=0., atol=0.)
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_sizing_preserves_native_input_and_parameter_gradients(self):
        torch.manual_seed(91)
        original = tiny_encoder(EncoderOptions("moment")).backbone.eval()
        optimized = copy.deepcopy(original)
        values = torch.randn(2, 61, generator=torch.Generator().manual_seed(117), requires_grad=True)
        other = values.detach().clone().requires_grad_()
        original_inputs = F.pad(values[:, None], (451, 0), value=0.)
        original_mask = torch.ones(2, 512, dtype=torch.long)
        original_mask[:, :451] = 0
        optimized_inputs, optimized_mask = prepare_moment_input(other)
        expected = original(x_enc=original_inputs, input_mask=original_mask, reduction="mean").embeddings
        actual = optimized(x_enc=optimized_inputs, input_mask=optimized_mask, reduction="mean").embeddings
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        coefficient = torch.linspace(-1., 1., expected.numel()).reshape_as(expected)
        (expected * coefficient).sum().backward()
        (actual * coefficient).sum().backward()
        torch.testing.assert_close(other.grad, values.grad, rtol=0., atol=0.)
        self.assertGreater(int(torch.count_nonzero(other.grad)), 0)
        for (name, parameter), (other_name, parameter_other) in zip(original.named_parameters(), optimized.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(parameter.grad is None, parameter_other.grad is None, name)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter_other.grad, parameter.grad, rtol=0., atol=0., msg=name)
                self.assertTrue(bool(torch.isfinite(parameter_other.grad).all()), name)

    def test_adapter_matches_native_padded_and_endpoint_resized_embedding(self):
        torch.manual_seed(91)
        encoder = tiny_encoder(EncoderOptions("moment")).eval()
        for length in (61, 512, 976):
            with self.subTest(length=length):
                values = torch.randn(2, 2, length, generator=torch.Generator().manual_seed(length))
                flat = values.reshape(-1, length)
                inputs, mask = prepare_moment_input(flat, encoder.backbone.seq_len, encoder.backbone.patch_len)
                with torch.no_grad():
                    expected = encoder.backbone(x_enc=inputs, input_mask=mask, reduction="mean").embeddings.reshape(2, -1)
                    actual = encoder(values)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_short_or_invalid_inputs_fail_before_native_nan_pooling(self):
        for length in (1, 2, 7):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "fully observed"):
                prepare_moment_input(torch.ones(2, length))
        for values in (torch.ones(2, 8, dtype=torch.long), torch.ones(2, 1, 8), torch.ones(0, 8)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                prepare_moment_input(values)
        for target, patch in ((510, 8), (512, 0), (512., 8), (512, True)):
            with self.subTest(target=target, patch=patch), self.assertRaises(ValueError):
                prepare_moment_input(torch.ones(2, 16), target, patch)

    def test_current_frozen_artifact_prediction_preserves_features(self):
        X = torch.randn(16, 2, 61, generator=torch.Generator().manual_seed(581)).numpy()
        labels = np.tile([0, 1], 8)
        torch.manual_seed(91)
        encoder = tiny_encoder(EncoderOptions("moment")).eval()
        features = extract_features(encoder, X, batch_size=2)
        scaler = StandardScaler().fit(features)
        classifier = fit_ridge(scaler.transform(features), labels, 1.)
        expected = classifier.decision_function(scaler.transform(features))
        bundle = {
            "format_version": 1, "model_type": "foundation_comparison", "mode": "frozen",
            "encoder_options": asdict(encoder.options), "encoder_provenance": encoder.provenance,
            "input_shape": [2, 61], "feature_dimension": features.shape[1],
            "scaler": scaler, "classifier": classifier,
        }
        resolved_versions = []

        def loader(resolved, cache_dir):
            resolved_versions.append(resolved.adapter_version)
            return Encoder(copy.deepcopy(encoder.backbone), resolved, encoder.provenance)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(bundle, path)
            with mock.patch.object(predict_foundation, "load_encoder", side_effect=loader):
                actual = predict_foundation.predict(path, X)
        self.assertEqual(resolved_versions, [3])
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


class FoundationChronosBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_pipeline_matches_partial_patches_and_context_truncation(self):
        from chronos.chronos2 import Chronos2Pipeline

        torch.manual_seed(159)
        encoder = tiny_encoder(EncoderOptions("chronos2", channel_batch_size=8, chronos_group_size=2)).eval()
        pipeline = Chronos2Pipeline(encoder.backbone)
        for length in (2, 7, 8, 9, 63, 64, 65, 511, 512, 513, 976):
            with self.subTest(length=length):
                values = torch.randn(2, 4, length, generator=torch.Generator().manual_seed(length))
                embeddings, _ = pipeline.embed(values.reshape(-1, 2, length), batch_size=8)
                # Native fixture has REG+one future token. Context pooling is
                # the comparison readout; the underlying embeddings are native.
                expected = torch.stack([item[:, :-2].mean(1) for item in embeddings]).reshape(2, -1)
                with torch.no_grad():
                    actual = encoder(values)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
