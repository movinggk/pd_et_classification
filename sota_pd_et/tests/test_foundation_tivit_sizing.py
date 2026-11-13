"""Author sqrt(T) signal sizing without mutable TiViT model settings."""

import copy
import math
import unittest
from unittest import mock

import torch

from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.foundations.fast_paths import tivit_tensor_features
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


class FoundationTiViTSizingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        torch.manual_seed(932)
        return tiny_encoder(EncoderOptions("tivit")).backbone.eval()

    def original(self, model, values, patch_size):
        images = model.ts2image_transformation(values[:, :, None], patch_size=patch_size,
                                               stride=model.stride)
        return model.aggregate_hidden_representations(model.forward_vit(images), model.aggregation)

    def test_author_sqrt_rule_matches_original_images_and_features(self):
        model = self.model()
        for length in (8, 64, 976):
            with self.subTest(length=length), torch.no_grad():
                values = torch.randn(2, length, generator=torch.Generator().manual_seed(length))
                patch = math.isqrt(length)
                expected = self.original(model, values, patch)
                actual = tivit_tensor_features(model, values, patch_size=patch)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(model.patch_size, 31)
        self.assertEqual(math.isqrt(976), 31)

    def test_pads_length_matches_original_features(self):
        model = self.model()
        values = torch.randn(2, 976)
        with torch.no_grad():
            original = model(values[:, :, None])
            dynamic = tivit_tensor_features(model, values, patch_size=31)
        torch.testing.assert_close(dynamic, original, rtol=0, atol=0)

    def test_local_override_preserves_vision_parameter_gradients(self):
        reference = self.model().train()
        optimized = copy.deepcopy(reference)
        values = torch.randn(2, 64)
        expected = self.original(reference, values, 8)
        actual = tivit_tensor_features(optimized, values, patch_size=8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.linspace(-1., 1., actual.shape[-1])
        (expected * weights).sum().backward()
        (actual * weights).sum().backward()
        reference_parameters = dict(reference.named_parameters())
        active = 0
        for name, parameter in optimized.named_parameters():
            expected_gradient = reference_parameters[name].grad
            if expected_gradient is None:
                self.assertIsNone(parameter.grad)
            else:
                active += 1
                torch.testing.assert_close(parameter.grad, expected_gradient, rtol=0, atol=0)
        self.assertGreater(active, 0)
        self.assertEqual(reference.patch_size, 31)
        self.assertEqual(optimized.patch_size, 31)

    def test_nonstandard_processor_fallback_keeps_the_local_override(self):
        model = self.model()
        model.processor.size = {"shortest_edge": 256}
        values = torch.randn(2, 64)
        with torch.no_grad():
            expected = self.original(model, values, 8)
            with mock.patch.object(model, "forward", side_effect=AssertionError("Stored patch size used")), \
                 mock.patch.object(model, "forward_vit", wraps=model.forward_vit) as original_processor:
                actual = tivit_tensor_features(model, values, patch_size=8)
                original_processor.assert_called_once()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(model.patch_size, 31)

    def test_nonstandard_image_shape_fallback_keeps_the_local_override(self):
        model = self.model()
        values = torch.randn(2, 64)
        transform = model.ts2image_transformation

        def smaller_image(*args, **kwargs):
            return transform(*args, **kwargs)[:, :, :200, :200]

        with mock.patch.object(model, "ts2image_transformation", side_effect=smaller_image), torch.no_grad():
            expected = self.original(model, values, 8)
            with mock.patch.object(model, "forward", side_effect=AssertionError("Stored patch size used")):
                actual = tivit_tensor_features(model, values, patch_size=8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(model.patch_size, 31)

    def test_invalid_patch_overrides_are_rejected(self):
        for patch in (0, -1, 65, True, 8.):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                tivit_tensor_features(self.model(), torch.randn(2, 64), patch_size=patch)


if __name__ == "__main__":
    unittest.main()
