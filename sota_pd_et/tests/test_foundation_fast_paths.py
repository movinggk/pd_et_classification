"""Compare optimized features/gradients against actual upstream code paths."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.foundations.fast_paths import (
    tirex_hidden_features, vqshape_token_features, tivit_tensor_features,
    _tivit_grayscale_pixel_values, _tivit_processor_key,
)
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


def original_features(name, model, values):
    if name == "vqshape":
        return model(values, mode="tokenize")[0]["token"].flatten(1)
    inputs, padding = model._prepare_context_for_embedding(values, None)
    _, hidden = model._forward_model(inputs, return_all_hidden=True)
    return F.normalize(hidden[:, padding:].mean(1), dim=-1).flatten(1)


FAST = {"vqshape": vqshape_token_features, "tirex": tirex_hidden_features}


class FoundationFastPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_feature_values_are_identical_in_eval_and_seeded_single_training_call(self):
        for name in FAST:
            for is_training in (False, True):
                with self.subTest(model=name, training=is_training):
                    torch.manual_seed(18)
                    reference = tiny_encoder(EncoderOptions(name)).backbone
                    optimized = copy.deepcopy(reference)
                    reference.train(is_training)
                    optimized.train(is_training)
                    lengths = [512] if name == "vqshape" else [37, 64]
                    for length in lengths:
                        values = torch.randn(3, length)
                        torch.manual_seed(121)
                        before = original_features(name, reference, values)
                        torch.manual_seed(121)
                        after = FAST[name](optimized, values)
                        self.assertTrue(torch.isfinite(after).all())
                        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_input_and_used_parameter_gradients_are_identical(self):
        for name in FAST:
            for is_training in (False, True):
                with self.subTest(model=name, training=is_training):
                    torch.manual_seed(19)
                    reference = tiny_encoder(EncoderOptions(name)).backbone.train(is_training)
                    optimized = copy.deepcopy(reference)
                    # Raw context is not optimized by the classifier. Use a
                    # non-padded TiRex input for its optional input-gradient
                    # check: upstream NaN padding has undefined input VJPs.
                    length = 512 if name == "vqshape" else 64
                    left = torch.randn(2, length, requires_grad=True)
                    right = left.detach().clone().requires_grad_(True)
                    torch.manual_seed(122)
                    before = original_features(name, reference, left)
                    torch.manual_seed(122)
                    after = FAST[name](optimized, right)
                    coefficients = torch.linspace(-1., 1., before.shape[-1])
                    (before * coefficients).sum().backward()
                    (after * coefficients).sum().backward()
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                    reference_parameters = dict(reference.named_parameters())
                    active = 0
                    for key, parameter in optimized.named_parameters():
                        expected = reference_parameters[key].grad
                        if expected is None:
                            self.assertIsNone(parameter.grad, key)
                        else:
                            active += 1
                            self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                            torch.testing.assert_close(expected, parameter.grad, rtol=0, atol=0)
                    self.assertGreater(active, 0)

    def test_tirex_parameter_gradients_match_with_padded_context(self):
        torch.manual_seed(20)
        reference = tiny_encoder(EncoderOptions("tirex")).backbone.train()
        optimized = copy.deepcopy(reference)
        values = torch.randn(2, 37)
        before = original_features("tirex", reference, values)
        after = tirex_hidden_features(optimized, values)
        coefficients = torch.linspace(-1., 1., before.shape[-1])
        (before * coefficients).sum().backward()
        (after * coefficients).sum().backward()
        reference_parameters = dict(reference.named_parameters())
        active = 0
        for key, parameter in optimized.named_parameters():
            expected = reference_parameters[key].grad
            if expected is None:
                self.assertIsNone(parameter.grad, key)
            else:
                active += 1
                self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                torch.testing.assert_close(expected, parameter.grad, rtol=0, atol=0)
        self.assertGreater(active, 0)

    def test_unused_prediction_modules_are_not_called_and_state_layout_is_preserved(self):
        for name in FAST:
            with self.subTest(model=name):
                model = tiny_encoder(EncoderOptions(name)).backbone.eval()
                saved = {key: value.clone() for key, value in model.state_dict().items()}
                unused = [model.decoder, model.shape_decoder, model.attr_encoder, model.codebook] if name == "vqshape" else [model.out_norm, model.output_patch_embedding]
                patches = [mock.patch.object(module, "forward", side_effect=AssertionError("Unused computation called")) for module in unused]
                import contextlib
                with contextlib.ExitStack() as stack:
                    for patch in patches:
                        stack.enter_context(patch)
                    values = torch.randn(2, 512 if name == "vqshape" else 64)
                    FAST[name](model, values)
                current = model.state_dict()
                self.assertEqual(saved.keys(), current.keys())
                for key in saved:
                    torch.testing.assert_close(saved[key], current[key], rtol=0, atol=0)
                model.load_state_dict(saved, strict=True)

    def test_tivit_quantized_processor_pixels_match_exactly(self):
        model = tiny_encoder(EncoderOptions("tivit")).backbone.eval()
        # Include a dense uint8-boundary palette and ordinary synthetic images.
        ramp = (torch.arange(224 * 224).reshape(1, 1, 224, 224) % 256).float() / 255
        ramp = ramp.repeat(1, 3, 1, 1)
        values = torch.randn(2, 64, 1)
        images = model.ts2image_transformation(values, model.patch_size, model.stride)
        for image_batch in (ramp, images):
            expected = model.processor(images=[model.to_pil(im) for im in image_batch],
                                       return_tensors="pt")["pixel_values"]
            actual = _tivit_grayscale_pixel_values(model, image_batch, _tivit_processor_key(model))
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_tivit_features_and_parameter_gradients_match_with_no_host_processor(self):
        for is_training in (False, True):
            with self.subTest(training=is_training):
                torch.manual_seed(23)
                reference = tiny_encoder(EncoderOptions("tivit")).backbone.train(is_training)
                optimized = copy.deepcopy(reference)
                saved = {key: value.clone() for key, value in optimized.state_dict().items()}
                values = torch.randn(2, 64)
                expected = reference(values[:, :, None])
                with mock.patch.object(optimized, "forward_vit", side_effect=AssertionError("Host path called")), \
                     mock.patch.object(optimized.processor, "preprocess", side_effect=AssertionError("Host processor called")):
                    actual = tivit_tensor_features(optimized, values, patch_size=31)
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                coefficients = torch.linspace(-1., 1., actual.shape[-1])
                (expected * coefficients).sum().backward()
                (actual * coefficients).sum().backward()
                expected_parameters = dict(reference.named_parameters())
                for key, parameter in optimized.named_parameters():
                    before = expected_parameters[key].grad
                    if before is None:
                        self.assertIsNone(parameter.grad, key)
                    else:
                        torch.testing.assert_close(before, parameter.grad, rtol=0, atol=0)
                current = optimized.state_dict()
                self.assertEqual(saved.keys(), current.keys())
                for key in saved:
                    torch.testing.assert_close(saved[key], current[key], rtol=0, atol=0)

    def test_tivit_falls_back_for_nonstandard_processor_settings(self):
        model = tiny_encoder(EncoderOptions("tivit")).backbone.eval()
        model.processor.size = {"shortest_edge": 256}
        values = torch.randn(1, 64)
        with mock.patch.object(model, "forward_vit", wraps=model.forward_vit) as original:
            actual = tivit_tensor_features(model, values, patch_size=31)
        original.assert_called_once()
        expected = model(values[:, :, None])
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
