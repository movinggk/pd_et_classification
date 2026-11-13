"""Native features, active gradients, grouping, and checkpoint replay checks."""

import copy
from dataclasses import replace
import unittest
from unittest import mock

import torch

from sota_pd_et.foundations import encoders
from sota_pd_et.foundations.attention_fast import (
    chronos_native_group_features, chronos_packed_group_features, install_moment_position_cache,
)
from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.foundations.fast_paths import prepare_tivit_inputs
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


def restore_native_t5(model):
    for module in model.modules():
        if hasattr(module, "_comparison_native_compute_bias"):
            module.compute_bias = module._comparison_native_compute_bias
            del module._comparison_native_compute_bias


class FoundationLargeModelSpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_gradients_match(self, reference, optimized, rtol=3e-4, atol=3e-5):
        expected = dict(reference.named_parameters())
        active = 0
        for name, parameter in optimized.named_parameters():
            gradient = expected[name].grad
            if gradient is None:
                self.assertIsNone(parameter.grad, name)
            else:
                active += 1
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                torch.testing.assert_close(parameter.grad, gradient, rtol=rtol, atol=atol, msg=name)
        self.assertGreater(active, 0)

    def moment_equivalence(self, device):
        torch.manual_seed(910)
        optimized = tiny_encoder(EncoderOptions("moment", device=device)).backbone
        reference = copy.deepcopy(optimized)
        restore_native_t5(reference)
        original_keys = list(reference.state_dict())
        for training in (False, True):
            reference.train(training).zero_grad(set_to_none=True)
            optimized.train(training).zero_grad(set_to_none=True)
            values = torch.randn(2, 64, 32, device=device)
            mask = torch.ones(2, 64, dtype=torch.long, device=device)
            mask[0, :45] = 0
            mask[1, :7] = 0
            expected = reference.encoder(inputs_embeds=values, attention_mask=mask).last_hidden_state
            actual = optimized.encoder(inputs_embeds=values, attention_mask=mask).last_hidden_state
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            weights = torch.linspace(-1., 1., actual.numel(), device=device).reshape_as(actual)
            (expected * weights).sum().backward()
            (actual * weights).sum().backward()
            # MPS embedding-backward scatter additions can differ in their
            # reduction order even for identical bucket IDs and forward bits.
            self.assert_gradients_match(reference.encoder, optimized.encoder,
                                        rtol=3e-4 if device == "mps" else 0,
                                        atol=3e-5 if device == "mps" else 0)
        self.assertEqual(list(optimized.state_dict()), original_keys)
        optimized.load_state_dict(reference.state_dict(), strict=True)

    def test_moment_position_cache_keeps_native_attention_bias_mask_and_gradients(self):
        self.moment_equivalence("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_moment_relative_bias_and_mask_gradients_on_mps(self):
        self.moment_equivalence("mps")

    def test_moment_position_cache_covers_arbitrary_positions_and_updated_bias_weights(self):
        model = tiny_encoder(EncoderOptions("moment")).backbone.eval()
        first = model.encoder.block[0].layer[0].SelfAttention
        for device in ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",):
            first.to(device)
            positions = torch.tensor([-10000, -129, -128, -127, -33, -1, 0, 1, 17, 128, 129, 10000], device=device)
            for update in range(2):
                actual = first.compute_bias(12, 257, device=device, cache_position=positions)
                expected = first._comparison_native_compute_bias(12, 257, device=device, cache_position=positions)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                with torch.no_grad():
                    first.relative_attention_bias.weight.add_(.2)
        restore_native_t5(model)
        with mock.patch("sota_pd_et.foundations.attention_fast._version_is", return_value=False):
            self.assertEqual(install_moment_position_cache(model), 0)
        self.assertFalse(hasattr(first, "_comparison_native_compute_bias"))

    def chronos_equivalence(self, device):
        for length in (37, 64, 527):
            with self.subTest(length=length, device=device):
                torch.manual_seed(911)
                reference = tiny_encoder(EncoderOptions("chronos2", device=device)).backbone.train()
                optimized = copy.deepcopy(reference)
                values = torch.randn(3, 2, length, device=device)
                expected = chronos_native_group_features(reference, values)
                actual = chronos_packed_group_features(optimized, values, complete_context=True)
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
                weights = torch.linspace(-1., 1., actual.numel(), device=device).reshape_as(actual)
                (expected * weights).sum().backward()
                (actual * weights).sum().backward()
                self.assert_gradients_match(reference, optimized)
                self.assertEqual(list(reference.state_dict()), list(optimized.state_dict()))

    def test_chronos_packed_groups_match_native_features_and_gradients(self):
        self.chronos_equivalence("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_chronos_packed_groups_and_gradients_on_mps(self):
        self.chronos_equivalence("mps")

    def test_chronos_packing_avoids_dense_mask_and_keeps_tasks_independent(self):
        model = tiny_encoder(EncoderOptions("chronos2")).backbone.eval()
        values = torch.randn(3, 2, 37)
        with torch.no_grad(), mock.patch.object(
            model.encoder, "_construct_and_invert_group_time_mask", side_effect=AssertionError("Dense group mask allocated")
        ):
            expected = chronos_packed_group_features(model, values, complete_context=True)
            changed = values.clone()
            changed[1, 0] = torch.randn_like(changed[1, 0]) * 50
            actual = chronos_packed_group_features(model, changed, complete_context=True)
        torch.testing.assert_close(actual[[0, 2]], expected[[0, 2]], rtol=0, atol=0)
        self.assertFalse(torch.equal(actual[1], expected[1]))

    def test_chronos_incomplete_context_retains_native_mask_behavior(self):
        model = tiny_encoder(EncoderOptions("chronos2")).eval()
        values = torch.randn(2, 4, 64)
        values[0, :2, :32] = float("nan")
        model.options = replace(model.options, channel_batch_size=8, chronos_group_size=2)
        with torch.no_grad(), mock.patch.object(model.backbone, "encode", wraps=model.backbone.encode) as native:
            actual = model(values)
        native.assert_called_once()
        expected = chronos_native_group_features(model.backbone, values.reshape(4, 2, 64)).reshape(2, -1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_checkpoint_replay_does_not_repeat_weight_independent_preparation(self):
        for name, target in (("tivit", "prepare_tivit_inputs"), ("moment", "prepare_moment_input"),
                             ("vqshape", "prepare_vqshape_values"), ("utica", "resize_channel_batch"),
                             ("mantis_v2", "resize_channel_batch")):
            with self.subTest(model=name):
                torch.manual_seed(912)
                options = EncoderOptions(name, channel_batch_size=2, gradient_checkpointing=True)
                model = tiny_encoder(options).train()
                reference = tiny_encoder(replace(options, gradient_checkpointing=False)).train()
                reference.load_state_dict(model.state_dict(), strict=True)
                values = torch.randn(1, 4, 64)
                torch.manual_seed(913)
                expected = reference(values)
                torch.manual_seed(913)
                with mock.patch.object(encoders, target, wraps=getattr(encoders, target)) as prepare:
                    actual = model(values)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    (actual.square().mean()).backward()
                self.assertEqual(prepare.call_count, 2)
                (expected.square().mean()).backward()
                self.assert_gradients_match(reference, model, rtol=0, atol=0)

    def test_chronos_checkpointed_features_and_gradients_match_without_recomputation(self):
        options = EncoderOptions("chronos2", channel_batch_size=8, chronos_group_size=2, gradient_checkpointing=True)
        model = tiny_encoder(options).train()
        reference = copy.deepcopy(model)
        reference.options = replace(options, gradient_checkpointing=False)
        values = torch.randn(1, 8, 37)
        expected, actual = reference(values), model(values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().mean().backward()
        actual.square().mean().backward()
        self.assert_gradients_match(reference, model, rtol=0, atol=0)

    def test_tivit_checkpoint_input_is_one_uint8_channel(self):
        model = tiny_encoder(EncoderOptions("tivit")).backbone
        prepared, quantized = prepare_tivit_inputs(model, torch.randn(2, 976), patch_size=31)
        self.assertTrue(quantized)
        self.assertEqual(prepared.dtype, torch.uint8)
        self.assertEqual(tuple(prepared.shape), (2, 224, 224))
        self.assertFalse(prepared.requires_grad)
        self.assertEqual(prepared.numel() * prepared.element_size(), 2 * 3 * 224 * 224 * 4 // 12)


if __name__ == "__main__":
    unittest.main()
