"""Check the classification-only path against the actual pinned upstream model."""

import copy
import unittest
from unittest.mock import patch

import torch

from sota_pd_et.foundations.tspulse_fast import (
    supports_tspulse_hidden_fast_path, tspulse_last_hidden,
)


class FoundationTSPulseFastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def backbone(self, **overrides):
        from tsfm_public.models.tspulse import TSPulseConfig, TSPulseForReconstruction

        torch.manual_seed(219)
        options = dict(
            context_length=512, patch_length=16, patch_stride=16, d_model=8,
            decoder_d_model=8, num_layers=2, decoder_num_layers=1,
            num_input_channels=1, mode="common_channel", decoder_mode="common_channel",
            patch_register_tokens=2, fuse_fft=True, dropout=.2, head_dropout=.2,
            scaling="revin", revin_affine=True, enable_fft_prob_loss=True,
            register_mixer_layers=1, fft_time_consistent_masking=True,
        )
        options.update(overrides)
        result = TSPulseForReconstruction(TSPulseConfig(**options)).backbone
        if options["scaling"] == "revin":
            # The ground-truth standard scaler has no affine transform. These
            # values catch accidentally taking its FFT instead of RevIN's FFT.
            with torch.no_grad():
                result.scaler.affine_weight.fill_(1.37)
                result.scaler.affine_bias.fill_(.23)
        return result

    def values(self):
        return torch.randn(3, 512, 1, generator=torch.Generator().manual_seed(912))

    def test_eval_hidden_is_bitwise_equal_and_one_fft_is_removed(self):
        from tsfm_public.models.tspulse import modeling_tspulse

        backbone = self.backbone().eval()
        self.assertTrue(supports_tspulse_hidden_fast_path(backbone))
        keys = tuple(backbone.state_dict())
        with torch.no_grad(), patch.object(modeling_tspulse, "get_fft", wraps=modeling_tspulse.get_fft) as fft:
            expected = backbone(past_values=self.values(), enable_masking=False,
                                return_dict=True).last_hidden_state
            self.assertEqual(fft.call_count, 2)
            fft.reset_mock()
            actual = tspulse_last_hidden(backbone, self.values())
            self.assertEqual(fft.call_count, 1)
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        self.assertEqual(tuple(backbone.state_dict()), keys)

    def test_seeded_training_outputs_and_every_gradient_match(self):
        for fft_input in ("scaled_ts", "raw_ts"):
            with self.subTest(fft_input=fft_input):
                original = self.backbone(fft_applied_on=fft_input).train()
                optimized = copy.deepcopy(original)
                x = self.values().requires_grad_()
                y = x.detach().clone().requires_grad_()
                torch.manual_seed(719)
                expected = original(past_values=x, enable_masking=False,
                                    return_dict=True).last_hidden_state
                torch.manual_seed(719)
                actual = tspulse_last_hidden(optimized, y)
                torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
                weights = torch.linspace(-.7, 1.1, expected.numel()).reshape_as(expected)
                (expected * weights).sum().backward()
                (actual * weights).sum().backward()
                torch.testing.assert_close(y.grad, x.grad, rtol=0., atol=0.)
                for (name, parameter), (other_name, other) in zip(
                        original.named_parameters(), optimized.named_parameters()):
                    self.assertEqual(name, other_name)
                    self.assertEqual(parameter.grad is None, other.grad is None, name)
                    if parameter.grad is not None:
                        torch.testing.assert_close(other.grad, parameter.grad, rtol=0., atol=0., msg=name)
                self.assertGreater(float(original.scaler.affine_weight.grad.abs().sum()), 0.)
                self.assertGreater(float(original.scaler.affine_bias.grad.abs().sum()), 0.)

    def test_std_scaler_hidden_matches(self):
        backbone = self.backbone(scaling="std").eval()
        with torch.no_grad():
            expected = backbone(past_values=self.values(), enable_masking=False,
                                return_dict=True).last_hidden_state
            actual = tspulse_last_hidden(backbone, self.values())
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)

    def test_unverified_config_and_version_fall_back_to_upstream(self):
        from tsfm_public.models.tspulse import modeling_tspulse

        for overrides in ({"fft_time_consistent_masking": False}, {"channel_virtual_expand_scale": 2}):
            with self.subTest(overrides=overrides):
                backbone = self.backbone(**overrides).eval()
                self.assertFalse(supports_tspulse_hidden_fast_path(backbone))
                with torch.no_grad(), patch.object(backbone, "forward", wraps=backbone.forward) as forward:
                    actual = tspulse_last_hidden(backbone, self.values())
                    forward.assert_called_once()
                    self.assertFalse(forward.call_args.kwargs["enable_masking"])
                    expected = backbone(past_values=self.values(), enable_masking=False,
                                        return_dict=True).last_hidden_state
                torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        backbone = self.backbone().eval()
        with patch("sota_pd_et.foundations.tspulse_fast._verified_runtime", return_value=False):
            self.assertFalse(supports_tspulse_hidden_fast_path(backbone))
            with torch.no_grad(), patch.object(modeling_tspulse, "get_fft", wraps=modeling_tspulse.get_fft) as fft:
                tspulse_last_hidden(backbone, self.values())
                self.assertEqual(fft.call_count, 2)


if __name__ == "__main__":
    unittest.main()
