"""Overflow reproduction and parity with TiRex's unchanged upstream equations."""

import unittest
from dataclasses import replace

import torch
from tirex.models.slstm.cell import sLSTMCellTorch

from sota_pd_et.foundations.tirex_stable import stable_pointwise, stable_recurrence, recurrent_product
from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.tests.foundation_fixtures import tiny_encoder


class TiRexStabilityTests(unittest.TestCase):
    def test_recurrent_product_has_exact_native_forward_and_correct_analytic_derivatives(self):
        torch.manual_seed(14)
        hidden = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
        recurrent = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
        expected = hidden.unsqueeze(-2).matmul(recurrent.unsqueeze(0)).squeeze(-2)
        actual = recurrent_product(hidden, recurrent)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.autograd.gradcheck(recurrent_product, (hidden, recurrent)))

    def _overflow_case(self, device):
        outputs, gradients = [], []
        for function in (sLSTMCellTorch.slstm_forward_pointwise, stable_pointwise):
            inputs = torch.tensor([[-120., 0., .5, 0.]], device=device, requires_grad=True)
            output = function(inputs, torch.zeros_like(inputs), torch.zeros(4, device=device),
                              [torch.zeros(1, 1, device=device) for _ in range(4)])
            output[0].sum().backward()
            outputs.append(output[0].detach())
            gradients.append(inputs.grad)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        self.assertFalse(bool(torch.isfinite(gradients[0]).all()))
        self.assertTrue(bool(torch.isfinite(gradients[1]).all()))
        self.assertEqual(float(gradients[1][0, 0]), 0.)
        self.assertEqual(float(gradients[1][0, 1]), 0.)

    def test_overflowing_forget_gate_keeps_forward_values_and_fixes_backward(self):
        self._overflow_case("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_overflowing_forget_gate_has_finite_backward(self):
        self._overflow_case("mps")

    def test_pointwise_values_and_gradients_match_for_finite_native_arithmetic(self):
        generator = torch.Generator().manual_seed(15)
        values = [torch.randn(3, 16, generator=generator) for _ in range(2)]
        values += [torch.randn(16, generator=generator)]
        states = [torch.randn(3, 4, generator=generator) for _ in range(4)]
        states[2] = states[2].abs() + 1.
        results = []
        for function in (sLSTMCellTorch.slstm_forward_pointwise, stable_pointwise):
            arguments = [v.clone().requires_grad_() for v in values + states]
            output = function(*arguments[:3], arguments[3:])
            sum(value.square().mean() for value in output).backward()
            results.append((output, [argument.grad for argument in arguments]))
        for first, second in zip(*[r[0] for r in results]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        for first, second in zip(*[r[1] for r in results]):
            if first is None:
                self.assertIsNone(second)
            else:
                torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)

    def test_recurrence_retains_native_bfloat16_state_and_outputs(self):
        generator = torch.Generator().manual_seed(6)
        inputs = torch.randn(8, 2, 16, generator=generator).to(torch.bfloat16)
        states = torch.zeros(4, 2, 4, dtype=torch.bfloat16)
        recurrent = (.05 * torch.randn(2, 2, 8, generator=generator)).to(torch.bfloat16)
        bias = torch.zeros(16)
        actual = stable_recurrence(inputs, states, recurrent, bias)
        expected = sLSTMCellTorch.slstm_forward(inputs, states, recurrent, bias)
        for first, second in zip(actual, expected):
            self.assertEqual(first.dtype, torch.bfloat16)
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_channel_batching_and_recomputation_preserve_the_feature_function(self):
        torch.set_num_threads(2)
        torch.manual_seed(18)
        encoder = tiny_encoder(EncoderOptions("tirex")).train()
        values = torch.randn(2, 20, 64)
        direction = torch.linspace(-1., 1., encoder.feature_dimension(20))
        reference = None
        for chunk, checkpointing in ((8, True), (32, True), (32, False)):
            with self.subTest(chunk=chunk, checkpointing=checkpointing):
                encoder.options = replace(encoder.options, channel_batch_size=chunk, gradient_checkpointing=checkpointing)
                encoder.zero_grad(set_to_none=True)
                features = encoder(values)
                (features * direction).mean().backward()
                gradients = {name: value.grad.clone() for name, value in encoder.named_parameters() if value.grad is not None}
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients.values()))
                if reference is None:
                    reference = (features.detach().clone(), gradients)
                else:
                    torch.testing.assert_close(features, reference[0], rtol=1e-5, atol=1e-6)
                    # Native BF16 gradient reductions can round differently
                    # when channel sums are partitioned into different chunks.
                    for name, gradient in gradients.items():
                        before = reference[1][name]
                        error = (gradient - before).norm() / before.norm().clamp_min(1e-8)
                        self.assertLess(float(error), .025, name)

    def test_recurrent_weights_are_not_materialized_once_per_series_for_backward(self):
        batch, heads, width, sequence = 32, 4, 16, 4
        inputs = torch.randn(sequence, batch, 4 * heads * width, requires_grad=True)
        states = torch.zeros(4, batch, heads * width)
        recurrent = torch.randn(heads, width, 4 * width, requires_grad=True)
        bias = torch.zeros(4 * heads * width)
        saved = []
        for function in (sLSTMCellTorch.slstm_forward, stable_recurrence):
            sizes = []
            def record(tensor):
                sizes.append(tensor.untyped_storage().nbytes())
                return tensor
            with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
                function(inputs, states, recurrent, bias)
            saved.append(max(sizes))
        broadcast_bytes = batch * recurrent.untyped_storage().nbytes()
        self.assertGreaterEqual(saved[0], broadcast_bytes)
        self.assertLess(saved[1], broadcast_bytes)


if __name__ == "__main__":
    unittest.main()
