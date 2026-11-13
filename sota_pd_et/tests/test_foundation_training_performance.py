"""Numerical and safety checks for training data-transfer optimizations."""

from dataclasses import replace
import unittest
from unittest import mock

import numpy as np
import torch

from sota_pd_et.foundations import training
from sota_pd_et.foundations.encoders import EncoderOptions
from sota_pd_et.tests.test_foundation_protocol import ToyEncoder, fixture, toy_loader


class FoundationTrainingPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_accumulation_matches_full_effective_batches_with_unequal_classes(self):
        cohort = fixture()
        y = np.r_[np.zeros(12, dtype=int), np.ones(4, dtype=int)]
        base = training.TrainOptions(max_epochs=2, batch_size=9, accumulation_steps=1, dropout=0.)
        models = []
        for options in (base, replace(base, batch_size=3, accumulation_steps=3)):
            model, result = training.fit_classifier(
                cohort.X, y, EncoderOptions("tspulse"), options, 71, None,
                epochs=2, loader=toy_loader,
            )
            self.assertTrue(result["backbone_gradient_verified"])
            models.append(model)
        # There are effective batches of 9 and 7 participants each epoch.
        for left, right in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)

    def test_gradient_audit_checks_every_parameter_and_nonfinite_loss(self):
        first = torch.nn.Parameter(torch.zeros(2))
        last = torch.nn.Parameter(torch.zeros(2))
        first.grad = torch.tensor([1., 0.])
        for bad in (float("nan"), float("inf")):
            last.grad = torch.tensor([0., bad])
            with self.assertRaisesRegex(RuntimeError, "Nonfinite backbone gradients"):
                training._audit_backbone_gradients((first, last), torch.tensor(True))
        last.grad = torch.zeros(2)
        with self.assertRaisesRegex(RuntimeError, "Nonfinite fine-tuning loss"):
            training._audit_backbone_gradients((first, last), torch.tensor(False))
        first.grad = None
        last.grad = None
        with self.assertRaisesRegex(RuntimeError, "No backbone gradients"):
            training._audit_backbone_gradients((first, last), torch.tensor(True))

    def test_gradient_audit_retains_subnormal_nonzero_gradients(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([1e-40, 0.])
        self.assertTrue(training._audit_backbone_gradients((parameter,), torch.tensor(True)))
        parameter.grad.zero_()
        self.assertFalse(training._audit_backbone_gradients((parameter,), torch.tensor(True)))

    def test_nonfinite_gradient_on_later_update_is_rejected_before_optimizer_step(self):
        cohort = fixture()

        def loader(options, cache_dir):
            encoder = ToyEncoder(options)
            backward_calls = 0

            def corrupt_second_gradient(gradient):
                nonlocal backward_calls
                backward_calls += 1
                return gradient if backward_calls == 1 else gradient * float("nan")

            encoder.backbone.weight.register_hook(corrupt_second_gradient)
            return encoder

        steps = []
        original_step = torch.optim.AdamW.step

        def record_step(optimizer, *args, **kwargs):
            steps.append(1)
            return original_step(optimizer, *args, **kwargs)

        with mock.patch.object(torch.optim.AdamW, "step", new=record_step), \
             self.assertRaisesRegex(RuntimeError, "non-finite|Nonfinite"):
            training.fit_classifier(
                cohort.X, cohort.y, EncoderOptions("tspulse"),
                training.TrainOptions(batch_size=8, accumulation_steps=1, dropout=0.),
                71, None, epochs=1, loader=loader,
            )
        self.assertEqual(len(steps), 1)

    def test_prediction_transfers_all_scores_to_cpu_once(self):
        cohort = fixture()
        options = training.TrainOptions(dropout=0.)
        model = training.new_classifier(EncoderOptions("tspulse"), options, cohort.X, None, toy_loader)
        calls = []
        original_cpu = torch.Tensor.cpu

        def record_cpu(tensor, *args, **kwargs):
            calls.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "cpu", new=record_cpu):
            scores = training.predict_scores(model, cohort.X, batch_size=2)
        self.assertEqual(calls, [(len(cohort.X),)])
        self.assertEqual(scores.shape, cohort.y.shape)
        self.assertTrue(np.isfinite(scores).all())

    def test_tensor_conversion_reuses_owned_batches_and_copies_unsafe_views(self):
        batch = fixture().X[:2].copy()
        result = training._tensor(batch, "cpu")
        self.assertEqual(result.data_ptr(), torch.from_numpy(batch).data_ptr())
        read_only = batch.copy()
        read_only.flags.writeable = False
        independent = training._tensor(read_only, "cpu")
        original = read_only[0, 0, 0]
        independent[0, 0, 0] = -999.
        self.assertEqual(read_only[0, 0, 0], original)
        reversed_batch = training._tensor(batch[..., ::-1], "cpu")
        np.testing.assert_array_equal(reversed_batch.numpy(), batch[..., ::-1])


if __name__ == "__main__":
    unittest.main()
