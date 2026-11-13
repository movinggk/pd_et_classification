"""Synthetic-only checks: layout, temporal ordering, interaction and inference."""

from dataclasses import replace
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from sota_pd_et import predict_task_interaction as prediction
from sota_pd_et.task_interaction import (
    CHANNEL_INDICES, NetworkConfig, TaskStandardizer, TemporalCrossTaskNet,
    balanced_class_weights, fit_trajectory, load_network, predict_logits,
    resolve_torch_device, save_network, select_task_segments, variant_config,
)


class TaskInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **kwargs):
        return NetworkConfig(samples_per_segment=32, hidden_dim=8, heads=2, dropout=0.0, **kwargs)

    def test_task_selection_preserves_segment_wrist_sensor_axis_order(self):
        X = np.broadcast_to(np.arange(132)[None, :, None], (2, 132, 32)).astype(np.float32)
        selected = select_task_segments(X)
        np.testing.assert_array_equal(selected.reshape(2, 60, 32)[0, :, 0], CHANNEL_INDICES)
        np.testing.assert_array_equal(select_task_segments(selected.reshape(2, 60, 32)), selected)
        self.assertEqual(selected.shape, (2, 5, 12, 32))
        self.assertEqual(selected[0, 4, 0, 0], 48)
        with self.assertRaises(ValueError):
            select_task_segments(np.zeros((2, 59, 32)))
        X[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            select_task_segments(X)

    def test_shared_training_standardizer_preserves_between_task_amplitude(self):
        X = np.random.default_rng(19).normal(size=(8, 5, 12, 32)).astype(np.float32)
        scaler = TaskStandardizer().fit(X[:6])
        np.testing.assert_allclose(scaler.mean_.ravel(), X[:6].mean(axis=(0, 1, 3), dtype=np.float64))
        frozen_mean, frozen_scale = scaler.mean_.copy(), scaler.scale_.copy()
        doubled = X[6:] * 2
        first, second = scaler.transform(X[6:]), scaler.transform(doubled)
        expected_delta = (doubled.astype(float) - X[6:]) / scaler.scale_
        np.testing.assert_allclose(second - first, expected_delta, atol=1e-6)
        np.testing.assert_array_equal(scaler.mean_, frozen_mean)
        np.testing.assert_array_equal(scaler.scale_, frozen_scale)
        constant = TaskStandardizer().fit(np.zeros_like(X))
        self.assertTrue(np.isfinite(constant.transform(X)).all())

    def test_cnn_windows_do_not_mix_segment_boundaries(self):
        network = TemporalCrossTaskNet(self.config())
        X = torch.arange(5.0)[None, :, None, None].expand(1, 5, 12, 32).clone()
        seen = []
        handle = network.window_cnn.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
        network(X)
        handle.remove()
        windows = seen[0]
        self.assertEqual(tuple(windows.shape), (20, 12, 8))
        for segment in range(5):
            torch.testing.assert_close(windows[segment * 4:(segment + 1) * 4],
                                       torch.full((4, 12, 8), float(segment)))

    def test_unordered_control_is_invariant_to_window_permutations(self):
        torch.manual_seed(29)
        X = torch.randn(3, 5, 12, 32)
        # Reorder entire local windows, preserving the samples inside each one.
        shuffled = X.reshape(3, 5, 12, 4, 8)[:, :, :, [2, 0, 3, 1], :].reshape_as(X)
        unordered = TemporalCrossTaskNet(self.config(temporal=False)).eval()
        with torch.inference_mode():
            torch.testing.assert_close(unordered(X), unordered(shuffled), atol=1e-6, rtol=1e-5)
        ordered = TemporalCrossTaskNet(self.config(temporal=True)).eval()
        with torch.inference_mode():
            difference = (ordered.encode_tasks(X) - ordered.encode_tasks(shuffled)).abs().max()
        self.assertGreater(float(difference), 1e-5)
        self.assertTrue(torch.all(ordered.positions_0[4:] != ordered.positions_0[:4]).any())

    def test_additive_control_has_zero_mixed_effect_and_bilinear_has_product(self):
        base = self.config(fusion="additive")
        additive, interaction = TemporalCrossTaskNet(base), TemporalCrossTaskNet(replace(base, fusion="bilinear"))
        with torch.no_grad():
            interaction.pair_left.fill_(0.1)
            interaction.pair_right.fill_(0.1)
        zeros = torch.zeros(1, 3, additive.task_dim)
        left, right = zeros.clone(), zeros.clone()
        left[:, 0], right[:, 1] = 1, 1
        for net, expected_zero in ((additive, True), (interaction, False)):
            mixed = net.score_tasks(left + right) - net.score_tasks(left) - net.score_tasks(right) + net.score_tasks(zeros)
            if expected_zero:
                torch.testing.assert_close(mixed, torch.zeros_like(mixed), atol=1e-6, rtol=0)
            else:
                self.assertGreater(float(mixed.detach()), 0.1)

    def test_all_ablation_variants_produce_finite_logits_and_gradients(self):
        X = torch.randn(3, 5, 12, 32)
        for name in ("full", "no_temporal", "no_interaction", "neither", "concat_mlp"):
            config = variant_config(name, samples_per_segment=32, hidden_dim=8, heads=2, dropout=0.0)
            net = TemporalCrossTaskNet(config)
            score = net(X)
            self.assertEqual(tuple(score.shape), (3,))
            self.assertTrue(torch.isfinite(score).all())
            score.square().sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in net.parameters()))

    def test_default_recording_length_is_supported_with_finite_scores(self):
        # Correct real INPUT SHAPE, but the values are entirely synthetic.
        net = TemporalCrossTaskNet(NetworkConfig()).eval()
        with torch.inference_mode():
            scores = net(torch.randn(2, 5, 12, 976))
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertLess(sum(p.numel() for p in net.parameters()), 100000)

    def test_device_policy_prefers_cuda_then_mps_then_cpu_and_rejects_unavailable(self):
        for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
            with mock.patch("torch.cuda.is_available", return_value=cuda), \
                    mock.patch("torch.backends.mps.is_available", return_value=mps):
                self.assertEqual(resolve_torch_device("auto"), expected)
                if not cuda:
                    with self.assertRaisesRegex(ValueError, "CUDA"):
                        resolve_torch_device("cuda")
                if not mps:
                    with self.assertRaisesRegex(ValueError, "MPS"):
                        resolve_torch_device("mps")

    def test_training_class_weights_and_safe_saved_prediction_roundtrip(self):
        rng = np.random.default_rng(41)
        X = rng.normal(size=(8, 5, 12, 32)).astype(np.float32)
        y = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        np.testing.assert_allclose(balanced_class_weights(y), [2 / 3, 2])
        with self.assertRaises(ValueError):
            balanced_class_weights(np.zeros(8))
        scaler = TaskStandardizer().fit(X)
        standardized = scaler.transform(X)
        net, predictions, history = fit_trajectory(
            standardized, y, self.config(), epochs=2, learning_rate=0.001, weight_decay=0.01,
            batch_size=4, seed=17, validation=standardized[:2], checkpoint_epochs=[1, 2],
        )
        self.assertEqual(set(predictions), {1, 2})
        self.assertEqual(len(history), 2)
        expected = predict_logits(net, standardized)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_network(root / "model.pt", net, scaler, threshold=0.15, metadata={"synthetic": True})
            restored, restored_scaler, payload = load_network(root / "model.pt")
            actual = predict_logits(restored, restored_scaler.transform(X))
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(payload["threshold"], 0.15)
            np.save(root / "inputs.npy", X)
            output = prediction.main(["--model", str(root / "model.pt"), "--inputs", str(root / "inputs.npy"),
                                      "--output", str(root / "predictions.csv"), "--device", "cpu"])
            import csv
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            np.testing.assert_allclose([float(r["raw_score_et"]) for r in rows], expected)
            self.assertEqual([int(r["predicted_et"]) for r in rows], (expected > 0.15).astype(int).tolist())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                prediction.main(["--model", str(root / "model.pt"), "--inputs", str(root / "inputs.npy"),
                                 "--output", str(output)])

    def test_invalid_network_configuration_is_rejected(self):
        for kwargs in ({"hidden_dim": 7}, {"windows_per_segment": 3}, {"dropout": 1.0},
                       {"interaction_rank": 0}, {"fusion": "unknown"}):
            with self.assertRaises(ValueError):
                NetworkConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
