"""Synthetic signals ONLY. Real checkpoint tests never load the PADS cohort."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm as real_tqdm

from sota_pd_et.data import PROJECT_ROOT, sha256_file
from sota_pd_et import mantis_finetune as fine
from sota_pd_et import predict_mantis_finetune as prediction


class TinyEncoder(nn.Module):
    """Fast test double with the pinned backbone's structural interface."""

    def __init__(self):
        super().__init__()
        self.return_transf_layer, self.output_token = 2, "combined"
        self.tokgen_unit = nn.Linear(4, 8)
        self.transf_unit = nn.Module()
        self.transf_unit.cls_token = nn.Parameter(torch.zeros(8))
        self.transf_unit.transformer = nn.Module()
        self.transf_unit.transformer.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Dropout(0.1)) for _ in range(6)])
        self.prj = nn.Linear(8, 8)

    def remove_transf_layers(self):
        self.transf_unit.transformer.layers = self.transf_unit.transformer.layers[:3]

    def forward(self, X):
        features = torch.stack((X.mean((1, 2)), X.square().mean((1, 2)), X[:, 0, 0], X[:, 0, -1]), dim=1)
        features = self.tokgen_unit(features) + self.transf_unit.cls_token
        for layer in self.transf_unit.transformer.layers[:3]:
            features = layer(features)
        return features.repeat(1, 64)


def tiny_model(config, checkpoint_dir, Z0, device="cpu"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)  # Every fit starts from identical test-backbone weights.
        encoder = TinyEncoder()
    mean, scale = fine.feature_statistics(Z0)
    return fine.FinetunedMantis(encoder, {"synthetic": True}, config, mean, scale).move_to(device)


class FinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **kwargs):
        return fine.FinetuneConfig(channels=2, samples=32, channel_batch_size=3, head_dropout=0, **kwargs)

    def samples(self, n=6, channels=2):
        rng = np.random.default_rng(11)
        return rng.normal(size=(n, channels, 32)).astype(np.float32), np.tile([0, 1], n // 2), rng.normal(size=(n, channels * 512)).astype(np.float32)

    def test_input_protocols_and_invalid_values(self):
        X = np.broadcast_to(np.arange(132)[None, :, None], (2, 132, 32)).astype(np.float32)
        self.assertEqual(fine.select_inputs(X, "all").shape, (2, 132, 32))
        selected = fine.select_inputs(X, "three_tasks")
        np.testing.assert_array_equal(selected[0, :, 0], np.arange(60))
        np.testing.assert_array_equal(fine.select_inputs(selected, "three_tasks"), selected)
        for value, name in ((selected, "all"), (X[:, :12], "three_tasks"), (X, "unknown")):
            with self.assertRaises(ValueError):
                fine.select_inputs(value, name)
        X[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            fine.select_inputs(X, "all")

    def test_statistics_are_training_only_and_constant_features_are_safe(self):
        Z = np.array([[1, 5, 3], [3, 5, 7], [1000, 5, -1000]], dtype=np.float32)
        mean, scale = fine.feature_statistics(Z[:2])
        np.testing.assert_allclose(mean, [2, 5, 5])
        np.testing.assert_allclose(scale, [1, 1, 2])
        self.assertTrue(np.isfinite((Z - mean) / scale).all())
        with self.assertRaises(ValueError):
            fine.feature_statistics(Z[:1])

    def test_trainability_and_dropout_policy_for_every_mode(self):
        _, _, Z = self.samples()
        for mode in fine.MODES:
            net = tiny_model(self.config(mode=mode), None, Z)
            net.train()
            self.assertEqual(len(net.encoder.transf_unit.transformer.layers), 3)
            self.assertTrue(all(not p.requires_grad for p in net.encoder.prj.parameters()))
            self.assertEqual(net.encoder.transf_unit.transformer.layers[0].training, mode == "full")
            self.assertEqual(net.encoder.transf_unit.transformer.layers[2].training, mode != "head-only")
            self.assertEqual(any(p.requires_grad for p in net.encoder.tokgen_unit.parameters()), mode == "full")
            self.assertEqual(any(p.requires_grad for p in net.encoder.transf_unit.transformer.layers[2].parameters()), mode != "head-only")
            net.set_warmup(True)
            self.assertFalse(net.encoder.transf_unit.transformer.layers[2].training)
            net.eval()
            self.assertFalse(net.head.training)

    def test_channel_chunking_preserves_eval_embeddings_and_participant_order(self):
        X, _, Z = self.samples()
        one = tiny_model(replace(self.config(), channel_batch_size=1), None, Z).eval()
        many = tiny_model(replace(self.config(), channel_batch_size=100), None, Z).eval()
        many.load_state_dict(one.state_dict())
        with torch.no_grad():
            a, b = one.embeddings(torch.from_numpy(X)), many.embeddings(torch.from_numpy(X))
            expected = one.encoder(nn.functional.interpolate(torch.from_numpy(X).reshape(-1, 1, 32), size=512,
                                                             mode="linear", align_corners=False)).reshape(6, 1024)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a, expected, atol=1e-6, rtol=1e-6)

    def test_activation_checkpointing_preserves_gradients(self):
        X, _, Z = self.samples()
        config = self.config(mode="full")
        a = tiny_model(config, None, Z)
        b = tiny_model(replace(config, activation_checkpointing=False), None, Z)
        b.load_state_dict(a.state_dict())
        # Disable stochastic dropout in the test encoder to compare exact paths.
        for net in (a, b):
            for layer in net.encoder.transf_unit.transformer.layers:
                layer[2].p = 0
            net(torch.from_numpy(X)).square().mean().backward()
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)

    def test_accumulation_matches_large_batch_including_short_last_group(self):
        X, y, Z = self.samples(n=6)
        config = self.config(mode="head-only")
        training = fine.TrainingConfig(encoder_lr=0, head_lr=0.001, batch_size=1, accumulation_steps=4, warmup_epochs=0)
        with mock.patch.object(fine, "create_model", side_effect=tiny_model), contextlib.redirect_stdout(io.StringIO()):
            a, _, _ = fine.fit_finetune(X, y, Z, config, training, None, epochs=1, seed=7)
            b, _, _ = fine.fit_finetune(X, y, Z, config, replace(training, batch_size=4, accumulation_steps=1), None, epochs=1, seed=7)
        for p, q in zip(a.head.parameters(), b.head.parameters()):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)

    def test_warmup_freezes_encoder_then_adaptation_changes_last_block(self):
        X, y, Z = self.samples()
        config = self.config()
        training = fine.TrainingConfig(batch_size=2, accumulation_steps=2, warmup_epochs=1)
        initial = tiny_model(config, None, Z)
        snapshots = []
        original_predict = fine.predict_scores

        def capture(model, values, **kwargs):
            snapshots.append({k: v.detach().clone() for k, v in model.encoder.named_parameters()})
            return original_predict(model, values, **kwargs)

        with mock.patch.object(fine, "create_model", side_effect=tiny_model), \
                mock.patch.object(fine, "predict_scores", side_effect=capture), contextlib.redirect_stdout(io.StringIO()):
            fitted, predictions, history = fine.fit_finetune(X, y, Z, config, training, None,
                epochs=2, seed=8, validation=X[:2], checkpoint_epochs=[1, 2])
        self.assertEqual(set(predictions), {1, 2})
        self.assertEqual([h["head_warmup"] for h in history], [True, False])
        for name, p in initial.encoder.named_parameters():
            torch.testing.assert_close(snapshots[0][name], p)
        self.assertTrue(any(not torch.equal(p, snapshots[1][name]) for name, p in initial.encoder.named_parameters()
                            if "layers.2." in name))
        for name, p in initial.encoder.named_parameters():
            if "layers.2." not in name:
                torch.testing.assert_close(p, snapshots[1][name])

    def test_safe_bundle_predictor_roundtrip_does_not_need_pretrained_files(self):
        X, _, Z = self.samples(channels=60)
        config = fine.FinetuneConfig(channels=60, samples=32, head_dropout=0, channel_batch_size=64)
        net = tiny_model(config, None, Z).eval()
        expected = fine.predict_scores(net, X)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fine.save_finetuned(root / "model.pt", net, input_set="three_tasks", metadata={"synthetic": True})
            with mock.patch.object(fine, "build_uninitialized_encoder", return_value=TinyEncoder()):
                restored, payload = fine.load_finetuned(root / "model.pt")
            np.testing.assert_allclose(fine.predict_scores(restored, X), expected, atol=1e-6)
            np.save(root / "signals.npy", X)
            with mock.patch.object(fine, "build_uninitialized_encoder", side_effect=lambda _: TinyEncoder()):
                output = prediction.main(["--model", str(root / "model.pt"), "--inputs", str(root / "signals.npy"),
                                          "--output", str(root / "predictions.csv"), "--device", "cpu", "--no-progress"])
            import csv
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            np.testing.assert_allclose([float(r["decision_score_et"]) for r in rows], expected, atol=1e-6)
            payload["state_dict"]["feature_scale"][0] = float("nan")
            torch.save(payload, root / "invalid.pt")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                fine.load_finetuned(root / "invalid.pt")

    def test_invalid_training_and_model_configurations_fail_early(self):
        for kwargs in ({"head_lr": 0}, {"encoder_lr": -1}, {"weight_decay": float("nan")},
                       {"batch_size": 0}, {"accumulation_steps": 0}, {"warmup_epochs": -1}):
            with self.assertRaises(ValueError):
                fine.TrainingConfig(**kwargs)
        for kwargs in ({"mode": "unknown"}, {"channels": 0}, {"head_dropout": 1}, {"channel_batch_size": 0}):
            with self.assertRaises(ValueError):
                fine.FinetuneConfig(**kwargs)

    def test_progress_counts_epochs_and_participants_without_changing_training(self):
        X, y, Z = self.samples()
        config = self.config(mode="full")
        training = fine.TrainingConfig(batch_size=2, accumulation_steps=2, warmup_epochs=0)
        bars, rendered = [], io.StringIO()

        def record_bar(*values, **kwargs):
            bar = real_tqdm(*values, **kwargs, file=rendered, mininterval=0)
            bars.append(bar)
            return bar

        with mock.patch.object(fine, "create_model", side_effect=tiny_model), \
                mock.patch.object(fine, "tqdm", side_effect=record_bar), contextlib.redirect_stdout(io.StringIO()):
            visible, visible_scores, _ = fine.fit_finetune(
                X, y, Z, config, training, None, epochs=2, seed=37,
                validation=X[:2], checkpoint_epochs=[1, 2], progress=True, progress_desc="Synthetic fit")
        quiet_output = io.StringIO()
        with mock.patch.object(fine, "create_model", side_effect=tiny_model), \
                contextlib.redirect_stderr(quiet_output), contextlib.redirect_stdout(io.StringIO()):
            quiet, quiet_scores, _ = fine.fit_finetune(
                X, y, Z, config, training, None, epochs=2, seed=37,
                validation=X[:2], checkpoint_epochs=[1, 2], progress=False)
        self.assertEqual(quiet_output.getvalue(), "")
        for a, b in zip(visible.parameters(), quiet.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for epoch in visible_scores:
            np.testing.assert_array_equal(visible_scores[epoch], quiet_scores[epoch])
        self.assertEqual([(b.n, b.total) for b in bars if b.desc == "Synthetic fit"], [(2, 2)])
        self.assertEqual([(b.n, b.total) for b in bars if b.desc.startswith("Epoch")], [(6, 6), (6, 6)])
        self.assertEqual([(b.n, b.total) for b in bars if b.desc == "Validation"], [(2, 2), (2, 2)])
        self.assertTrue(all(b.disable for b in bars))
        self.assertIn("BCE", rendered.getvalue())

    def test_progress_bars_close_on_failed_training_without_reaching_completion(self):
        X, y, Z = self.samples()
        bars, stream = [], io.StringIO()

        def record_bar(*values, **kwargs):
            bar = real_tqdm(*values, **kwargs, file=stream)
            bars.append(bar)
            return bar

        with mock.patch.object(fine, "create_model", side_effect=tiny_model), \
                mock.patch.object(fine, "tqdm", side_effect=record_bar), \
                mock.patch.object(fine.FinetunedMantis, "forward", side_effect=RuntimeError("synthetic failure")), \
                self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            fine.fit_finetune(X, y, Z, self.config(mode="head-only"),
                fine.TrainingConfig(encoder_lr=0, warmup_epochs=0), None, epochs=1, seed=38, progress=True)
        self.assertEqual(len(bars), 2)
        self.assertTrue(all(b.disable and b.n < b.total for b in bars))


@unittest.skipUnless((PROJECT_ROOT / "models/mantis_v2/model.safetensors").is_file(), "Local pinned checkpoint required; no download in tests.")
class ActualMantisGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_last_block_updates_through_checkpointing_after_validation(self):
        checkpoint = PROJECT_ROOT / "models/mantis_v2"
        digest = sha256_file(checkpoint / "model.safetensors")
        X = np.random.default_rng(18).normal(size=(2, 1, 32)).astype(np.float32)
        original_signals = X.copy()
        y = np.array([0, 1])
        Z = np.random.default_rng(19).normal(size=(2, 512)).astype(np.float32)
        config = fine.FinetuneConfig(channels=1, samples=32, head_dropout=0, channel_batch_size=1)
        initial = fine.create_model(config, checkpoint, Z)
        before = {k: p.detach().clone() for k, p in initial.encoder.named_parameters()}
        training = fine.TrainingConfig(batch_size=1, accumulation_steps=2, warmup_epochs=0)
        with contextlib.redirect_stdout(io.StringIO()):
            model, predictions, _ = fine.fit_finetune(X, y, Z, config, training, checkpoint,
                epochs=2, seed=8, validation=X[:1], checkpoint_epochs=[1, 2])
        self.assertTrue(all(np.isfinite(x).all() for x in predictions.values()))
        changed = [name for name, p in model.encoder.named_parameters() if not torch.equal(before[name], p)]
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("transf_unit.transformer.layers.2.") for name in changed), changed)
        self.assertEqual(model.parameter_counts()["encoder_trainable"], 658176)
        self.assertEqual(sha256_file(checkpoint / "model.safetensors"), digest)
        np.testing.assert_array_equal(X, original_signals)

    def test_real_full_mode_updates_tokenizer_and_has_no_dead_trainable_projector(self):
        checkpoint = PROJECT_ROOT / "models/mantis_v2"
        X = np.random.default_rng(20).normal(size=(2, 1, 32)).astype(np.float32)
        Z = np.random.default_rng(21).normal(size=(2, 512)).astype(np.float32)
        config = fine.FinetuneConfig(channels=1, samples=32, mode="full", channel_batch_size=2)
        initial = fine.create_model(config, checkpoint, Z)
        before = {k: p.detach().clone() for k, p in initial.encoder.tokgen_unit.named_parameters()}
        training = fine.TrainingConfig(batch_size=2, accumulation_steps=1, warmup_epochs=0)
        with contextlib.redirect_stdout(io.StringIO()):
            net, _, _ = fine.fit_finetune(X, np.array([0, 1]), Z, config, training, checkpoint, epochs=1, seed=9)
        self.assertTrue(any(not torch.equal(before[k], p) for k, p in net.encoder.tokgen_unit.named_parameters()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in net.encoder.prj.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in net.encoder.parameters() if p.requires_grad))

    def test_real_checkpoint_bundle_roundtrip_reconstructs_pruned_backbone(self):
        config = fine.FinetuneConfig(channels=60, samples=32, channel_batch_size=60, head_dropout=0)
        Z = np.random.default_rng(22).normal(size=(2, 60 * 512)).astype(np.float32)
        X = np.random.default_rng(23).normal(size=(1, 60, 32)).astype(np.float32)
        model = fine.create_model(config, PROJECT_ROOT / "models/mantis_v2", Z)
        expected = fine.predict_scores(model, X)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "real-architecture.pt"
            fine.save_finetuned(path, model, input_set="three_tasks", metadata={"synthetic_signals_only": True})
            restored, _ = fine.load_finetuned(path)
            actual = fine.predict_scores(restored, X)
        np.testing.assert_array_equal(actual, expected)

    def test_constant_synthetic_channels_have_finite_training_gradients(self):
        config = fine.FinetuneConfig(channels=1, samples=32, channel_batch_size=2)
        X = np.zeros((2, 1, 32), dtype=np.float32)
        Z = np.ones((2, 512), dtype=np.float32)
        training = fine.TrainingConfig(batch_size=2, accumulation_steps=1, warmup_epochs=0)
        with contextlib.redirect_stdout(io.StringIO()):
            model, _, _ = fine.fit_finetune(X, np.array([0, 1]), Z, config, training,
                PROJECT_ROOT / "models/mantis_v2", epochs=1, seed=25)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.requires_grad))

    def exercise_accelerator(self, device):
        config = fine.FinetuneConfig(channels=1, samples=32, channel_batch_size=2)
        X = np.random.default_rng(26).normal(size=(2, 1, 32)).astype(np.float32)
        Z = np.random.default_rng(27).normal(size=(2, 512)).astype(np.float32)
        training = fine.TrainingConfig(batch_size=2, accumulation_steps=1, warmup_epochs=0)
        with contextlib.redirect_stdout(io.StringIO()):
            model, predictions, _ = fine.fit_finetune(X, np.array([0, 1]), Z, config, training,
                PROJECT_ROOT / "models/mantis_v2", epochs=2, seed=28, device=device,
                validation=X[:1], checkpoint_epochs=[1, 2])
        self.assertEqual(next(model.parameters()).device.type, device)
        self.assertTrue(all(np.isfinite(v).all() for v in predictions.values()))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware is unavailable.")
    def test_cuda_synthetic_gradient_path(self):
        self.exercise_accelerator("cuda")

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS is unavailable in this test environment.")
    def test_mps_synthetic_gradient_path(self):
        self.exercise_accelerator("mps")
