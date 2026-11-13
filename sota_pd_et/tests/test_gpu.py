"""Optional CUDA HYDRA tests; synthetic fixtures require no downloaded data."""

from __future__ import annotations

from importlib.util import find_spec
import io
import pickle
import unittest
from unittest import mock

import numpy as np


HAS_TORCH = find_spec("torch") is not None
HAS_MODEL_DEPS = HAS_TORCH and all(
    find_spec(name) is not None for name in ("aeon", "sklearn", "joblib")
)
if HAS_TORCH:
    import torch

    HAS_CUDA = torch.cuda.is_available()
else:
    HAS_CUDA = False


@unittest.skipUnless(HAS_TORCH, "Install torch for optional device-selection tests.")
class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_is_accepted_without_cuda_hardware(self):
        from sota_pd_et.hydra_gpu import resolve_device

        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("cpu"), "cpu")

    def test_auto_selects_cuda_when_available(self):
        from sota_pd_et.hydra_gpu import resolve_device

        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("auto"), "cuda")
            self.assertEqual(resolve_device("cuda"), "cuda")

    def test_auto_falls_back_to_cpu_when_cuda_is_unavailable(self):
        from sota_pd_et.hydra_gpu import resolve_device

        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), "cpu")

    def test_explicit_cuda_fails_clearly_when_unavailable(self):
        from sota_pd_et.hydra_gpu import resolve_device

        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA|cuda"):
                resolve_device("cuda")

    def test_unknown_device_is_rejected(self):
        from sota_pd_et.hydra_gpu import resolve_device

        with self.assertRaises(ValueError):
            resolve_device("gpu")


@unittest.skipUnless(HAS_MODEL_DEPS, "Install full dependencies for HYDRA parity tests.")
class HydraAdapterTests(unittest.TestCase):
    @staticmethod
    def fixture(n_channels, n_timepoints, n_groups=4):
        from aeon.transformations.collection.convolution_based import HydraTransformer

        rng = np.random.default_rng(71 + n_channels + n_timepoints)
        X = rng.normal(size=(5, n_channels, n_timepoints)).astype(np.float32)
        hydra = HydraTransformer(
            n_kernels=2, n_groups=n_groups, n_jobs=1,
            random_state=17, output_type="tensor",
        ).fit(X)
        return X, hydra

    def test_cpu_adapter_matches_upstream_values_and_column_order(self):
        from sota_pd_et.hydra_gpu import HydraDeviceTransform

        for n_channels in (1, 3):
            for n_timepoints in (64, 65):
                with self.subTest(channels=n_channels, timepoints=n_timepoints):
                    X, hydra = self.fixture(n_channels, n_timepoints)
                    expected = hydra.transform(X)
                    actual = HydraDeviceTransform(hydra, "cpu").transform(X)
                    self.assertIsInstance(actual, torch.Tensor)
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertFalse(actual.requires_grad)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_one_group_without_difference_branch_matches_upstream(self):
        from sota_pd_et.hydra_gpu import HydraDeviceTransform

        for n_channels in (1, 3):
            with self.subTest(channels=n_channels):
                X, hydra = self.fixture(n_channels, 65, n_groups=1)
                actual = HydraDeviceTransform(hydra, "cpu").transform(X)
                torch.testing.assert_close(actual, hydra.transform(X), rtol=0, atol=0)

    def test_adapter_preserves_inputs_and_upstream_cpu_tensors(self):
        from sota_pd_et.hydra_gpu import HydraDeviceTransform

        X, hydra = self.fixture(3, 64)
        original = X.copy()
        weights = hydra._hydra.W.clone()
        indices = [index.clone() for index in hydra._hydra.idx]
        adapter = HydraDeviceTransform(hydra, "cpu")
        first = adapter.transform(X)
        second = adapter.transform(X)
        np.testing.assert_array_equal(X, original)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(hydra._hydra.W, weights, rtol=0, atol=0)
        self.assertEqual(hydra._hydra.W.device.type, "cpu")
        for actual, expected in zip(hydra._hydra.idx, indices, strict=True):
            self.assertEqual(actual.device.type, "cpu")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(HAS_CUDA, "A working CUDA GPU is required for numerical CUDA verification.")
    def test_cuda_adapter_returns_cpu_features_close_to_upstream(self):
        from sota_pd_et.hydra_gpu import HydraDeviceTransform

        for n_channels in (1, 3):
            for n_timepoints in (64, 65):
                with self.subTest(channels=n_channels, timepoints=n_timepoints):
                    X, hydra = self.fixture(n_channels, n_timepoints)
                    expected = hydra.transform(X)
                    actual = HydraDeviceTransform(hydra, "cuda").transform(X)
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    # HYDRA contains winner selection, so CUDA convolution rounding
                    # need not give bit-identical features. Random fixtures avoid
                    # deliberate ties while checking translation and feature order.
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
                    self.assertEqual(hydra._hydra.W.device.type, "cpu")
                    self.assertTrue(all(index.device.type == "cpu" for index in hydra._hydra.idx))


@unittest.skipUnless(HAS_MODEL_DEPS, "Install full dependencies for model portability tests.")
class ModelDevicePortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sota_pd_et.evaluate import fit_full_model
        from sota_pd_et.model import ModelConfig

        rng = np.random.default_rng(37)
        X = rng.normal(size=(12, 3, 64)).astype(np.float32)
        y = np.tile([0, 1], 6)
        config = ModelConfig(
            mr_kernels=84, hydra_groups=4, hydra_kernels=2,
            batch_size=3, n_jobs=1, hydra_device="cpu",
        )
        cls.fitted = fit_full_model(X[:8], y[:8], config, alpha=1.0, seed=17)
        cls.test_inputs = X[8:]

    def copy_model(self):
        return pickle.loads(pickle.dumps(self.fitted))

    def test_cpu_default_and_explicit_cpu_preserve_predictions(self):
        from sota_pd_et.model import ModelConfig

        self.assertEqual(ModelConfig().hydra_device, "cpu")
        fitted = self.copy_model()
        expected = fitted.decision_function(self.test_inputs)
        fitted.features.set_hydra_device("cpu")
        actual = fitted.decision_function(self.test_inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_auto_cpu_fallback_preserves_predictions_after_serialization(self):
        fitted = self.copy_model()
        expected = fitted.decision_function(self.test_inputs)
        with mock.patch("torch.cuda.is_available", return_value=False):
            fitted.features.set_hydra_device("auto")
            loaded = pickle.loads(pickle.dumps(fitted))
            actual = loaded.decision_function(self.test_inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(loaded.features.hydra_device_, "cpu")

    def test_legacy_saved_cpu_model_without_device_attributes_still_predicts(self):
        fitted = self.copy_model()
        expected = fitted.decision_function(self.test_inputs)
        fitted.features.__dict__.pop("_hydra_device_request_", None)
        fitted.features.__dict__.pop("hydra_device_", None)
        fitted.features.__dict__.pop("_hydra_device_transform_", None)
        # Pickles made before this change have no new dataclass instance field.
        fitted.features.config.__dict__.pop("hydra_device", None)
        loaded = pickle.loads(pickle.dumps(fitted))
        with mock.patch("torch.cuda.is_available", return_value=False):
            actual = loaded.decision_function(self.test_inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_failed_device_override_keeps_previous_working_selection(self):
        fitted = self.copy_model()
        expected = fitted.decision_function(self.test_inputs)
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                fitted.features.set_hydra_device("cuda")
            actual = fitted.decision_function(self.test_inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_gpu_backend_selection_and_cache_with_a_cpu_execution_double(self):
        from sota_pd_et.hydra_gpu import HydraDeviceTransform

        fitted = self.copy_model()
        expected = fitted.decision_function(self.test_inputs)

        def cpu_execution_double(transformer, device):
            self.assertEqual(device, "cuda")
            return HydraDeviceTransform(transformer, "cpu")

        # Exercise FeatureMap's GPU branch and cache on CPU test hosts. The
        # separate hardware-conditional tests provide actual GPU verification.
        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("sota_pd_et.hydra_gpu.HydraDeviceTransform",
                           side_effect=cpu_execution_double) as factory:
            fitted.features.set_hydra_device("auto")
            first = fitted.decision_function(self.test_inputs)
            second = fitted.decision_function(self.test_inputs)
            self.assertEqual(fitted.features.hydra_device_, "cuda")
        self.assertEqual(factory.call_count, 1)
        np.testing.assert_allclose(first, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(second, expected, rtol=1e-12, atol=1e-12)

    def test_adapter_cache_is_excluded_from_saved_models(self):
        import joblib

        fitted = self.copy_model()
        expected = fitted.decision_function(self.test_inputs)

        class UnpicklableAdapter:
            def __reduce__(self):
                raise AssertionError("The device adapter cache must not be serialized.")

        adapter = UnpicklableAdapter()
        fitted.features._hydra_device_transform_ = adapter
        stream = io.BytesIO()
        joblib.dump(fitted, stream)
        self.assertIs(fitted.features._hydra_device_transform_, adapter)
        stream.seek(0)
        loaded = joblib.load(stream)
        self.assertNotIn("_hydra_device_transform_", loaded.features.__dict__)
        actual = loaded.decision_function(self.test_inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(loaded.features.hydra_._hydra.W.device.type, "cpu")
        self.assertTrue(all(
            index.device.type == "cpu" for index in loaded.features.hydra_._hydra.idx
        ))

    def test_saved_cuda_policy_falls_back_on_a_cpu_only_host(self):
        fitted = self.copy_model()
        expected = fitted.decision_function(self.test_inputs)
        # Simulate the saved selection without constructing CUDA tensors. The
        # saved fitted kernels and scalers remain CPU resident on a GPU host too.
        fitted.features._hydra_device_request_ = "cuda"
        fitted.features.hydra_device_ = "cuda"
        stream = pickle.dumps(fitted)
        with mock.patch("torch.cuda.is_available", return_value=False):
            loaded = pickle.loads(stream)
            actual = loaded.decision_function(self.test_inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(loaded.features.hydra_device_, "cpu")

    @unittest.skipUnless(HAS_CUDA, "A working CUDA GPU is required for model integration verification.")
    def test_real_cuda_training_and_saved_model_cpu_inference(self):
        import joblib
        from sota_pd_et.evaluate import fit_full_model
        from sota_pd_et.model import ModelConfig

        rng = np.random.default_rng(43)
        X = rng.normal(size=(12, 3, 65)).astype(np.float32)
        y = np.tile([0, 1], 6)
        config = ModelConfig(
            mr_kernels=84, hydra_groups=4, hydra_kernels=2,
            batch_size=3, n_jobs=1, hydra_device="cuda",
        )
        fitted = fit_full_model(X[:8], y[:8], config, alpha=1.0, seed=17)
        features = fitted.features
        self.assertEqual(features.hydra_device_, "cuda")
        adapter = features._hydra_device_transform_
        self.assertEqual(adapter.device, "cuda")
        self.assertEqual(adapter.weights.device.type, "cuda")
        self.assertTrue(all(index.device.type == "cuda" for index in adapter.indices))
        gpu_scores = fitted.decision_function(X[8:])
        self.assertEqual(gpu_scores.shape, (4,))
        self.assertTrue(np.isfinite(gpu_scores).all())
        self.assertEqual(features.hydra_._hydra.W.device.type, "cpu")
        self.assertTrue(all(index.device.type == "cpu" for index in features.hydra_._hydra.idx))
        for name in ("epsilon", "mu", "sigma"):
            self.assertEqual(getattr(features.hydra_scaler_, name).device.type, "cpu")

        stream = io.BytesIO()
        joblib.dump(fitted, stream)
        self.assertIs(features._hydra_device_transform_, adapter)
        stream.seek(0)
        loaded = joblib.load(stream)
        self.assertNotIn("_hydra_device_transform_", loaded.features.__dict__)
        self.assertEqual(loaded.features.hydra_._hydra.W.device.type, "cpu")
        self.assertTrue(all(
            index.device.type == "cpu" for index in loaded.features.hydra_._hydra.idx
        ))
        for name in ("epsilon", "mu", "sigma"):
            self.assertEqual(getattr(loaded.features.hydra_scaler_, name).device.type, "cpu")
        with mock.patch("torch.cuda.is_available", return_value=False):
            cpu_scores = loaded.decision_function(X[8:])
        self.assertEqual(loaded.features.hydra_device_, "cpu")
        self.assertEqual(cpu_scores.shape, (4,))
        self.assertTrue(np.isfinite(cpu_scores).all())
        self.assertNotIn("_hydra_device_transform_", loaded.features.__dict__)
        # The same fitted scaler and classifier serve CPU inference; CUDA's
        # convolution and scatter order need not reproduce bit-identical scores.


if __name__ == "__main__":
    unittest.main()
