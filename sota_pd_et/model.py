"""MultiROCKET-HYDRA features and a class-balanced ridge classifier.

Uses the aeon 1.6.0 implementation, including its HYDRA sparse scaler.
Unlike aeon's ready-made classifier, regularization is chosen by explicitly
refitting the ENTIRE feature map inside each inner training fold. Real-valued
ridge margins are exposed for AUROC; no hard labels are called probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelConfig:
    mr_kernels: int = 6250
    hydra_groups: int = 64
    hydra_kernels: int = 8
    batch_size: int = 32
    n_jobs: int = 4
    hydra_device: str = "cpu"


class FeatureMap:
    """Fit HYDRA/MultiROCKET and their scalers on a training fold only."""

    def __init__(self, config: ModelConfig, seed: int):
        self.config = config
        self.seed = seed

    def set_hydra_device(self, requested: str):
        """Select CPU/CUDA without refitting kernels, scalers, or the classifier."""
        from .hydra_gpu import resolve_device

        resolved = resolve_device(requested)
        self._hydra_device_request_ = requested
        self.hydra_device_ = resolved
        self.__dict__.pop("_hydra_device_transform_", None)
        return self

    def __getstate__(self):
        # aeon's fitted kernels and the sparse scaler stay on CPU. The CUDA
        # adapter contains disposable copies and must never enter a model file.
        state = self.__dict__.copy()
        state.pop("_hydra_device_transform_", None)
        if state.get("hydra_device_") == "cuda":
            # A saved GPU-trained model can perform inference on a CPU runtime.
            state["_hydra_device_request_"] = "auto"
        return state

    @staticmethod
    def _validate(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or len(X) == 0 or X.shape[1] < 1 or X.shape[2] < 10:
            raise ValueError("Expected nonempty (participants, channels, timepoints>=10).")
        if not np.isfinite(X).all():
            raise ValueError("The model cannot accept NaN or infinite samples.")
        return np.ascontiguousarray(X)

    def _hydra_features(self, X):
        import torch
        from .hydra_gpu import HydraDeviceTransform, resolve_device

        requested = getattr(self, "_hydra_device_request_",
                            getattr(self.config, "hydra_device", "cpu"))
        self.hydra_device_ = resolve_device(requested)
        if self.hydra_device_ == "cuda":
            adapter = getattr(self, "_hydra_device_transform_", None)
            if adapter is None:
                adapter = HydraDeviceTransform(self.hydra_, "cuda")
                self._hydra_device_transform_ = adapter
            transform = adapter.transform
        else:
            # Preserve the original CPU implementation and numerical behavior.
            transform = self.hydra_.transform

        with torch.inference_mode():
            return torch.cat([
                transform(X[start:start + self.config.batch_size])
                for start in range(0, len(X), self.config.batch_size)
            ], dim=0)

    def _mr_features(self, X):
        return np.concatenate([
            self.multirocket_.transform(X[start:start + self.config.batch_size])
            for start in range(0, len(X), self.config.batch_size)
        ], axis=0)

    @staticmethod
    def _join(hydra, multirocket):
        # float64 stabilizes the dual ridge solve with many standardized features.
        Z = np.concatenate((hydra.detach().cpu().numpy(), multirocket), axis=1)
        Z = np.ascontiguousarray(Z, dtype=np.float64)
        if not np.isfinite(Z).all():
            raise ValueError("Nonfinite transformed features.")
        return Z

    def fit_transform(self, X):
        X = self._validate(X)
        if len(X) < 2:
            raise ValueError("At least two training participants are needed for HYDRA scaling.")
        if version("aeon") != "1.6.0":
            raise RuntimeError("Install aeon==1.6.0: this code pins its private HYDRA scaler API.")
        from aeon.classification.convolution_based._hydra import _SparseScaler
        from aeon.transformations.collection.convolution_based import HydraTransformer, MultiRocket

        self.input_shape_ = X.shape[1:]
        self.hydra_ = HydraTransformer(
            n_kernels=self.config.hydra_kernels,
            n_groups=self.config.hydra_groups,
            n_jobs=self.config.n_jobs,
            random_state=self.seed,
            output_type="tensor",
        )
        self.hydra_.fit(X)
        self.set_hydra_device(getattr(self.config, "hydra_device", "cpu"))
        counts = self._hydra_features(X)
        self.hydra_scaler_ = _SparseScaler()
        hydra = self.hydra_scaler_.fit_transform(counts)
        self.multirocket_ = MultiRocket(
            n_kernels=self.config.mr_kernels,
            normalise=False,
            n_jobs=self.config.n_jobs,
            random_state=self.seed,
        )
        self.multirocket_.fit(X)
        self.mr_scaler_ = StandardScaler()
        multirocket = self.mr_scaler_.fit_transform(self._mr_features(X))
        return self._join(hydra, multirocket)

    def transform(self, X):
        X = self._validate(X)
        if X.shape[1:] != self.input_shape_:
            raise ValueError(f"Expected input shape (*, {self.input_shape_}), got {X.shape}.")
        hydra = self.hydra_scaler_.transform(self._hydra_features(X))
        multirocket = self.mr_scaler_.transform(self._mr_features(X))
        return self._join(hydra, multirocket)


def fit_ridge(Z: np.ndarray, y: np.ndarray, alpha: float) -> RidgeClassifier:
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Each training fold must include PD=0 and ET=1.")
    classifier = RidgeClassifier(alpha=float(alpha), class_weight="balanced", solver="auto")
    classifier.fit(Z, y)
    if not np.array_equal(classifier.classes_, [0, 1]):
        raise RuntimeError("Unexpected class order; a positive margin must mean ET.")
    return classifier


@dataclass
class FittedModel:
    features: FeatureMap
    classifier: RidgeClassifier

    def decision_function(self, X):
        return np.asarray(self.classifier.decision_function(self.features.transform(X))).reshape(-1)

    def predict(self, X):
        # Matches sklearn's binary ridge rule, including a score exactly equal to zero.
        return (self.decision_function(X) > 0.0).astype(np.int64)
