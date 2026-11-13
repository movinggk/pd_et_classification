"""Small temporal/cross-task network for the *three-condition* PADS protocol.

No dataset loading, downloads or experiment execution occurs on import. The
author-supplied 10-second segments are kept separate inside the window CNN;
nominal positions account for the crop between the two halves of long tasks.
This does not reconstruct, interpolate or repair the raw recording timestamps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math

import numpy as np
import torch
from torch import nn

from .data import RETAINED_SEGMENTS


SEGMENTS = ("Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold")
TASKS = ("rest", "cognitive_rest", "posture")
TASK_SEGMENTS = ((0, 1), (2, 3), (4,))
CHANNEL_INDICES = tuple(
    RETAINED_SEGMENTS.index(segment) * 12 + channel
    for segment in SEGMENTS for channel in range(12)
)
VARIANTS = {
    "full": (True, "bilinear"),
    "no_temporal": (False, "bilinear"),
    "no_interaction": (True, "additive"),
    "neither": (False, "additive"),
    "concat_mlp": (True, "mlp"),
}


def select_task_segments(X):
    """Return (N, 5, 12, T) from official 132-channel or selected 60-channel data.

    The returned segment/wrist/sensor/axis order is fixed. In a 60-channel
    input the caller must already have used exactly CHANNEL_INDICES order.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3 and X.shape[1] == 132:
        X = X[:, CHANNEL_INDICES, :]
    if X.ndim == 3 and X.shape[1] == 60:
        X = X.reshape(len(X), 5, 12, X.shape[-1])
    if X.ndim != 4 or X.shape[1:3] != (5, 12) or len(X) < 1 or X.shape[-1] < 8:
        raise ValueError("Expected finite (N,132,T), (N,60,T), or (N,5,12,T) input.")
    if not np.isfinite(X).all():
        raise ValueError("Task input contains NaN or infinite samples.")
    return np.ascontiguousarray(X)


class TaskStandardizer:
    """One training-only mean/std per physical channel, shared across tasks.

    A participant/task is never divided by its own standard deviation. This
    preserves between-task amplitude differences up to the same affine map.
    """

    def fit(self, X):
        X = select_task_segments(X)
        self.mean_ = np.mean(X, axis=(0, 1, 3), dtype=np.float64).reshape(1, 1, 12, 1)
        self.scale_ = np.std(X, axis=(0, 1, 3), dtype=np.float64).reshape(1, 1, 12, 1)
        self.scale_ = np.maximum(self.scale_, 1e-6)
        self.shape_ = tuple(X.shape[1:])
        return self

    def transform(self, X):
        if not hasattr(self, "mean_"):
            raise ValueError("Fit the task standardizer on training participants first.")
        X = select_task_segments(X)
        if tuple(X.shape[1:]) != self.shape_:
            raise ValueError(f"Expected task shape {self.shape_}, got {X.shape[1:]}.")
        Z = np.asarray((X - self.mean_) / self.scale_, dtype=np.float32)
        if not np.isfinite(Z).all():
            raise ValueError("Nonfinite standardized task input.")
        return np.ascontiguousarray(Z)


@dataclass(frozen=True)
class NetworkConfig:
    samples_per_segment: int = 976
    windows_per_segment: int = 4
    hidden_dim: int = 32
    heads: int = 4
    temporal_layers: int = 1
    interaction_rank: int = 4
    dropout: float = 0.2
    temporal: bool = True
    fusion: str = "bilinear"

    def __post_init__(self):
        integer_fields = (self.samples_per_segment, self.windows_per_segment,
                          self.hidden_dim, self.heads, self.temporal_layers, self.interaction_rank)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in integer_fields):
            raise ValueError("Network dimensions must be positive integers.")
        if self.samples_per_segment % self.windows_per_segment or self.samples_per_segment // self.windows_per_segment < 8:
            raise ValueError("Each segment must divide into equal windows of at least eight samples.")
        if self.hidden_dim % self.heads or self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even and divisible by heads.")
        if not 0 <= self.dropout < 1 or self.fusion not in ("bilinear", "additive", "mlp"):
            raise ValueError("Invalid dropout or fusion type.")


def variant_config(name, **kwargs):
    if name not in VARIANTS:
        raise ValueError(f"Unknown neural variant: {name}.")
    temporal, fusion = VARIANTS[name]
    return NetworkConfig(temporal=temporal, fusion=fusion, **kwargs)


def positional_encoding(positions, dimension):
    frequencies = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / dimension))
    result = torch.zeros(len(positions), dimension)
    phase = torch.as_tensor(positions, dtype=torch.float32)[:, None] * frequencies[None, :]
    result[:, 0::2], result[:, 1::2] = torch.sin(phase), torch.cos(phase)
    return result


class TemporalCrossTaskNet(nn.Module):
    """Shared local CNN, optional ordered encoder, additive or nonlinear fusion.

    With additive fusion the *logit* is a sum of three separate task functions.
    Bilinear fusion adds low-rank products of different tasks' representations.
    concat_mlp is a separate, generic nonlinear-fusion control; it is not an
    interaction-free control. Neither attention nor a product proves causality.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.window_cnn = nn.Sequential(
            nn.Conv1d(12, d, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 3, stride=2, padding=1), nn.GELU(),
        )
        self.window_projection = nn.Sequential(
            nn.Linear(2 * d + 12, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(config.dropout),
        )
        if config.temporal:
            layer = nn.TransformerEncoderLayer(
                d, config.heads, dim_feedforward=2 * d, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                layer, config.temporal_layers, enable_nested_tensor=False,
            )
            # TransformerEncoder clones layers: initialize each weight separately.
            for name, parameter in self.temporal_encoder.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        window = config.samples_per_segment // config.windows_per_segment
        centers = (np.arange(config.windows_per_segment) + 0.5) * window + 48
        for task, segments in enumerate(TASK_SEGMENTS):
            offsets = [half * (config.samples_per_segment + 48) for half in range(len(segments))]
            seconds = np.concatenate([(centers + offset) / 100.0 for offset in offsets])
            self.register_buffer(f"positions_{task}", positional_encoding(seconds, d))
        # Both moments of window embeddings and explicit log-RMS statistics.
        self.task_dim = 2 * d + 24
        self.main_effect = nn.Parameter(torch.empty(3, self.task_dim))
        nn.init.normal_(self.main_effect, std=0.02)
        self.bias = nn.Parameter(torch.zeros(()))
        self.pairs = tuple(combinations(range(3), 2))
        if config.fusion == "bilinear":
            self.pair_left = nn.Parameter(torch.empty(3, self.task_dim, config.interaction_rank))
            self.pair_right = nn.Parameter(torch.empty_like(self.pair_left))
            nn.init.normal_(self.pair_left, std=0.02)
            nn.init.normal_(self.pair_right, std=0.02)
        elif config.fusion == "mlp":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(3 * self.task_dim, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, 1),
            )

    def encode_tasks(self, X):
        if X.ndim != 4 or tuple(X.shape[1:]) != (5, 12, self.config.samples_per_segment):
            raise ValueError("Network input must have shape (batch,5,12,samples_per_segment).")
        n, _, channels, length = X.shape
        windows = self.config.windows_per_segment
        # Convolutions never cross a segment boundary or a task boundary.
        local = X.reshape(n, 5, channels, windows, length // windows).permute(0, 1, 3, 2, 4)
        flat = local.reshape(n * 5 * windows, channels, length // windows)
        amplitude = torch.log(torch.sqrt(flat.square().mean(dim=-1) + 1e-8))
        feature = self.window_cnn(flat)
        feature = self.window_projection(torch.cat((feature.mean(-1), feature.amax(-1), amplitude), dim=-1))
        feature = feature.reshape(n, 5, windows, -1)
        amplitude = amplitude.reshape(n, 5, windows, channels)
        tasks = []
        for task, segments in enumerate(TASK_SEGMENTS):
            tokens = feature[:, list(segments)].reshape(n, len(segments) * windows, -1)
            amps = amplitude[:, list(segments)].reshape(n, len(segments) * windows, channels)
            if self.config.temporal:
                tokens = self.temporal_encoder(tokens + getattr(self, f"positions_{task}")[None])
            tasks.append(torch.cat((tokens.mean(1), tokens.std(1, unbiased=False),
                                    amps.mean(1), amps.std(1, unbiased=False)), dim=-1))
        return torch.stack(tasks, dim=1)

    def score_tasks(self, tasks):
        score = (tasks * self.main_effect[None]).sum(dim=(1, 2)) + self.bias
        if self.config.fusion == "bilinear":
            for index, (left, right) in enumerate(self.pairs):
                a = tasks[:, left] @ self.pair_left[index]
                b = tasks[:, right] @ self.pair_right[index]
                score = score + (a * b).sum(-1) / math.sqrt(self.config.interaction_rank)
        elif self.config.fusion == "mlp":
            score = score + self.fusion_mlp(tasks.flatten(1)).squeeze(-1)
        return score

    def forward(self, X):
        return self.score_tasks(self.encode_tasks(X))


def resolve_torch_device(requested="auto"):
    if requested not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("Choose auto, cpu, cuda or mps.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS requested but unavailable.")
    return requested


def balanced_class_weights(y):
    y = np.asarray(y)
    if y.ndim != 1 or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Every fit needs both PD=0 and ET=1 participants.")
    return (len(y) / (2 * np.bincount(y.astype(int), minlength=2))).astype(np.float32)


def seed_torch(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def predict_logits(network, X, device="cpu", batch_size=16):
    network.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            values = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size])).to(device)
            output.append(network(values).detach().cpu().numpy())
    scores = np.concatenate(output).astype(float)
    if scores.shape != (len(X),) or not np.isfinite(scores).all():
        raise RuntimeError("Expected one finite logit per participant.")
    return scores


def fit_trajectory(X, y, config, *, epochs, learning_rate, weight_decay, batch_size,
                   seed, device="cpu", validation=None, checkpoint_epochs=()):
    """Train on X/y only; validation is forward-only, at requested checkpoints.

    Epoch selection is performed by the nested evaluator, never on outer test
    participants. No automatic early stopping or outer-test metric is used.
    """
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid training configuration.")
    if any(e < 1 or e > epochs for e in checkpoint_epochs):
        raise ValueError("Checkpoint epochs must lie inside the training trajectory.")
    seed_torch(seed)
    network = TemporalCrossTaskNet(config).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    values = torch.from_numpy(np.ascontiguousarray(X))
    targets = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    class_weights = torch.as_tensor(balanced_class_weights(y), device=device)
    generator = torch.Generator().manual_seed(int(seed))
    predictions, history = {}, []
    for epoch in range(1, epochs + 1):
        network.train()
        losses = []
        permutation = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), batch_size):
            index = permutation[start:start + batch_size]
            batch = values[index].to(device)
            target = targets[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(batch)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
                    * class_weights[target.long()]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite neural training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()) * len(index))
        history.append({"epoch": epoch, "training_weighted_bce": sum(losses) / len(y)})
        if epoch in checkpoint_epochs and validation is not None:
            predictions[epoch] = predict_logits(network, validation, device, batch_size)
    return network, predictions, history


def save_network(path, network, standardizer, *, threshold, metadata):
    """A tensor/primitive-only bundle loadable with torch weights_only=True."""
    payload = {
        "format_version": 1, "model_type": "temporal_cross_task",
        "network_config": asdict(network.config),
        "state_dict": {k: v.detach().cpu() for k, v in network.state_dict().items()},
        "mean": torch.from_numpy(standardizer.mean_), "scale": torch.from_numpy(standardizer.scale_),
        "input_shape": list(standardizer.shape_), "threshold": float(threshold),
        "segment_names": list(SEGMENTS), "task_names": list(TASKS), "metadata": metadata,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_network(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("model_type") != "temporal_cross_task":
        raise ValueError("Unsupported temporal/cross-task model bundle.")
    if payload["segment_names"] != list(SEGMENTS) or payload["task_names"] != list(TASKS):
        raise ValueError("Saved task order differs from this implementation.")
    standardizer = TaskStandardizer()
    standardizer.mean_ = payload["mean"].numpy()
    standardizer.scale_ = payload["scale"].numpy()
    standardizer.shape_ = tuple(payload["input_shape"])
    if (standardizer.mean_.shape != (1, 1, 12, 1) or standardizer.scale_.shape != (1, 1, 12, 1)
            or not np.isfinite(standardizer.mean_).all() or not np.isfinite(standardizer.scale_).all()
            or np.any(standardizer.scale_ <= 0) or not math.isfinite(payload["threshold"])):
        raise ValueError("Invalid saved normalization or threshold.")
    network = TemporalCrossTaskNet(NetworkConfig(**payload["network_config"]))
    if standardizer.shape_ != (5, 12, network.config.samples_per_segment):
        raise ValueError("Saved normalization and network shapes differ.")
    network.load_state_dict(payload["state_dict"], strict=True)
    network.to(device).eval()
    return network, standardizer, payload
