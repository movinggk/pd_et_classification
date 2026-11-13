"""Trainable MantisV2 participant classifier, with fold-local feature scaling.

The backbone starts from the same pinned checkpoint / third-layer combined
representation as the frozen experiment. No training, data loading or network
access happens on import. Training ALWAYS takes raw participant signals, not
cached adapted features. Initial frozen embeddings fit normalization only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm

from .task_interaction import CHANNEL_INDICES, balanced_class_weights, seed_torch
from .transformer import CHECKPOINT_HASHES, MODEL_ID, REVISION, EncoderConfig, load_encoder


EMBEDDING_DIM = 512
MODES = ("last-block", "full", "head-only")


def select_inputs(X, input_set):
    X = np.asarray(X, dtype=np.float32)
    if input_set not in ("all", "three_tasks"):
        raise ValueError("input_set must be all or three_tasks.")
    if X.ndim != 3 or len(X) < 1 or X.shape[-1] < 2 or not np.isfinite(X).all():
        raise ValueError("Expected finite nonempty (participants,channels,samples>=2) input.")
    if input_set == "all" and X.shape[1] != 132:
        raise ValueError("The all input protocol requires 132 official channels.")
    if input_set == "three_tasks":
        if X.shape[1] == 132:
            X = X[:, CHANNEL_INDICES, :]
        elif X.shape[1] != 60:
            raise ValueError("The three_tasks protocol requires official 132 or selected 60 channels.")
    return np.ascontiguousarray(X)


@dataclass(frozen=True)
class FinetuneConfig:
    channels: int = 132
    samples: int = 976
    mode: str = "last-block"
    head_dropout: float = 0.1
    channel_batch_size: int = 16
    activation_checkpointing: bool = True

    def __post_init__(self):
        for value in (self.channels, self.samples, self.channel_batch_size):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("Channels, sample count and channel batch size must be positive integers.")
        if self.samples < 2 or self.mode not in MODES or not 0 <= self.head_dropout < 1:
            raise ValueError("Invalid fine-tuning mode, sample count or head dropout.")


@dataclass(frozen=True)
class TrainingConfig:
    encoder_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 2
    accumulation_steps: int = 8
    warmup_epochs: int = 1
    max_grad_norm: float = 1.0

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.encoder_lr, self.head_lr, self.weight_decay, self.max_grad_norm))
                or self.encoder_lr < 0 or self.head_lr <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0):
            raise ValueError("Learning rates/decay/gradient norm must be finite and in their declared ranges.")
        if (any(not isinstance(v, int) or isinstance(v, bool) for v in
                (self.batch_size, self.accumulation_steps, self.warmup_epochs))
                or min(self.batch_size, self.accumulation_steps) < 1 or self.warmup_epochs < 0):
            raise ValueError("Batch/accumulation counts must be positive; warmup must be nonnegative.")


def feature_statistics(initial_features):
    """Fit one fixed mean/std per feature using TRAINING participants only.

    Statistics are fitted on the INITIAL pretrained representation and stay
    fixed while the encoder adapts. Constant initial features use scale 1.
    """
    Z = np.asarray(initial_features, dtype=np.float32)
    if Z.ndim != 2 or len(Z) < 2 or Z.shape[1] < 1 or not np.isfinite(Z).all():
        raise ValueError("Expected at least two finite training feature rows.")
    mean = np.mean(Z, axis=0, dtype=np.float64)
    scale = np.std(Z, axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def build_uninitialized_encoder(architecture):
    """Reconstruct the local pinned package architecture; no downloads."""
    if version("mantis-tsfm") != "1.1.0":
        raise RuntimeError("This fine-tuning adapter requires mantis-tsfm==1.1.0.")
    from mantis.architecture import MantisV2

    parameters = dict(architecture)
    parameters.update(device="cpu", return_transf_layer=2, output_token="combined", pre_training=False)
    encoder = MantisV2(**parameters)
    encoder.remove_transf_layers()
    return encoder


class FinetunedMantis(nn.Module):
    """One class-weighted logit per participant, preserving channel identity."""

    def __init__(self, encoder, architecture, config, feature_mean, feature_scale):
        super().__init__()
        self.config = config
        self.architecture = dict(architecture)
        self.encoder = encoder
        if getattr(encoder, "return_transf_layer", None) != 2 or getattr(encoder, "output_token", None) != "combined":
            raise ValueError("Require the pinned third-layer combined-token representation.")
        encoder.remove_transf_layers()
        layers = encoder.transf_unit.transformer.layers
        if len(layers) != 3:
            raise ValueError("Expected exactly the three active pretrained Transformer blocks.")
        encoder.requires_grad_(False)
        if config.mode == "last-block":
            layers[-1].requires_grad_(True)
        elif config.mode == "full":
            encoder.requires_grad_(True)
            # The pretraining projector is not on the classification forward path.
            encoder.prj.requires_grad_(False)
        width = config.channels * EMBEDDING_DIM
        mean = torch.as_tensor(feature_mean, dtype=torch.float32).clone()
        scale = torch.as_tensor(feature_scale, dtype=torch.float32).clone()
        if (mean.shape != (width,) or scale.shape != (width,) or not torch.isfinite(mean).all()
                or not torch.isfinite(scale).all() or torch.any(scale <= 0)):
            raise ValueError("Feature statistics must be finite, positive-scale vectors of channels*512.")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        self.head = nn.Sequential(nn.Dropout(config.head_dropout), nn.Linear(width, 1))
        self.encoder_warmup = False
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        # Frozen pretrained blocks stay in eval mode; changing this would add
        # train-time noise to a supposedly frozen representation.
        self.encoder.eval()
        if mode and not self.encoder_warmup:
            if self.config.mode == "full":
                self.encoder.train()
                self.encoder.prj.eval()
            elif self.config.mode == "last-block":
                self.encoder.transf_unit.transformer.layers[-1].train()
        return self

    def set_warmup(self, enabled):
        self.encoder_warmup = bool(enabled)
        self.train(self.training)

    def embeddings(self, X):
        if X.ndim != 3 or tuple(X.shape[1:]) != (self.config.channels, self.config.samples):
            raise ValueError("Participant input shape differs from the saved channel/sample protocol.")
        batch, channels, length = X.shape
        records = X.reshape(batch * channels, 1, length)
        records = nn.functional.interpolate(records, size=512, mode="linear", align_corners=False)
        chunks = []
        adapted = self.config.mode != "head-only" and not self.encoder_warmup
        for start in range(0, len(records), self.config.channel_batch_size):
            values = records[start:start + self.config.channel_batch_size]
            if adapted and torch.is_grad_enabled():
                if self.training and self.config.activation_checkpointing:
                    encoded = checkpoint(self.encoder, values, use_reentrant=False, preserve_rng_state=True)
                else:
                    encoded = self.encoder(values)
            else:
                # no_grad, rather than inference_mode: encoder caches must stay
                # usable by a later gradient-enabled training epoch.
                with torch.no_grad():
                    encoded = self.encoder(values)
            if encoded.shape != (len(values), EMBEDDING_DIM):
                raise RuntimeError("MantisV2 must return 512 features per independent channel.")
            chunks.append(encoded)
        return torch.cat(chunks, dim=0).reshape(batch, channels * EMBEDDING_DIM)

    def forward(self, X):
        Z = self.embeddings(X)
        return self.head((Z - self.feature_mean) / self.feature_scale).squeeze(-1)

    def parameter_counts(self):
        return {"stored_total": sum(p.numel() for p in self.parameters()),
                "encoder_trainable": sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
                "head_trainable": sum(p.numel() for p in self.head.parameters() if p.requires_grad)}

    def move_to(self, device):
        self.encoder.to(device)  # Also updates the upstream encoder's device attribute.
        self.to(device)
        return self


def create_model(config, checkpoint_dir, initial_training_features, device="cpu"):
    """A FRESH pinned backbone per inner/outer/final fit, never an adapted cache."""
    encoder, _ = load_encoder(EncoderConfig(device="cpu"), checkpoint_dir)
    architecture = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    mean, scale = feature_statistics(initial_training_features)
    return FinetunedMantis(encoder, architecture, config, mean, scale).move_to(device)


def predict_scores(model, X, *, device="cpu", batch_size=2, progress=False,
                   progress_desc="Prediction", progress_position=0):
    X = np.asarray(X, dtype=np.float32)
    if (X.ndim != 3 or len(X) == 0 or tuple(X.shape[1:]) != (model.config.channels, model.config.samples)
            or not np.isfinite(X).all() or batch_size < 1):
        raise ValueError("Expected finite participant signals with the model's channel/sample shape.")
    model.eval()
    scores = []
    # Avoid inference tensors in persistent upstream rotary caches before the
    # next fine-tuning epoch. Evaluation remains gradient-free.
    with torch.no_grad(), tqdm(
        total=len(X), desc=progress_desc, unit="participant", position=progress_position,
        dynamic_ncols=True, leave=False, disable=not progress,
    ) as bar:
        for start in range(0, len(X), batch_size):
            values = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size])).to(device)
            scores.append(model(values).cpu().numpy())
            bar.update(len(values))
    result = np.concatenate(scores).astype(float)
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise RuntimeError("Missing or nonfinite participant logits.")
    return result


def fit_finetune(X, y, initial_training_features, config, training, checkpoint_dir, *,
                 epochs, seed, device="cpu", validation=None, checkpoint_epochs=(),
                 progress=False, progress_desc="Training", progress_position=1):
    """Fit only supplied TRAINING participants; validation is forward-only.

    Losses are normalized by the actual participant count in each accumulated
    optimizer step, including the final incomplete group. Cache data fit only
    normalization; the forward/backward path consumes raw X throughout.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    class_weight = balanced_class_weights(y)
    if (X.shape != (len(y), config.channels, config.samples) or not np.isfinite(X).all()
            or np.shape(initial_training_features) != (len(y), config.channels * EMBEDDING_DIM)):
        raise ValueError("Training signals, labels and initial feature rows must align.")
    if (not isinstance(epochs, int) or epochs < 1 or any(e < 1 or e > epochs for e in checkpoint_epochs)
            or (checkpoint_epochs and validation is None)):
        raise ValueError("Invalid epoch/validation checkpoint specification.")
    if config.mode != "head-only" and (training.encoder_lr <= 0 or epochs <= training.warmup_epochs):
        raise ValueError("Fine-tuning requires positive encoder LR and epochs beyond head warmup.")
    seed_torch(seed)
    model = create_model(config, checkpoint_dir, initial_training_features, device)
    head_parameters = [p for p in model.head.parameters() if p.requires_grad]
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = [{"params": head_parameters, "lr": training.head_lr}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": training.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=training.weight_decay)
    tensor_X = torch.from_numpy(np.ascontiguousarray(X))
    targets = torch.as_tensor(y, dtype=torch.float32)
    weights = torch.as_tensor(class_weight, device=device)
    generator = torch.Generator().manual_seed(int(seed))
    predictions, history = {}, []
    participants_per_step = training.batch_size * training.accumulation_steps
    with tqdm(total=epochs, desc=progress_desc, unit="epoch", position=progress_position,
              dynamic_ncols=True, leave=False, disable=not progress) as epoch_bar:
        for epoch in range(1, epochs + 1):
            model.set_warmup(config.mode != "head-only" and epoch <= training.warmup_epochs)
            model.train()
            order = torch.randperm(len(y), generator=generator)
            loss_sum, processed = 0.0, 0
            with tqdm(total=len(y), desc=f"Epoch {epoch}/{epochs}", unit="participant",
                      position=progress_position + 1, dynamic_ncols=True, leave=False,
                      disable=not progress) as batch_bar:
                for group_start in range(0, len(y), participants_per_step):
                    group = order[group_start:group_start + participants_per_step]
                    optimizer.zero_grad(set_to_none=True)
                    for start in range(0, len(group), training.batch_size):
                        index = group[start:start + training.batch_size]
                        target = targets[index].to(device)
                        logits = model(tensor_X[index].to(device))
                        losses = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights[target.long()]
                        loss = losses.sum() / len(group)
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite fine-tuning loss.")
                        loss.backward()
                        loss_sum += float(losses.detach().sum().cpu())
                        processed += len(index)
                        batch_bar.set_postfix(BCE=f"{loss_sum / processed:.4f}", refresh=False)
                        batch_bar.update(len(index))
                    nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
            history.append({"epoch": epoch, "training_weighted_bce": loss_sum / len(y),
                            "head_warmup": bool(model.encoder_warmup)})
            if epoch in checkpoint_epochs:
                predictions[epoch] = predict_scores(
                    model, validation, device=device, batch_size=training.batch_size,
                    progress=progress, progress_desc="Validation", progress_position=progress_position + 1,
                )
            epoch_bar.set_postfix(BCE=f"{history[-1]['training_weighted_bce']:.4f}",
                                  phase="warmup" if model.encoder_warmup else "train", refresh=False)
            epoch_bar.update(1)
            if not progress and (epoch in checkpoint_epochs or epoch == epochs):
                print(f"      epoch {epoch}/{epochs}: training weighted BCE={history[-1]['training_weighted_bce']:.6f}", flush=True)
    return model, predictions, history


def save_finetuned(path, model, *, input_set, metadata):
    if (input_set not in ("all", "three_tasks") or model.config.channels != (132 if input_set == "all" else 60)):
        raise ValueError("Saved channel count must match the named PADS input protocol.")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError("Refusing to save nonfinite fine-tuned weights.")
    payload = {"format_version": 1, "model_type": "mantis_v2_finetuned",
               "model_config": asdict(model.config), "architecture": model.architecture,
               "state_dict": state, "input_set": input_set, "decision_threshold": 0.0,
               "initial_checkpoint": {"model_id": MODEL_ID, "revision": REVISION, "sha256": dict(CHECKPOINT_HASHES)},
               "metadata": metadata}
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_finetuned(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("model_type") != "mantis_v2_finetuned":
        raise ValueError("Unsupported fine-tuned MantisV2 bundle.")
    origin = payload.get("initial_checkpoint", {})
    if origin != {"model_id": MODEL_ID, "revision": REVISION, "sha256": dict(CHECKPOINT_HASHES)}:
        raise ValueError("Saved model has an unexpected initial checkpoint identity.")
    config = FinetuneConfig(**payload["model_config"])
    if config.channels != (132 if payload["input_set"] == "all" else 60) or payload["input_set"] not in ("all", "three_tasks"):
        raise ValueError("Saved input protocol and channel count disagree.")
    if payload.get("decision_threshold") != 0.0:
        raise ValueError("This protocol fixes the ET logit threshold at zero.")
    state = payload["state_dict"]
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Saved model contains nonfinite tensors.")
    encoder = build_uninitialized_encoder(payload["architecture"])
    model = FinetunedMantis(encoder, payload["architecture"], config, state["feature_mean"], state["feature_scale"])
    model.load_state_dict(state, strict=True)
    model.move_to(device).eval()
    return model, payload
