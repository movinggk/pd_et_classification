"""Differentiable adapters for seven released encoders and optional MantisV2.

The same representation is used in frozen and fine-tuned modes. There are no
cohort-fitted statistics here. Chronos group IDs never connect participants or
different task segments. Upstream inference-only decorators are not used.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from functools import lru_cache
from functools import partial
import json
import math
from pathlib import Path
import urllib.request
import warnings
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..data import PROJECT_ROOT, sha256_file
from .registry import SPECS
from .progress import progress_bar
from .fast_paths import (tirex_hidden_features, vqshape_token_features, tivit_tensor_features,
                         prepare_tivit_inputs, tivit_prepared_features,
                         prepare_vqshape_values, vqshape_normalized_features)
from .attention_fast import install_moment_position_cache, chronos_packed_group_features
from .tspulse_fast import tspulse_last_hidden
from .moment_input import prepare_moment_input
from .vqshape_config import vqshape_kwargs_from_hyperparameters
from .tirex_stable import install_stable_recurrence


ADAPTER_VERSION = 3


@dataclass(frozen=True)
class EncoderOptions:
    model: str
    device: str = "cpu"
    channel_batch_size: int | None = None
    gradient_checkpointing: bool | None = None
    chronos_group_size: int = 12
    adapter_version: int = field(default=ADAPTER_VERSION, init=False)

    def __post_init__(self):
        if self.model not in SPECS:
            raise ValueError(f"Unknown model: {self.model}")
        object.__setattr__(self, "channel_batch_size", resolve_channel_batch_size(self.model, self.channel_batch_size))
        object.__setattr__(self, "gradient_checkpointing", resolve_gradient_checkpointing(self.model, self.gradient_checkpointing))
        if self.device not in ("cpu", "cuda", "mps") or self.channel_batch_size < 1:
            raise ValueError("Need device cpu/cuda/mps and a positive channel batch size.")
        if self.chronos_group_size < 1:
            raise ValueError("chronos_group_size must be positive.")


def resolve_channel_batch_size(model, requested=None):
    """Small TSPulse batches waste GPU launches; larger backbones stay bounded."""
    return requested if requested is not None else {"tspulse": 128, "chronos2": 24}.get(model, 8)


def resolve_gradient_checkpointing(model, requested=None):
    """Recomputation saves memory for large models, but costs more for TSPulse."""
    return requested if requested is not None else model != "tspulse"


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable.")
    return requested


def resize_channel_batch(x, length=512):
    return F.interpolate(x[:, None, :], size=length, mode="linear", align_corners=False)[:, 0]


@lru_cache(maxsize=128)
def _hash_file_version(path, identity):
    return sha256_file(Path(path))


def checkpoint_sha256(path):
    """Hash each unchanged local checkpoint once per process, not per CV fit."""
    path = Path(path).resolve()
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return _hash_file_version(str(path), identity)


@lru_cache(maxsize=32)
def _cached_hub_snapshot(model_id, revision, cache_dir):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=model_id, revision=revision, cache_dir=cache_dir,
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
    ))


def _hub_snapshot(model_id, revision, cache_dir):
    path = _cached_hub_snapshot(model_id, revision, cache_dir)
    if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        _cached_hub_snapshot.cache_clear()
        path = _cached_hub_snapshot(model_id, revision, cache_dir)
    return path


_hub_snapshot.cache_clear = _cached_hub_snapshot.cache_clear


@lru_cache(maxsize=32)
def _cached_hub_file(model_id, filename, revision, cache_dir):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir))


def _hub_file(model_id, filename, revision, cache_dir):
    path = _cached_hub_file(model_id, filename, revision, cache_dir)
    if not path.is_file():
        _cached_hub_file.cache_clear()
        path = _cached_hub_file(model_id, filename, revision, cache_dir)
    return path


_hub_file.cache_clear = _cached_hub_file.cache_clear


@contextmanager
def tspulse_fft_warning_guard(device_type):
    """Filter the verified PyTorch 2.11.0 MPS FFT buffer-resize warning only.

    The FFT math and autograd path are unchanged. The upstream MPS issue is
    https://github.com/pytorch/pytorch/issues/174003. The CPU/MPS comparison
    is recorded in tmp/foundation_validation/mps_fft_warning_validation.json.
    """
    if device_type != "mps" or str(torch.__version__).split("+")[0] != "2.11.0":
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning,
            message=(r"An output with one or more elements was resized since it had shape \[\], "
                     r"which does not match the required output shape \[\d+, 257, 1\]\."),
            module=r"tsfm_public\.models\.tspulse\.modeling_tspulse$",
        )
        yield


class Encoder(nn.Module):
    """Input (participants, channels, samples), output (participants, features)."""

    def __init__(self, backbone, options, provenance=None):
        super().__init__()
        self.backbone = backbone
        self.options = options
        self.provenance = provenance or {}
        if options.model == "moment":
            count = install_moment_position_cache(backbone)
            self.provenance = {**self.provenance, "attention_execution": {
                "cached_position_bucket_layers": count,
                "policy": "Native T5 attention; only deterministic integer position buckets cached. Learned bias values and gradients remain native.",
            }}
        if options.model == "chronos2":
            self.provenance = {**self.provenance, "group_attention_execution": (
                "Complete equal-size tasks packed independently; native preparation/time attention/context pooling. Incomplete contexts or unsupported versions use native encode."
            )}
        if options.model == "tirex":
            cells = install_stable_recurrence(backbone)
            self.provenance = {**self.provenance, "tirex_recurrence_stability": {
                "cells": cells, "operation": "exp(min(log_gate, 0)) instead of min(exp(log_gate), 1)",
                "recurrent_precision": "Native BF16 state/matmul and FP32 pointwise equations retained.",
                "recurrent_backward": "Native forward; analytic headwise backward avoids per-series recurrent-weight copies.",
            }}

    def feature_dimension(self, channels):
        """Read fixed readout widths without running a participant forward."""
        name, model = self.options.model, self.backbone
        if name in ("utica", "mantis_v2"):
            width = model.hidden_dim
        elif name in ("moment", "chronos2"):
            width = model.config.d_model
        elif name == "tspulse":
            width = model.config.d_model_layerwise[-1] * (3 if model.config.patch_register_tokens else 2)
        elif name == "tirex":
            width = model.config.block_kwargs["num_blocks"] * model.config.block_kwargs["embedding_dim"]
        elif name == "tivit":
            width = model.vit.config.hidden_size
        elif name == "vqshape":
            width = model.num_token * (model.codebook.embedding.embedding_dim + 4)
        else:
            raise ValueError(f"Unknown feature width: {name}")
        return channels * width

    def _channels(self, values):
        name = self.options.model
        if name in ("utica", "mantis_v2"):
            return self.backbone(resize_channel_batch(values)[:, None])
        if name == "moment":
            x, mask = prepare_moment_input(values, self.backbone.config.seq_len, self.backbone.patch_len)
            return self.backbone(x_enc=x, input_mask=mask, reduction="mean").embeddings
        if name == "tspulse":
            x = resize_channel_batch(values)[:, :, None]
            with tspulse_fft_warning_guard(x.device.type):
                hidden = tspulse_last_hidden(self.backbone, x)
            registers = self.backbone.config.patch_register_tokens or 0
            patches = (hidden.shape[2] - registers) // 2
            if patches < 1 or not self.backbone.config.fuse_fft:
                raise ValueError("The TSPulse comparison requires the dual time/FFT checkpoint.")
            views = [hidden[:, 0, :patches].mean(1), hidden[:, 0, patches:2 * patches].mean(1)]
            if registers:
                views.append(hidden[:, 0, -registers:].mean(1))
            return torch.cat(views, -1)
        if name == "tirex":
            return tirex_hidden_features(self.backbone, values)
        if name == "tivit":
            return tivit_tensor_features(self.backbone, values, patch_size=math.isqrt(values.shape[-1]))
        if name == "vqshape":
            return vqshape_token_features(self.backbone, resize_channel_batch(values))
        raise RuntimeError(f"Missing per-channel adapter for {name}")

    def _chronos_group(self, values, complete_context=True):
        return chronos_packed_group_features(self.backbone, values, complete_context=complete_context)

    def _moment_prepared(self, values, mask):
        return self.backbone(x_enc=values, input_mask=mask, reduction="mean").embeddings

    def forward(self, x):
        if x.ndim != 3 or min(x.shape) < 1 or x.shape[-1] < 2:
            raise ValueError("Expected nonempty (participants, channels, samples>=2).")
        count, channels, length = x.shape
        checkpointed = self.training and torch.is_grad_enabled() and self.options.gradient_checkpointing
        if self.options.model == "chronos2":
            group_size = self.options.chronos_group_size
            if channels % group_size:
                raise ValueError("Chronos channel count must be divisible by its task group size.")
            groups = x.reshape(-1, group_size, length)
            group_batch = max(1, self.options.channel_batch_size // group_size)
            # One check per participant batch, never per task/layer or during
            # checkpoint replay. Fully missing context patches keep native masks.
            complete_context = group_batch > 1 and bool(torch.isfinite(x).all())
            parts = []
            for batch in groups.split(group_batch):
                z = (checkpoint(self._chronos_group, batch, complete_context, use_reentrant=False)
                     if checkpointed else self._chronos_group(batch, complete_context))
                parts.append(z)
            return torch.cat(parts).reshape(count, -1)
        flat = x.reshape(count * channels, length)
        parts = []
        for batch in flat.split(self.options.channel_batch_size):
            function, inputs = self._channels, (batch,)
            if self.options.model == "tivit":
                prepared, is_quantized = prepare_tivit_inputs(self.backbone, batch, patch_size=math.isqrt(length))
                function = partial(tivit_prepared_features, self.backbone, is_quantized=is_quantized)
                inputs = (prepared,)
            elif self.options.model == "moment":
                inputs = prepare_moment_input(batch, self.backbone.config.seq_len, self.backbone.patch_len)
                function = self._moment_prepared
            elif self.options.model == "vqshape":
                inputs = (prepare_vqshape_values(resize_channel_batch(batch)),)
                function = partial(vqshape_normalized_features, self.backbone)
            elif self.options.model in ("utica", "mantis_v2"):
                inputs = (resize_channel_batch(batch)[:, None],)
                function = self.backbone
            z = checkpoint(function, *inputs, use_reentrant=False) if checkpointed else function(*inputs)
            parts.append(z)
        features = torch.cat(parts).reshape(count, -1)
        if self.options.model == "tirex":
            # Author ordering is layer, channel, hidden coordinate. Per-layer
            # L2 normalization is already applied by tirex_hidden_features.
            config = self.backbone.config.block_kwargs
            features = features.reshape(count, channels, config["num_blocks"], config["embedding_dim"])
            features = features.transpose(1, 2).flatten(1)
            return F.layer_norm(features, (features.shape[-1],))
        if self.options.model == "tivit":
            return F.normalize(features, p=2, dim=-1)
        return features


@lru_cache(maxsize=1)
def _package_versions():
    result = {}
    for package in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "joblib",
                    "transformers", "huggingface-hub", "safetensors", "tqdm", "mantis-tsfm",
                    "momentfm", "granite-tsfm", "chronos-forecasting", "tirex-ts", "open_clip_torch"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def package_versions():
    return _package_versions().copy()


def _snapshot(spec, cache_dir):
    path = _hub_snapshot(spec.model_id, spec.revision, str(cache_dir))
    return path, {p.name: checkpoint_sha256(p) for p in path.iterdir() if p.is_file()}


def _check_loaded(info):
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Incomplete pretrained backbone: {info}")


def load_utica_state(backbone, state):
    """Explicitly adapt the pinned self-distillation checkpoint to MantisV1.

    Its projection head is absent and unused for intermediate-layer features.
    Its mask token and final normalization belong to pretraining/final-layer
    readout, and are not part of MantisV1's layer-2 representation. Every tensor
    in the selected feature-extraction path must still load strictly.
    """
    normalized = {("transf_unit." + k[len("vit_unit."):] if k.startswith("vit_unit.") else k): v
                  for k, v in state.items()}
    ignored = {"transf_unit.mask_token", "transf_unit.norm.weight", "transf_unit.norm.bias"}
    if not ignored.issubset(normalized) or any(k.startswith("prj.") for k in normalized):
        raise ValueError("UTICA checkpoint structure differs from its pinned release.")
    backbone.prj = nn.Identity()
    backbone.load_state_dict({k: v for k, v in normalized.items() if k not in ignored}, strict=True)
    return sorted(ignored)


def _load_vqshape(cache_dir):
    from ._vendor.vqshape.model import VQShape

    directory = Path(cache_dir) / "vqshape-v0.1.0-cls"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "uea_dim256_codebook512.zip"
    checkpoint_path = directory / "VQShape.ckpt"
    url = "https://github.com/YunshiWen/VQShape/releases/download/v0.1.0-cls/uea_dim256_codebook512.zip"
    if not checkpoint_path.exists():
        if not archive.exists():
            temporary = archive.with_suffix(".download")
            try:
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                    import shutil
                    shutil.copyfileobj(response, out)
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
        with zipfile.ZipFile(archive) as z:
            members = [i for i in z.infolist() if i.filename.endswith("/VQShape.ckpt") or i.filename == "VQShape.ckpt"]
            if len(members) != 1:
                raise ValueError("Expected exactly one VQShape.ckpt in the released archive.")
            # Copy one member by content; never extract archive-provided paths.
            temporary = checkpoint_path.with_suffix(".tmp")
            temporary.write_bytes(z.read(members[0]))
            temporary.replace(checkpoint_path)
    with torch.serialization.safe_globals([argparse.Namespace]):
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = VQShape(**vqshape_kwargs_from_hyperparameters(saved["hyper_parameters"]))
    state = {k.removeprefix("model."): v for k, v in saved["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model, {"VQShape.ckpt": checkpoint_sha256(checkpoint_path)}


def load_encoder(options: EncoderOptions, cache_dir=None):
    """Always load published weights. Tiny random models exist only in tests."""
    options = replace(options, device=resolve_device(options.device))
    cache_dir = Path(cache_dir or PROJECT_ROOT / "models" / "foundation_cache").expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = SPECS[options.model]
    name = options.model
    if name == "vqshape":
        backbone, hashes = _load_vqshape(cache_dir)
    elif name == "tirex":
        from tirex.models.tirex import TiRexZero
        path = _hub_file(spec.model_id, "model.ckpt", spec.revision, str(cache_dir))
        # The official local loader restores all weights strictly; avoid a
        # second Hub lookup for the checkpoint we just resolved.
        backbone = TiRexZero.from_pretrained(str(path), device="cpu", backend="torch", compile=False)
        hashes = {"model.ckpt": checkpoint_sha256(path)}
    elif name == "utica":
        from mantis.architecture import MantisV1
        path = _hub_file(spec.model_id, "pytorch_model.bin", spec.revision, str(cache_dir))
        backbone = MantisV1(device="cpu", return_transf_layer=2, output_token="combined")
        ignored_utica = load_utica_state(backbone, torch.load(path, map_location="cpu", weights_only=True))
        hashes = {"pytorch_model.bin": checkpoint_sha256(path)}
    else:
        directory, hashes = _snapshot(spec, cache_dir)
        if name == "mantis_v2":
            from mantis.architecture import MantisV2
            from safetensors.torch import load_file
            config = json.loads((directory / "config.json").read_text())
            config.update(device="cpu", return_transf_layer=2, output_token="combined", pre_training=False)
            backbone = MantisV2(**config)
            backbone.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        elif name == "moment":
            from momentfm import MOMENTPipeline
            backbone = MOMENTPipeline.from_pretrained(str(directory), strict=True, model_kwargs={
                "task_name": "embedding", "freeze_encoder": False, "freeze_embedder": False,
                "enable_gradient_checkpointing": False,
            })
            backbone.init()
        elif name == "tspulse":
            from tsfm_public.models.tspulse import TSPulseForReconstruction
            loaded, info = TSPulseForReconstruction.from_pretrained(str(directory), output_loading_info=True)
            _check_loaded(info)
            backbone = loaded.backbone
        elif name == "chronos2":
            from chronos.chronos2 import Chronos2Model
            backbone, info = Chronos2Model.from_pretrained(str(directory), output_loading_info=True)
            _check_loaded(info)
        elif name == "tivit":
            from transformers import CLIPImageProcessor, CLIPVisionModel
            from ._vendor.tivit.tivit import TiViT_HF
            vision, info = CLIPVisionModel.from_pretrained(str(directory), output_loading_info=True)
            _check_loaded(info)
            backbone = TiViT_HF(
                processor=CLIPImageProcessor.from_pretrained(str(directory)), vit=vision.vision_model,
                layer_idx=14, aggregation="mean", patch_size=31, stride=.1,
            )
        else:
            raise RuntimeError(f"Unimplemented checkpoint loader: {name}")
    # Load the original feature-path modules; unused stored modules receive
    # no classification gradient and AdamW allocates no state for them.
    encoder = Encoder(backbone, options, {"specification": asdict(spec), "checkpoint_sha256": hashes,
                                         "options": asdict(options), "package_versions": package_versions()})
    encoder.provenance["comparison_recipe"] = {
        "type": "shared_classifier_backbone_comparison",
        "author_classification_recipe_reproduction": False,
        "adapter_version": options.adapter_version,
        "note": "Pretrained backbone readouts with shared nested-CV Ridge/neural heads; model-specific published training recipes are not reproduced.",
    }
    if name == "tirex":
        encoder.provenance["participant_feature_processing"] = (
            "Author TiRexEmbedding order (layer,channel,hidden), then participant LayerNorm; optional augmentation disabled."
        )
    elif name == "tivit":
        encoder.provenance["participant_feature_processing"] = (
            "L2 normalization after concatenating all channel features, matching the author embedding wrapper."
        )
    if name == "utica":
        encoder.provenance["excluded_pretraining_only_tensors"] = ignored_utica
        encoder.provenance["unused_projection_head"] = "Removed; absent in checkpoint and never called for features."
    if name == "moment":
        encoder.provenance["input_processing"] = (
            "v3: left-zero-pad short series with observed mask; endpoint-preserving interpolation for long series; native RevIN and masked patch mean."
        )
    elif name == "tivit":
        encoder.provenance["signal_patch_policy"] = (
            "floor(sqrt(input_length)); 31 at PADS length 976."
        )
    elif name == "vqshape":
        encoder.provenance["architecture_metadata_policy"] = "All inference architecture fields required from checkpoint; no guessed attention-head counts."
    if name == "tspulse":
        encoder.provenance["mps_fft_resize_warning_filter"] = {
            "enabled": options.device == "mps" and str(torch.__version__).split("+")[0] == "2.11.0",
            "scope": "Only the upstream scalar-to-[batch,257,1] FFT buffer-resize UserWarning during this backbone call.",
            "upstream_issue": "https://github.com/pytorch/pytorch/issues/174003",
        }
    if name in ("tirex", "vqshape", "tivit", "tspulse"):
        encoder.provenance["feature_only_forward"] = "Unused forecasting/reconstruction heads and losses omitted; original backbone state layout retained."
        if name == "tspulse":
            encoder.provenance["feature_only_forward"] = "One FFT for classifier features; unused second FFT and reconstruction-target scaler omitted for verified granite-tsfm/configuration, otherwise upstream fallback."
        elif name == "tivit":
            encoder.provenance["feature_only_forward"] = "Official signal-to-image and 8-bit quantization retained; tensor lookup replaces PIL/NumPy processing for the verified CLIP processor, otherwise upstream fallback."
        if name == "vqshape":
            encoder.provenance["dropout_rng_note"] = "Skipping unused decoder dropout changes RNG consumption across training chunks relative to the older adapter."
    encoder.to(options.device).float().eval()
    for module in encoder.modules():
        # Mantis stores the runtime device separately from parameter devices.
        if module.__class__.__module__.startswith("mantis.") and hasattr(module, "device"):
            module.device = options.device
    return encoder


def extract_features(encoder, X, batch_size=2, *, progress=False, description=None,
                     progress_position=0):
    """Independent inference with feature ordering fixed by model and adapter version."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("Need finite nonempty (participants, channels, samples) input.")
    encoder.eval().requires_grad_(False)
    embeddings = None
    with torch.inference_mode(), progress_bar(
        total=len(X), desc=description or f"{encoder.options.model} features",
        unit="participant", enabled=progress, position=progress_position,
    ) as bar:
        for start in range(0, len(X), batch_size):
            x = torch.from_numpy(np.array(X[start:start + batch_size], copy=True)).to(encoder.options.device)
            z = encoder(x).float().cpu().numpy()
            if z.ndim != 2 or len(z) != len(x) or not np.isfinite(z).all():
                raise ValueError("Encoder returned invalid features.")
            if embeddings is None:
                embeddings = np.empty((len(X), z.shape[1]), dtype=np.float32)
            embeddings[start:start + len(x)] = z
            bar.update(len(x))
    return embeddings
