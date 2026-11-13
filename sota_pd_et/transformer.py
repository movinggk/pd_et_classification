"""Pinned, frozen MantisV2 embeddings; no fitting on the PADS cohort.

Each channel is independently resized along time, encoded, and concatenated
in its original position. Flattening the participant/channel axes here batches
independent records; it never concatenates channels along the time axis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time
import urllib.request

import numpy as np
from tqdm.auto import tqdm

from .data import PROJECT_ROOT, sha256_file


MODEL_ID = "paris-noah/MantisV2"
REVISION = "99fe0f548960e272fbfa4b82fd9b5b5956779dfd"
CHECKPOINT_HASHES = {
    "config.json": "f28885b7aa662cedcaf5627659fc0fba9cf8ee49d5b991280522e04fa5b1ae80",
    "model.safetensors": "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1",
}


@dataclass(frozen=True)
class EncoderConfig:
    model_id: str = MODEL_ID
    revision: str = REVISION
    batch_size: int = 128
    device: str = "cpu"
    sequence_length: int = 512
    return_transf_layer: int = 2
    output_token: str = "combined"

    def __post_init__(self):
        if self.model_id != MODEL_ID or self.revision != REVISION:
            raise ValueError("This experiment requires the pinned MantisV2 checkpoint.")
        if self.batch_size < 1 or self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("Use a positive batch size and device cpu, mps or cuda.")
        if (self.sequence_length, self.return_transf_layer, self.output_token) != (512, 2, "combined"):
            raise ValueError("The predefined representation is length 512, layer 2, combined tokens.")


def ensure_checkpoint(directory: Path) -> dict:
    """Copy verified workspace weights or fetch only the two pinned Hub files."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cache = PROJECT_ROOT / "models" / "mantis_v2"
    for name, expected in CHECKPOINT_HASHES.items():
        target = directory / name
        if not target.exists():
            cached = cache / name
            if cached.is_file() and sha256_file(cached) == expected:
                shutil.copyfile(cached, target)
            else:
                url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
                temporary = target.with_suffix(target.suffix + ".download")
                try:
                    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                        shutil.copyfileobj(response, stream)
                    if sha256_file(temporary) != expected:
                        raise ValueError(f"Checkpoint SHA-256 mismatch: {name}")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        if sha256_file(target) != expected:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {name}")
    return {"directory": str(directory), "sha256": dict(CHECKPOINT_HASHES)}


def load_encoder(config: EncoderConfig, checkpoint_dir: Path):
    """Load all checkpoint tensors strictly, freeze parameters, disable dropout."""
    import torch
    from mantis.architecture import MantisV2
    from safetensors.torch import load_file

    if version("mantis-tsfm") != "1.1.0":
        raise RuntimeError("This adapter was verified with mantis-tsfm==1.1.0.")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS requested but unavailable in this execution environment.")
    checkpoint = ensure_checkpoint(checkpoint_dir)
    parameters = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    parameters.update(device=config.device, return_transf_layer=config.return_transf_layer,
                      output_token=config.output_token, pre_training=False)
    # Loading safetensors directly avoids remote code and verifies every tensor
    # name/shape rather than accepting a partially initialized feature extractor.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        network = MantisV2(**parameters)
    network.load_state_dict(load_file(str(Path(checkpoint_dir) / "model.safetensors")), strict=True)
    network.requires_grad_(False)
    network.eval()
    if any(parameter.requires_grad for parameter in network.parameters()):
        raise RuntimeError("The encoder must be completely frozen.")
    return network, checkpoint


def _encode_batches(X, network, config: EncoderConfig, *, progress=True):
    import torch
    from torch.nn.functional import interpolate

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or min(X.shape) < 1 or X.shape[2] < 2 or not np.isfinite(X).all():
        raise ValueError("Expected finite nonempty (participants, channels, timepoints>=2).")
    count, channels, timepoints = X.shape
    flat = np.ascontiguousarray(X).reshape(count * channels, 1, timepoints)
    embeddings = np.empty((len(flat), 512), dtype=np.float32)
    with torch.inference_mode(), tqdm(
        total=len(flat), desc="Initial MantisV2 features", unit="channel",
        dynamic_ncols=True, disable=not progress,
    ) as bar:
        for start in range(0, len(flat), config.batch_size):
            stop = min(start + config.batch_size, len(flat))
            values = torch.from_numpy(flat[start:stop]).to(config.device)
            values = interpolate(values, size=config.sequence_length, mode="linear", align_corners=False)
            encoded = network(values).cpu().numpy()
            if encoded.shape != (stop - start, 512) or not np.isfinite(encoded).all():
                raise ValueError("Expected 512 finite pretrained features per channel.")
            embeddings[start:stop] = encoded
            bar.update(stop - start)
    return embeddings.reshape(count, channels * 512)


def extract_features(X, config: EncoderConfig, checkpoint_dir: Path, *, progress=True):
    """Return raw float32 frozen embeddings and their auditable provenance."""
    started = time.perf_counter()
    network, checkpoint = load_encoder(config, checkpoint_dir)
    Z = _encode_batches(X, network, config, progress=progress)
    provenance = {
        "encoder": asdict(config),
        "checkpoint": checkpoint,
        "package_version": version("mantis-tsfm"),
        "parameters_full_checkpoint": sum(p.numel() for p in network.parameters()),
        "parameters_trainable": sum(p.numel() for p in network.parameters() if p.requires_grad),
        "mode": "eval with torch.inference_mode; no PADS fitting or fine-tuning",
        "pretraining_reported_by_authors": "CauKer2M synthetic time series only",
        "input_shape": list(np.shape(X)),
        "feature_shape": list(Z.shape),
        "feature_dtype": str(Z.dtype),
        "channel_processing": "Each original channel independently encoded; original channel order concatenated.",
        "time_adapter": "torch linear interpolation 976->512; align_corners=False; no antialias filter.",
        "normalization": "No external input z-score; pretrained internal signal scaling and raw patch mean/std encoders retained.",
        "cohort_fitted_parameters": "None in extraction. Embedding StandardScaler fitted later within each training fold.",
        "representation": "Transformer layer index 2; concatenate 256-dimensional CLS and mean patch tokens per channel.",
        "cross_channel_attention": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return Z, provenance
