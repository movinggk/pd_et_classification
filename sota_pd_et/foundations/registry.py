"""Fixed checkpoint identities and representation settings.

Model revisions were read from the publishers' Hub API on 2026-11-11.
All input channels retain their positions in the participant representation.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    revision: str
    representation: str
    source_url: str


SPECS = {
    "tspulse": ModelSpec(
        "tspulse", "ibm-granite/granite-timeseries-tspulse-r1",
        "66cfb5464ed5e5a5cd7f5df1339af4e51461d955",
        "Backbone time-patch mean, FFT-patch mean, and mean register token; per channel.",
        "https://github.com/ibm-granite/granite-tsfm",
    ),
    "utica": ModelSpec(
        "utica", "fegounna/Utica", "3cff4f954191b5bf9839b7a41117e2b24e7693ab",
        "MantisV1 intermediate layer index 2, combined CLS/mean patch tokens; per channel.",
        "https://github.com/fegounna/Utica",
    ),
    "tirex": ModelSpec(
        "tirex", "NX-AI/TiRex", "63c740922493f5fbe60b277609ec62babfba2762",
        "Mean context patches per xLSTM layer, per-layer L2, author layer/channel ordering and final participant LayerNorm; no optional augmentation.",
        "https://github.com/NX-AI/tirex",
    ),
    "moment": ModelSpec(
        "moment", "AutonLab/MOMENT-1-base", "5e44b0ea26376a176360f87831124e018f876d96",
        "MOMENT embedding mode, native masked patch mean and channel concatenation; masked left-padding or endpoint-preserving resize.",
        "https://github.com/moment-timeseries-foundation-model/moment",
    ),
    "chronos2": ModelSpec(
        "chronos2", "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "Mean context patch tokens per variate; group attention within each task's 12 axes only.",
        "https://github.com/amazon-science/chronos-forecasting",
    ),
    "tivit": ModelSpec(
        "tivit", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
        "Official TiViT image transformation; CLIP ViT-H first 14 blocks, mean tokens per channel, participant L2 and length-dependent sqrt signal patches.",
        "https://github.com/ExplainableML/TiViT",
    ),
    "vqshape": ModelSpec(
        "vqshape", "YunshiWen/VQShape:uea_dim256_codebook512", "v0.1.0-cls",
        "Flattened shape tokens (straight-through code vectors and attributes); per channel.",
        "https://github.com/YunshiWen/VQShape",
    ),
    "mantis_v2": ModelSpec(
        "mantis_v2", "paris-noah/MantisV2", "99fe0f548960e272fbfa4b82fd9b5b5956779dfd",
        "MantisV2 intermediate layer index 2, combined CLS/mean patch tokens; per channel.",
        "https://github.com/vfeofanov/mantis",
    ),
}

DEFAULT_MODELS = tuple(name for name in SPECS if name != "mantis_v2")


def specifications():
    return {name: asdict(spec) for name, spec in SPECS.items()}
