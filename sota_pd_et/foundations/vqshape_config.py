"""Restore the pinned VQShape release without guessing architecture settings.

Attention head counts are not encoded by tensor shapes. Strict state-dict
loading alone therefore cannot detect a guessed head count that changes the
feature function. All saved architecture settings must be present explicitly.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
import math
from numbers import Integral, Real


_ARCHITECTURE_KEYS = {
    "len_input": "normalize_length",
    "dim_embedding": "dim_embedding",
    "patch_size": "patch_size",
    "num_patch": "num_patch",
    "num_token": "num_token",
    "num_enc_head": "num_transformer_enc_heads",
    "num_enc_layer": "num_transformer_enc_layers",
    "num_tokenizer_head": "num_tokenizer_heads",
    "num_tokenizer_layer": "num_tokenizer_layers",
    "num_dec_head": "num_transformer_dec_heads",
    "num_dec_layer": "num_transformer_dec_layers",
    "len_s": "len_s",
    "s_smooth_factor": "s_smooth_factor",
    "num_code": "num_code",
    "dim_code": "dim_code",
    "codebook_type": "codebook_type",
}


def _mapping(value, location):
    if isinstance(value, Namespace):
        value = vars(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"VQShape {location} must be a mapping or argparse.Namespace.")
    return dict(value)


def vqshape_kwargs_from_hyperparameters(hyperparameters):
    """Return complete constructor kwargs for the selected 512-point release.

    Both direct Lightning hyperparameter dictionaries and the older nested
    ``args`` Namespace layout are accepted. Only pretraining-loss settings
    have defaults: they do not select the shape-token inference function.
    The caller must still load the complete checkpoint state strictly.
    """
    parameters = _mapping(hyperparameters, "hyper_parameters")
    # Retain support for a nested Lightning args container, while bounding
    # malformed/self-referential layouts instead of recursing indefinitely.
    for _ in range(8):
        if "args" not in parameters:
            break
        parameters = _mapping(parameters["args"], "hyper_parameters.args")
    else:
        raise ValueError("VQShape hyper_parameters.args nesting is too deep.")

    missing = sorted(set(_ARCHITECTURE_KEYS.values()) - parameters.keys())
    if missing:
        raise ValueError("VQShape checkpoint is missing required architecture settings: " + ", ".join(missing))
    kwargs = {}
    for target, source in _ARCHITECTURE_KEYS.items():
        value = parameters[source]
        if target == "codebook_type":
            if value != "standard":
                raise ValueError("The pinned VQShape adapter requires codebook_type='standard'.")
        else:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"VQShape architecture setting {source} must be a positive integer.")
            value = int(value)
        kwargs[target] = value

    if kwargs["len_input"] != 512:
        raise ValueError("The pinned VQShape adapter requires normalize_length=512.")
    if kwargs["patch_size"] * kwargs["num_patch"] != kwargs["len_input"]:
        raise ValueError("VQShape patch_size*num_patch must equal normalize_length for this release.")
    for field in ("num_enc_head", "num_tokenizer_head", "num_dec_head"):
        if kwargs["dim_embedding"] % kwargs[field]:
            raise ValueError(f"VQShape dim_embedding must be divisible by {_ARCHITECTURE_KEYS[field]}.")

    optional = {
        "lambda_commit": ("lambda_vq_commit", 1.),
        "lambda_entropy": ("lambda_vq_entropy", 0.),
        "entropy_gamma": ("entropy_gamma", 1.),
        "mask_ratio": ("mask_ratio", .25),
    }
    for target, (source, default) in optional.items():
        value = parameters.get(source, default)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"VQShape pretraining setting {source} must be finite and numeric.")
        kwargs[target] = float(value)
    if not 0 <= kwargs["mask_ratio"] <= 1:
        raise ValueError("VQShape mask_ratio must be in [0,1].")
    return kwargs
