"""Validated per-model execution settings, independent of training selection."""

import argparse

from .registry import SPECS


def _pair(value):
    model, separator, setting = value.partition("=")
    if not separator or model not in SPECS:
        raise argparse.ArgumentTypeError(f"Expected MODEL=VALUE with MODEL in {', '.join(SPECS)}.")
    return model, setting


def channel_override(value):
    model, setting = _pair(value)
    try:
        count = int(setting)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Channel batch size must be a positive integer.") from error
    if count < 1:
        raise argparse.ArgumentTypeError("Channel batch size must be a positive integer.")
    return model, count


def checkpoint_override(value):
    model, setting = _pair(value)
    if setting not in ("on", "off"):
        raise argparse.ArgumentTypeError("Checkpointing must be MODEL=on or MODEL=off.")
    return model, setting


def override_mapping(pairs):
    result = {}
    for model, setting in pairs or ():
        if model in result:
            raise ValueError(f"Duplicate runtime override for {model}.")
        result[model] = setting
    return result


def model_runtime_settings(args, model):
    legacy_channels = args.tirex_channel_batch_size if model == "tirex" else None
    legacy_checkpointing = args.tirex_gradient_checkpointing if model == "tirex" else None
    channels = args.model_channel_batch_size.get(
        model, args.channel_batch_size if legacy_channels is None else legacy_channels,
    )
    setting = args.model_gradient_checkpointing.get(model)
    if setting is None:
        setting = legacy_checkpointing
    checkpointing = args.gradient_checkpointing if setting is None else setting == "on"
    return channels, checkpointing
