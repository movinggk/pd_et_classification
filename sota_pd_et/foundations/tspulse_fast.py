"""TSPulse's classification-only hidden state without reconstruction targets.

The pinned upstream backbone computes two FFTs when time/FFT masking is
consistent. Its second FFT and separate ground-truth scaler only populate
reconstruction outputs; neither contributes to ``last_hidden_state``. This
path retains the first FFT, including the trainable RevIN affine transform,
and calls the original model's modules without replacing any parameters.
"""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch


@lru_cache(maxsize=1)
def _verified_runtime():
    try:
        return version("granite-tsfm") == "0.3.12"
    except PackageNotFoundError:
        return False


def supports_tspulse_hidden_fast_path(backbone):
    """Limit the shortcut to the verified single-channel checkpoint layout."""
    if not _verified_runtime():
        return False
    from tsfm_public.models.tspulse.modeling_tspulse import TSPulseModel

    if type(backbone) is not TSPulseModel:
        return False
    config = backbone.config
    return (
        config.fuse_fft is True
        and backbone.fft_time_consistent_masking is True
        and config.fft_time_consistent_masking is True
        and config.fft_applied_on in ("scaled_ts", "raw_ts")
        and config.context_length == 512
        and config.num_input_channels == 1
        and config.patch_length == 16
        and config.patch_stride == 16
        and config.mode == "common_channel"
        and config.channel_virtual_expand_scale in (None, 1)
        and config.scaling in ("std", "revin")
    )


def tspulse_last_hidden(backbone, past_values):
    """Return exactly the unmasked backbone hidden state used by classification.

    Unverified package versions, model layouts, and input shapes use the
    upstream implementation. This function never enables pretraining masks,
    returns reconstruction targets, or changes the original state dictionary.
    """
    if (not supports_tspulse_hidden_fast_path(backbone)
            or past_values.ndim != 3 or past_values.shape[1:] != (512, 1)):
        return backbone(past_values=past_values, enable_masking=False,
                        return_dict=True).last_hidden_state

    from tsfm_public.models.tspulse import modeling_tspulse

    config = backbone.config
    scaled, _, _ = backbone.scaler(past_values, torch.ones_like(past_values))
    time_encoding = backbone.time_encoding(backbone.patching(scaled))
    fft_input = scaled if config.fft_applied_on == "scaled_ts" else past_values
    fft_values, _, _, _, _ = modeling_tspulse.get_fft(
        fft_input, epsilon=config.minimum_scale,
        fft_remove_component=config.fft_remove_component,
        fft_prob_mode=config.fft_prob_mode, fft_prob_length=config.fft_prob_length,
    )
    fft_encoding = backbone.fft_encoding(backbone.patching(fft_values))
    encoding = backbone.add_tokens(torch.cat((time_encoding, fft_encoding), dim=-2))
    batch, channels, patches, width = encoding.shape
    encoding = backbone.base_norm(encoding.reshape(batch, channels, patches * width))
    hidden, _ = backbone.encoder_block(encoding.reshape(batch, channels, patches, width))
    return hidden
