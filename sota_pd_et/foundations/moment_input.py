"""MOMENT input sizing and observed masks for the audited adapter.

The author's short-classification example left-pads and masks observations;
the PTB-XL example uses endpoint-preserving linear interpolation. This helper
implements those sizing mechanics without dataset-fitted input statistics.
The model's native per-series RevIN remains the normalization step. It does
not additionally apply PTB-XL's dataset-specific external z-score operation.

Reference sources (pinned package commit):
https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/data/classification_dataset.py
https://github.com/moment-timeseries-foundation-model/moment/blob/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/data/ptbxl_classification_dataset.py
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def prepare_moment_input(values, target_length=512, patch_length=8):
    """Return ``(B,1,target_length)`` values and ``(B,target_length)`` mask.

    ``values`` contains complete, finite channel series, one per row. The
    caller validates finiteness before batching. Short series retain their
    sample spacing through zero left-padding; long series are resampled over
    their entire endpoint-inclusive interval. Native masked mean pooling
    requires at least one completely observed patch, so shorter inputs are
    rejected instead of returning a NaN feature vector.
    """
    if (not isinstance(values, torch.Tensor) or values.ndim != 2
            or min(values.shape) < 1 or not values.is_floating_point()):
        raise ValueError("MOMENT needs a nonempty floating-point (channel_series,time) tensor.")
    if (not isinstance(target_length, int) or isinstance(target_length, bool)
            or not isinstance(patch_length, int) or isinstance(patch_length, bool)
            or min(target_length, patch_length) < 1 or target_length % patch_length):
        raise ValueError("MOMENT target length must be a positive multiple of its positive integer patch length.")
    observed_length = values.shape[-1]
    if observed_length < patch_length:
        raise ValueError(f"MOMENT masked mean pooling requires at least one fully observed {patch_length}-sample patch; "
                         f"received {observed_length} samples.")

    inputs = values[:, None, :]
    mask = torch.ones((len(values), target_length), device=values.device, dtype=torch.long)
    if observed_length < target_length:
        padding = target_length - observed_length
        inputs = F.pad(inputs, (padding, 0), mode="constant", value=0.)
        mask[:, :padding] = 0
    elif observed_length > target_length:
        inputs = F.interpolate(inputs, size=target_length, mode="linear", align_corners=True)
    return inputs, mask
