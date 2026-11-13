"""Feature-only paths that omit unused pretrained prediction computations.

These functions retain the original backbone modules and state-dict layout.
They use the same feature equations and straight-through quantization as the
pinned upstream implementations. They do not modify installed dependencies.

VQShape's skipped reconstruction decoder consumes dropout randomness in train
mode in the original implementation. Removing unused decoder work therefore
changes RNG consumption between successive channel chunks. The stochastic
feature function is unchanged, but a full fixed-seed training trajectory need
not be bitwise identical to the unoptimized implementation.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def vqshape_token_features(model, values):
    """Return flattened published shape tokens without decoding predictions.

    ``values`` is a float tensor of shape (channel_series, 512). Token order is
    code vector, start position, length, location, and scale, exactly as in
    the upstream ``tokenize`` method. The standard codebook uses its original
    hard nearest-vector assignment and straight-through gradient estimator.
    """
    return vqshape_normalized_features(model, prepare_vqshape_values(values))


def prepare_vqshape_values(values):
    """Native per-series normalization; no fitted or trainable statistics."""
    mean = values.mean(dim=-1, keepdim=True)
    std = (values.var(dim=-1, keepdim=True) + 1e-5).sqrt()
    return (values - mean) / std


def vqshape_normalized_features(model, values):
    encoded = model.encoder(values)
    shape_hidden = model.tokenizer(encoded, None)
    codes, position_length, location, scale = model.attr_decoder(shape_hidden)
    position = position_length[..., [0]] * (1 - model.min_shape_len)
    length = position_length[..., [1]] * (1 - position) + model.min_shape_len

    if model.codebook_type == "standard":
        # Assignment has no useful derivative. The upstream representation
        # explicitly routes classification gradients through `codes` instead
        # of through the argmin or the selected codebook embedding.
        with torch.no_grad():
            flat = codes.reshape(-1, codes.shape[-1])
            weight = model.codebook.embedding.weight
            distances = (torch.sum(flat ** 2, dim=1, keepdim=True)
                         + torch.sum(weight ** 2, dim=1)
                         - 2 * torch.matmul(flat, weight.t()))
            selected = model.codebook.embedding(torch.argmin(distances, dim=1)).reshape_as(codes)
        quantized = codes + (selected - codes).detach()
    else:
        # The shipped checkpoint uses `standard`. Preserve upstream handling
        # if a caller deliberately supplies its optional vqtorch alternative.
        quantized, _ = model.codebook(codes)

    return torch.cat([quantized, position, length, location, scale], dim=-1).flatten(1)


def tirex_hidden_features(model, values):
    """Return the original layer-pooled features without the forecast head.

    The stack/reduction layout is preserved to avoid changing the floating
    point reduction order. Final output normalization and quantile prediction
    are absent because none contributes to the returned intermediate layers.
    """
    inputs, padding = model._prepare_context_for_embedding(values, None)
    hidden = model.input_patch_embedding(inputs)
    all_hidden = []
    for block in model.blocks:
        hidden = block(hidden)
        all_hidden.append(hidden)
    pooled = torch.stack(all_hidden, dim=-2)[:, padding:].mean(1)
    return F.normalize(pooled, dim=-1).flatten(1)


def _tivit_processor_key(model):
    """Return the supported processor settings, or None for upstream fallback."""
    from transformers import CLIPImageProcessor
    from torchvision.transforms import ToPILImage

    processor = model.processor
    if type(processor) is not CLIPImageProcessor:
        return None
    if not isinstance(model.to_pil, ToPILImage) or model.to_pil.mode is not None:
        return None
    if (processor.do_resize is not True or processor.size != {"shortest_edge": 224}
            or processor.resample != 3
            or processor.do_center_crop is not True
            or processor.crop_size != {"height": 224, "width": 224}
            or processor.do_rescale is not True or processor.rescale_factor != 1 / 255
            or processor.do_normalize is not True or processor.do_convert_rgb is not True
            or model.layer_idx == 0):
        return None
    try:
        import math
        mean, std = tuple(processor.image_mean), tuple(processor.image_std)
        if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(v) for v in (*mean, *std)):
            return None
        if not all(v > 0 for v in std):
            return None
    except (TypeError, ValueError):
        return None
    return (processor.rescale_factor, mean, std)


def _tivit_grayscale_pixel_values(model, images, key):
    """Replace PIL/NumPy processing of official repeated-grayscale TiViT images.

    Input is the original TiViT (B,3,224,224) repeated-grayscale image tensor.
    Integer quantization matches ToPILImage. A 3x256 table is computed using the
    processor's own CPU transforms, preserving its float64 rescale followed by
    float32 normalization exactly; only a tiny table moves to the image device.
    The derived cache is an ordinary attribute and adds no state-dict entries.
    """
    quantized = (images[:, 0] * 255).to(torch.uint8)
    return _tivit_pixels_from_quantized(model, quantized, key)


def _tivit_pixels_from_quantized(model, quantized, key):
    cached = getattr(model, "_comparison_pixel_lookup", None)
    cache_key = (key, quantized.device)
    if cached is None or cached[0] != cache_key:
        import numpy as np
        ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=-1)
        processor = model.processor
        scaled = processor.rescale(ramp, scale=processor.rescale_factor,
                                   input_data_format="channels_last")
        normalized = processor.normalize(scaled, mean=processor.image_mean,
                                         std=processor.image_std,
                                         input_data_format="channels_last")
        lookup = torch.from_numpy(normalized[0].T.copy()).to(quantized.device)
        cached = (cache_key, lookup)
        model._comparison_pixel_lookup = cached
    return cached[1][:, quantized.long()].permute(1, 0, 2, 3).contiguous()


def tivit_tensor_features(model, values, *, patch_size):
    """Preserve TiViT's 8-bit image representation without host round trips.

    ``values`` has shape (channel_series,time). The official signal-to-image
    transformation is retained. For the pinned 224-pixel CLIP processor, PIL
    resizing/cropping are identities and quantization/rescale/normalization
    are replaced by equivalent tensor indexing. Unsupported processors use
    the original vision processing. The required local signal patch size
    implements the author's length-dependent sizing without mutating the model,
    including during activation-checkpoint recomputation.
    """
    prepared, is_quantized = prepare_tivit_inputs(model, values, patch_size=patch_size)
    return tivit_prepared_features(model, prepared, is_quantized=is_quantized)


def prepare_tivit_inputs(model, values, *, patch_size):
    """Build signal images once; retain only one uint8 channel for replay.

    A supported image needs 1/12 of the storage of three FP32 CLIP channels.
    Checkpoint replay rebuilds pixels by a small table lookup, without
    repeating robust scaling, rasterization, interpolation, or quantization.
    """
    if (not isinstance(patch_size, int) or isinstance(patch_size, bool)
            or not 1 <= patch_size <= values.shape[-1]):
        raise ValueError("TiViT signal patch size must be an integer between 1 and the input length.")
    key = _tivit_processor_key(model)
    # The upstream PIL conversion detaches signal inputs. Avoid constructing
    # its discarded input-gradient graph while preserving backbone gradients.
    with torch.no_grad():
        images = model.ts2image_transformation(values[:, :, None],
                                               patch_size=patch_size,
                                               stride=model.stride)
    if key is None or tuple(images.shape[1:]) != (3, 224, 224):
        # Reuse the official vision processor for the already transformed
        # image. Calling model.forward here would silently use its stored
        # patch size instead of this call's explicit override.
        return images, False
    with torch.no_grad():
        quantized = (images[:, 0] * 255).to(torch.uint8)
    return quantized, True


def tivit_prepared_features(model, prepared, *, is_quantized):
    if is_quantized:
        with torch.no_grad():
            pixels = _tivit_pixels_from_quantized(model, prepared, _tivit_processor_key(model))
        outputs = model.vit(pixel_values=pixels, output_hidden_states=(model.layer_idx is None))
        hidden = outputs.last_hidden_state if model.layer_idx else torch.stack(outputs.hidden_states, dim=-1)
    else:
        hidden = model.forward_vit(prepared)
    return model.aggregate_hidden_representations(hidden, aggregation=model.aggregation)
