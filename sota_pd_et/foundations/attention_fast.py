"""Attention shortcuts for the pinned embedding-only model contracts.

No installed library or parameter/state-dict layout is changed. T5 retains
its native attention, learned relative bias, observed mask, and dropout. The
Chronos path packs independent, equally sized task groups into the batch axis
instead of evaluating a larger block-diagonal attention matrix. Its native
input preparation, time attention, residuals, and context-only readout remain.
"""

from functools import lru_cache
from importlib.metadata import version
from types import MethodType

import torch


@lru_cache(maxsize=None)
def _version_is(package, expected):
    return version(package) == expected


def _t5_cached_position_bias(self, query_length, key_length, device=None, cache_position=None):
    if device is None:
        device = self.relative_attention_bias.weight.device
    distance, buckets = self.relative_attention_max_distance, self.relative_attention_num_buckets
    cache_key = (buckets, distance, not self.is_decoder, torch.device(device))
    cached = getattr(self, "_comparison_position_lookup", None)
    if cached is None or cached[0] != cache_key:
        # Outside this finite interval native T5 saturates to its last bucket.
        # Keep only integer bucket IDs; learned bias values are NEVER cached.
        # Use the same device arithmetic as native T5 when a bucket boundary
        # involves logarithms; do not assume CPU/CUDA/MPS log rounding agrees.
        offsets = torch.arange(-distance, distance + 1, dtype=torch.long, device=device)
        lookup = self._relative_position_bucket(
            offsets, bidirectional=not self.is_decoder, num_buckets=buckets, max_distance=distance,
        )
        cached = (cache_key, lookup)
        self._comparison_position_lookup = cached
    context = (torch.arange(query_length, dtype=torch.long, device=device)
               if cache_position is None else cache_position.to(device))
    memory = torch.arange(key_length, dtype=torch.long, device=device)
    relative = memory[None, :] - context[:, None]
    indices = cached[1][relative.clamp(-distance, distance) + distance]
    return self.relative_attention_bias(indices).permute(2, 0, 1).unsqueeze(0)


def install_moment_position_cache(backbone):
    """Cache only deterministic bucket IDs for the audited native T5 encoder."""
    if not _version_is("transformers", "4.57.6"):
        return 0
    from transformers.models.t5.modeling_t5 import T5Attention
    count = 0
    for module in backbone.encoder.modules():
        if (type(module) is T5Attention and not module.is_decoder and module.has_relative_attention_bias
                and module.relative_attention_max_distance > module.relative_attention_num_buckets // 2):
            if not hasattr(module, "_comparison_native_compute_bias"):
                module._comparison_native_compute_bias = module.compute_bias
                module.compute_bias = MethodType(_t5_cached_position_bias, module)
            count += 1
    return count


def chronos_native_group_features(model, values):
    groups, variates, length = values.shape
    flat = values.reshape(groups * variates, length)
    ids = torch.arange(groups, device=flat.device).repeat_interleave(variates)
    output, _, _, n_context = model.encode(context=flat, group_ids=ids, num_output_patches=1)
    return output.last_hidden_state[:, :n_context].mean(1).reshape(groups, variates, -1)


def chronos_packed_group_features(model, values, *, complete_context):
    """Pack complete independent tasks; incomplete contexts use native encode.

    ``complete_context`` is checked once by the adapter before chunking. For
    finite complete series every context patch contains an observation, even
    when the patch helper adds left padding. This excludes entirely masked
    groups, whose finite-minimum-mask behavior must stay with native encode.
    """
    from chronos.chronos2 import Chronos2Model
    if (not complete_context or type(model) is not Chronos2Model
            or not _version_is("chronos-forecasting", "2.3.2")):
        return chronos_native_group_features(model, values)
    groups, variates, length = values.shape
    flat = values.reshape(groups * variates, length)
    # Same preparation as Chronos2Model.encode, with no covariates/targets and
    # exactly one native future token, as required by this embedding adapter.
    patched, observed, loc_scale = model._prepare_patched_context(context=flat)
    n_context = observed.shape[-1]
    embedded = model.input_patch_embedding(patched)
    if model.chronos_config.use_reg_token:
        ids = torch.full((len(flat), 1), model.config.reg_token_id, device=flat.device)
        embedded = torch.cat((embedded, model.shared(ids)), dim=-2)
        observed = torch.cat((observed.to(model.dtype), torch.ones_like(ids).to(model.dtype)), dim=-1)
    future, _ = model._prepare_patched_future(
        future_covariates=None, future_covariates_mask=None, loc_scale=loc_scale,
        num_output_patches=1, batch_size=len(flat),
    )
    embedded = torch.cat((embedded, model.input_patch_embedding(future)), dim=-2)
    observed = torch.cat((observed, torch.ones(len(flat), 1, dtype=model.dtype, device=flat.device)), dim=-1)
    stack = model.encoder
    seq_length = embedded.shape[1]
    positions = torch.arange(seq_length, dtype=torch.long, device=flat.device).unsqueeze(0)
    time_mask = stack._expand_and_invert_time_attention_mask(observed, embedded.dtype)
    # Only keys from the current task are present. No cross-task logits or
    # quadratic block-diagonal masks are allocated.
    group_observed = observed.reshape(groups, variates, seq_length).transpose(1, 2).reshape(-1, variates)
    group_mask = (1.0 - group_observed[:, None, None, :]) * torch.finfo(observed.dtype).min
    hidden = stack.dropout(embedded)
    for block in stack.block:
        hidden = block.layer[0](hidden, position_ids=positions, attention_mask=time_mask)[0]
        group_layer = block.layer[1]
        packed = hidden.reshape(groups, variates, seq_length, -1).transpose(1, 2).reshape(groups * seq_length, variates, -1)
        normed = group_layer.layer_norm(packed)
        attended = group_layer.self_attention(normed, mask=group_mask)[0]
        packed = packed + group_layer.dropout(attended)
        hidden = packed.reshape(groups, seq_length, variates, -1).transpose(1, 2).reshape(len(flat), seq_length, -1)
        hidden = block.layer[2](hidden)
    hidden = stack.dropout(stack.final_layer_norm(hidden))
    return hidden[:, :n_context].mean(1).reshape(groups, variates, -1)
