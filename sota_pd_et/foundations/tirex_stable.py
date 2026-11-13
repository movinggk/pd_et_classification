"""Overflow-safe derivatives for TiRex's published PyTorch sLSTM recurrence.

The recurrence follows NXAI's TiRex sLSTM implementation and its xLSTM
equations (https://github.com/NX-AI/tirex). TiRex remains subject to the NXAI
Community License. No installed library files or pretrained tensors change.

min(exp(a), 1) == exp(min(a, 0)). Clipping before exp prevents an intermediate
infinity and its undefined 0*inf backward product. The original BF16 recurrent
matmul/state casts, FP32 pointwise equations, gate layout and tie derivatives
are retained. Only each loaded TiRex cell's implementation callback changes.

The recurrent product preserves the native forward calculation and uses an
analytic headwise backward to avoid saving a broadcast copy of the recurrent
weights for every series at every timestep. BF16 gradient reductions can round
differently from the broadcasted derivative, as with changing channel chunks.
"""

import torch
from torch.nn import functional as F


class _RecurrentProduct(torch.autograd.Function):
    """Native forward, headwise backward without saved per-series R copies."""

    @staticmethod
    def forward(ctx, hidden, recurrent):
        ctx.save_for_backward(hidden, recurrent)
        # A custom Function's forward does not retain the internal broadcast
        # matmul graph. Its exact native forward arithmetic is still used.
        return hidden.unsqueeze(-2).matmul(recurrent.unsqueeze(0)).squeeze(-2)

    @staticmethod
    def backward(ctx, gradient):
        hidden, recurrent = ctx.saved_tensors
        per_head_gradient = gradient.transpose(0, 1)
        hidden_gradient = recurrent_gradient = None
        if ctx.needs_input_grad[0]:
            hidden_gradient = torch.bmm(per_head_gradient, recurrent.transpose(1, 2)).transpose(0, 1)
        if ctx.needs_input_grad[1]:
            recurrent_gradient = torch.bmm(hidden.transpose(0, 1).transpose(1, 2), per_head_gradient)
        return hidden_gradient, recurrent_gradient


def recurrent_product(hidden, recurrent):
    if torch.is_grad_enabled() and (hidden.requires_grad or recurrent.requires_grad):
        return _RecurrentProduct.apply(hidden, recurrent)
    return hidden.unsqueeze(-2).matmul(recurrent.unsqueeze(0)).squeeze(-2)


def stable_pointwise(Wx, Ry, bias, states):
    _, cell, normalizer, stabilizer = states
    raw = Wx + Ry + bias
    input_raw, forget_raw, update_raw, output_raw = raw.reshape(raw.shape[0], 4, -1).unbind(1)
    forget_log = stabilizer + F.logsigmoid(torch.clamp(forget_raw, max=15))
    next_stabilizer = torch.where(torch.all(normalizer == 0), input_raw,
                                  torch.maximum(input_raw, forget_log))
    zero = torch.zeros_like(input_raw)
    input_gate = torch.exp(torch.minimum(input_raw - next_stabilizer, zero))
    forget_gate = torch.exp(torch.minimum(forget_log - next_stabilizer, zero))
    next_cell = forget_gate * cell + input_gate * torch.tanh(update_raw)
    next_normalizer = forget_gate * normalizer + input_gate
    hidden = torch.sigmoid(output_raw) * next_cell / next_normalizer
    return [hidden, next_cell, next_normalizer, next_stabilizer]


def stable_recurrence(inputs, states, recurrent, bias):
    sequence, batch, _ = inputs.shape
    heads = recurrent.shape[0]
    states = states.to(recurrent.dtype).unbind(0)
    outputs = []
    for step in range(sequence):
        # Preserve native forward rounding. The analytic headwise backward
        # avoids retaining one full R copy per series at every timestep.
        projected = (recurrent_product(states[0].reshape(batch, heads, -1), recurrent)
                     .reshape(batch, heads, 4, -1)
                     .transpose(1, 2).reshape(batch, -1))
        updated = stable_pointwise(inputs[step].float(), projected.float(), bias.float(),
                                   [state.float() for state in states])
        states = [state.to(recurrent.dtype) for state in updated]
        outputs.append(states[0])
    return torch.stack(outputs), torch.stack(states)


def install_stable_recurrence(model):
    from tirex.models.slstm.cell import sLSTMCell
    count = 0
    for block in model.blocks:
        cell = block.slstm_layer.slstm_cell
        if not isinstance(cell, sLSTMCell) or cell.backend != "torch":
            raise ValueError("The TiRex stability adapter requires the supported PyTorch sLSTM cells.")
        cell._impl_forward_torch = stable_recurrence
        count += 1
    return count
