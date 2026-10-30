"""Experimental CPU Qwen3.5 split-QKV convolution and gated-delta core.

Inputs are already projected. No checkpoint loading, output projection,
transaction identity, production state store or GPU execution is provided.
"""
import torch
import torch.nn.functional as F
from .qwen_gdn_cpu import recurrent_delta, GdnInputError


class ProjectedInputError(ValueError):
    pass


def projected_core(qkv, a, b, conv_weight, a_log, dt_bias, conv_history,
                   recurrent_state, *, key_heads, value_heads, key_dim, value_dim):
    sizes = (key_heads, value_heads, key_dim, value_dim)
    if any(type(x) is not int or x <= 0 for x in sizes) or value_heads % key_heads:
        raise ProjectedInputError('invalid head layout')
    tensors = (qkv, a, b, conv_weight, a_log, dt_bias, conv_history)
    if any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu'
           or t.dtype != torch.float32 for t in tensors):
        raise ProjectedInputError('CPU float32 tensors required')
    if qkv.ndim != 3:
        raise ProjectedInputError('qkv must be B,T,C')
    batch, length, channels = qkv.shape
    ksize = key_heads * key_dim
    vsize = value_heads * value_dim
    if (batch <= 0 or length <= 0 or length > 4096 or channels != 2 * ksize + vsize
            or a.shape != (batch, length, value_heads) or b.shape != a.shape
            or conv_weight.ndim != 2 or conv_weight.shape[0] != channels
            or not 2 <= conv_weight.shape[1] <= 64
            or conv_history.shape != (batch, channels, conv_weight.shape[1] - 1)
            or a_log.shape != (value_heads,) or dt_bias.shape != (value_heads,)
            or sum(t.numel() for t in tensors) > 16_000_000):
        raise ProjectedInputError('invalid projected core shapes or size')
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ProjectedInputError('nonfinite projected core input')
    with torch.inference_mode():
        full = torch.cat((conv_history, qkv.transpose(1, 2)), dim=-1)
        next_history = full[:, :, -(conv_weight.shape[1] - 1):].clone()
        conv = F.silu(F.conv1d(full, conv_weight.unsqueeze(1), groups=channels)).transpose(1, 2)
        query, key, value = conv.split((ksize, ksize, vsize), dim=-1)
        query = query.reshape(batch, length, key_heads, key_dim).repeat_interleave(value_heads // key_heads, dim=2)
        key = key.reshape(batch, length, key_heads, key_dim).repeat_interleave(value_heads // key_heads, dim=2)
        value = value.reshape(batch, length, value_heads, value_dim)
        decay_log = -a_log.exp() * F.softplus(a + dt_bias)
        beta = b.sigmoid()
        try:
            output, state = recurrent_delta(query, key, value, decay_log, beta, recurrent_state, normalize=True)
        except GdnInputError as exc:
            raise ProjectedInputError(str(exc)) from exc
    return output, next_history, state
