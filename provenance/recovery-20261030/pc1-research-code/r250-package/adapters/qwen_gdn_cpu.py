"""Bounded CPU reference seam for Qwen gated-delta recurrence.

Reuses Transformers' PyTorch recurrence; caller owns transaction and state
identity. This is not a full Qwen attention module or a CUDA implementation.
"""
import torch
from transformers.models.qwen3_next.modeling_qwen3_next import torch_recurrent_gated_delta_rule


class GdnInputError(ValueError):
    pass


def recurrent_delta(query, key, value, decay_log, beta, initial_state=None, *, normalize=False):
    tensors = (query, key, value, decay_log, beta)
    if (type(normalize) is not bool or any(not isinstance(t, torch.Tensor)
            or t.device.type != 'cpu' or not t.is_floating_point() for t in tensors)):
        raise GdnInputError('CPU floating-point tensors required')
    if (query.ndim != 4 or key.shape != query.shape or value.ndim != 4
            or query.shape[:3] != value.shape[:3] or decay_log.shape != query.shape[:3]
            or beta.shape != query.shape[:3] or min(query.shape) <= 0
            or value.shape[-1] <= 0 or query.dtype != key.dtype or query.dtype != value.dtype):
        raise GdnInputError('invalid recurrent dimensions or dtypes')
    batch, tokens, heads, kdim = query.shape
    vdim = value.shape[-1]
    if (tokens > 4096 or batch * heads * kdim * vdim > 16_000_000
            or sum(t.numel() for t in tensors) > 16_000_000):
        raise GdnInputError('reference limits exceeded')
    if any(not torch.isfinite(t).all() for t in tensors):
        raise GdnInputError('nonfinite input')
    if (decay_log > 0).any() or (beta < 0).any() or (beta > 1).any():
        raise GdnInputError('decay or beta outside Qwen domain')
    if initial_state is not None:
        if (not isinstance(initial_state, torch.Tensor) or initial_state.device.type != 'cpu'
                or initial_state.dtype != torch.float32 or initial_state.shape != (batch, heads, kdim, vdim)
                or not torch.isfinite(initial_state).all()):
            raise GdnInputError('invalid recurrent state')
        initial_state = initial_state.clone()
    with torch.inference_mode():
        output, state = torch_recurrent_gated_delta_rule(
            query, key, value, decay_log, beta, initial_state,
            output_final_state=True, use_qk_l2norm_in_kernel=normalize)
    if not torch.isfinite(output).all() or not torch.isfinite(state).all():
        raise GdnInputError('nonfinite recurrence result')
    return output, state
