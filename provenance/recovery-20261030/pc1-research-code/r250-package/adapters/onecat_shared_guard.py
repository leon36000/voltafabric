"""Strict CPU-only seam for 1Cat SharedExperts in NO_OVERLAP mode.

The 1Cat runner computes shared experts before invoking its routed quant method,
then consumes and adds the shared result itself. This adapter never reads the
consuming `output` property, computes the shared branch, or adds it to routed
results. It only accepts the explicitly pinned non-overlapped one-rank path.
"""
import torch

class SharedOrderError(RuntimeError):
    pass

def call_after_shared(session, layer, x, weights, ids, shared, shared_input, *, shared_type, no_overlap):
    if type(shared) is not shared_type or shared_input is not x:
        raise SharedOrderError('shared object or input identity mismatch')
    if shared.enable_dbo or shared._output_idx != 0:
        raise SharedOrderError('dual-batch overlap is not qualified')
    if shared._determine_shared_experts_order(shared_input) != no_overlap:
        raise SharedOrderError('shared order is not NO_OVERLAP')
    if not isinstance(shared._output, list) or len(shared._output) != 2 or shared._output[1] is not None:
        raise SharedOrderError('invalid pending shared state')
    out=shared._output[0]
    if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float16
        or not isinstance(out,torch.Tensor) or out.device!=x.device or out.dtype!=x.dtype
        or out.shape!=x.shape or not torch.isfinite(out).all()):
        raise SharedOrderError('shared output is absent or invalid')
    return session(layer, x, weights, ids, None, None)
