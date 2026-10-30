"""CPU-only NVFP4 reference projection for a private, checked load transaction.

Not a GPU kernel or a production fast path. The caller must verify checkpoint
identity and keep the model private until every load and validation succeeds.
"""
import torch
from vllm.model_executor.layers.linear import LinearMethodBase, register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4W4A16LinearMethod


@register_weight_loader_v2_supported_method
class CpuNvfp4Reference(LinearMethodBase):
    def __init__(self, quant_config):
        if quant_config.group_size != 16 or quant_config.quant_method != 'W4A16_NVFP4':
            raise ValueError('expected W4A16 NVFP4 K16')
        self.quant_config = quant_config

    def create_weights(self, layer, *args, **kwargs):
        if hasattr(self, '_vf_layer'):
            raise ValueError('reference method cannot be reused across layers')
        ModelOptNvFp4W4A16LinearMethod.create_weights(self, layer, *args, **kwargs)
        self._vf_layer = layer
        self._vf_loaded = set()
        self._vf_failed = False
        self._vf_finalized = False
        for field in ('weight', 'weight_scale', 'weight_scale_2'):
            param = getattr(layer, field)
            original = param.weight_loader
            def tracked(actual, tensor, shard=None, *, _field=field, _param=param, _original=original):
                if self._vf_failed or self._vf_finalized or actual is not _param:
                    self._vf_failed = True
                    raise ValueError('invalid projection load state')
                if shard is None and len(layer.logical_widths) == 1:
                    shard_id = 0
                elif type(shard) is int and 0 <= shard < len(layer.logical_widths):
                    shard_id = shard
                else:
                    self._vf_failed = True
                    raise ValueError('explicit projection shard required')
                key = (shard_id, _field)
                if key in self._vf_loaded:
                    self._vf_failed = True
                    raise ValueError('duplicate projection field')
                try:
                    if shard is None:
                        _original(actual, tensor)
                    else:
                        _original(actual, tensor, shard)
                except BaseException:
                    self._vf_failed = True
                    raise
                self._vf_loaded.add(key)
            param.weight_loader = tracked

    def process_weights_after_loading(self, layer):
        expected = frozenset((i, name) for i in range(len(layer.logical_widths))
                             for name in ('weight', 'weight_scale', 'weight_scale_2'))
        if (getattr(self, '_vf_layer', None) is not layer or self._vf_failed
                or self._vf_finalized or self._vf_loaded != expected):
            raise ValueError('unverified or incomplete projection fields')
        if layer.weight.device.type != 'cpu' or layer.weight_scale.device.type != 'cpu':
            raise ValueError('CPU tensors required')
        group = layer.weight_scale.float()
        scales = layer.weight_scale_2.float()
        if (not torch.isfinite(group).all() or not torch.isfinite(scales).all()
                or (group < 0).any() or (scales < 0).any()):
            raise ValueError('invalid NVFP4 scales')
        codes = layer.weight.detach()
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                            -0., -.5, -1., -1.5, -2., -3., -4., -6.])
        indices = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(codes.shape[0], -1).long()
        per_row = torch.repeat_interleave(scales, torch.tensor(layer.logical_widths))
        decoded = (lut[indices] * group.repeat_interleave(16, dim=1) * per_row[:, None]).to(torch.float16)
        if not torch.isfinite(decoded).all():
            raise ValueError('nonfinite decoded weights')
        layer.register_buffer('_vf_decoded_weight', decoded, persistent=False)
        self._vf_finalized = True

    def apply(self, layer, x, bias=None):
        if x.device.type != 'cpu' or x.dtype != torch.float16 or bias is not None:
            raise ValueError('CPU FP16 input and no bias required')
        if (getattr(self, '_vf_layer', None) is not layer or self._vf_failed
                or not self._vf_finalized or not hasattr(layer, '_vf_decoded_weight')
                or not torch.isfinite(x).all()):
            raise ValueError('projection not admitted or invalid input')
        return torch.matmul(x.to(torch.float64), layer._vf_decoded_weight.to(torch.float64).T).to(torch.float16)
