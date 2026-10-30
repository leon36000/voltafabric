"""Research CPU reference for ModelOpt static E4M3 tensor-scaled linear.

Reuses 1Cat's native PyTorch FP8 activation quantizer; this is not a GPU
kernel or an assertion of bitwise equivalence with hardware matmul.
"""
import torch
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


class Fp8ReferenceError(ValueError):
    pass


class CpuStaticFp8Linear:
    def __init__(self, weight, input_scale, weight_scale):
        if (not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu'
                or weight.dtype != torch.float8_e4m3fn or weight.ndim != 2
                or min(weight.shape) <= 0 or weight.numel() > 32_000_000):
            raise Fp8ReferenceError('bounded CPU E4M3 matrix required')
        for scale in (input_scale, weight_scale):
            if (not isinstance(scale, torch.Tensor) or scale.device.type != 'cpu'
                    or scale.dtype != torch.float32 or scale.numel() != 1
                    or not torch.isfinite(scale).all() or (scale <= 0).any()):
                raise Fp8ReferenceError('positive finite scalar FP32 scale required')
        if not torch.isfinite(weight.float()).all():
            raise Fp8ReferenceError('nonfinite FP8 weight')
        self.weight = weight.detach()
        self.input_scale = input_scale.detach().clone()
        self.weight_scale = weight_scale.detach().clone()
        self.quantizer = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR,
                                  compile_native=False, quant_dtype=torch.float8_e4m3fn)

    def __call__(self, x):
        if (not isinstance(x, torch.Tensor) or x.device.type != 'cpu'
                or x.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or x.ndim != 2 or x.shape[0] <= 0 or x.shape[0] > 4096
                or x.shape[1] != self.weight.shape[1] or not torch.isfinite(x).all()):
            raise Fp8ReferenceError('invalid CPU input')
        with torch.inference_mode():
            quantized, scale = self.quantizer.forward_native(x, self.input_scale)
            out = torch.matmul(quantized.float(), self.weight.float().T)
            out = out * scale.float() * self.weight_scale
        if not torch.isfinite(out).all():
            raise Fp8ReferenceError('nonfinite output')
        return out
