"""Independent CPU oracle on original row-major ModelOpt NVFP4 tensors.

This intentionally does not call QPN2 decode, the owner, or Rust. It models one
FP16-rounded dequant per matrix and FP16 intermediates; other correct arithmetic
paths can differ due to rounding and must be compared with an explicit tolerance.
"""
import torch
from safetensors import safe_open

E2M1 = [0.0, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

def expert_forward(path, layer, expert, x):
    if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float16
        or x.ndim!=1 or not bool(torch.isfinite(x).all())):
        raise ValueError('finite CPU FP16 vector required')
    prefix=f'model.language_model.layers.{layer}.mlp.experts.{expert}.'
    with safe_open(str(path),framework='pt',device='cpu') as f:
        def matrix(proj):
            name=prefix+proj+'.';codes=f.get_tensor(name+'weight');scales=f.get_tensor(name+'weight_scale');global_scale=f.get_tensor(name+'weight_scale_2')
            if codes.dtype!=torch.uint8 or codes.ndim!=2 or scales.dtype!=torch.float8_e4m3fn or global_scale.numel()!=1:
                raise ValueError('NVFP4 tensor ABI differs')
            n,k2=codes.shape
            if tuple(scales.shape)!=(n,k2//8):raise ValueError('NVFP4 scale geometry differs')
            factor=float(global_scale.item())
            if not factor>0 or not torch.isfinite(global_scale).all():raise ValueError('invalid global scale')
            ids=torch.stack((codes&15,codes>>4),dim=-1).reshape(n,k2*2).long()
            lut=torch.tensor(E2M1,dtype=torch.float64)
            scale=scales.to(torch.float64).repeat_interleave(16,dim=1)
            if not torch.isfinite(scale).all() or (scale<0).any():raise ValueError('invalid block scale')
            return (lut[ids]*scale*factor).to(torch.float16).to(torch.float64)
        gate=torch.mv(matrix('gate_proj'),x.to(torch.float64)).to(torch.float16)
        up=torch.mv(matrix('up_proj'),x.to(torch.float64)).to(torch.float16)
        hidden=(torch.nn.functional.silu(gate)*up).to(torch.float16)
        return torch.mv(matrix('down_proj'),hidden.to(torch.float64)).to(torch.float16)
