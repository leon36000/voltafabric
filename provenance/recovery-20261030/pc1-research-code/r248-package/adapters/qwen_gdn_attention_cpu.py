"""Experimental unfused FP32 CPU reference for Qwen3.5 GDN layer 0 layout.

Requires authenticated checkpoint tensors and a private state transaction from
caller. No V100/Blackwell kernel, compiled graph or production cache semantics.
"""
import math
import torch
import torch.nn.functional as F
from .onecat_cpu_fp8 import CpuStaticFp8Linear
from .qwen_gdn_projected_cpu import projected_core


class QwenGdnCpuReference:
    def __init__(self, weights, *, eps=1e-6):
        if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError('invalid norm epsilon')
        specs={'in_proj_qkv.weight':((8192,2048),torch.float8_e4m3fn),
               'in_proj_z.weight':((4096,2048),torch.float8_e4m3fn),
               'out_proj.weight':((2048,4096),torch.float8_e4m3fn),
               'in_proj_a.weight':((32,2048),torch.bfloat16),
               'in_proj_b.weight':((32,2048),torch.bfloat16),
               'conv1d.weight':((8192,1,4),torch.bfloat16),
               'A_log':((32,),torch.bfloat16),'dt_bias':((32,),torch.bfloat16),
               'norm.weight':((128,),torch.bfloat16)}
        for name,(shape,dtype) in specs.items():
            t=weights.get(name)
            if (not isinstance(t,torch.Tensor) or t.device.type!='cpu'
                    or t.shape!=shape or t.dtype!=dtype or not torch.isfinite(t.float()).all()):
                raise ValueError('invalid checkpoint tensor '+name)
        self.qkv=CpuStaticFp8Linear(weights['in_proj_qkv.weight'],weights['in_proj_qkv.input_scale'],weights['in_proj_qkv.weight_scale'])
        self.z=CpuStaticFp8Linear(weights['in_proj_z.weight'],weights['in_proj_z.input_scale'],weights['in_proj_z.weight_scale'])
        self.out=CpuStaticFp8Linear(weights['out_proj.weight'],weights['out_proj.input_scale'],weights['out_proj.weight_scale'])
        self.a=weights['in_proj_a.weight'].float();self.b=weights['in_proj_b.weight'].float()
        self.conv=weights['conv1d.weight'].squeeze(1).float();self.a_log=weights['A_log'].float();self.dt=weights['dt_bias'].float();self.norm=weights['norm.weight'].float();self.eps=float(eps)

    def __call__(self, hidden, conv_history, recurrent_state):
        if (not isinstance(hidden,torch.Tensor) or hidden.device.type!='cpu'
                or hidden.dtype!=torch.float32 or hidden.ndim!=3 or hidden.shape[-1]!=2048
                or min(hidden.shape)<=0 or hidden.shape[0]*hidden.shape[1]>4096
                or not torch.isfinite(hidden).all()):
            raise ValueError('invalid bounded CPU hidden states')
        batch,tokens,_=hidden.shape
        with torch.inference_mode():
            flat=hidden.reshape(batch*tokens,2048)
            qkv=self.qkv(flat).reshape(batch,tokens,8192)
            z=self.z(flat).reshape(batch,tokens,32,128)
            a=F.linear(hidden,self.a);b=F.linear(hidden,self.b)
            core,next_history,next_state=projected_core(qkv,a,b,self.conv,self.a_log,self.dt,conv_history,recurrent_state,key_heads=16,value_heads=32,key_dim=128,value_dim=128)
            normalized=core*torch.rsqrt(core.square().mean(-1,keepdim=True)+self.eps)
            gated=normalized*self.norm*F.silu(z)
            output=self.out(gated.reshape(batch*tokens,4096)).reshape(batch,tokens,2048)
        if not torch.isfinite(output).all():
            raise ValueError('nonfinite attention output')
        return output,next_history,next_state
