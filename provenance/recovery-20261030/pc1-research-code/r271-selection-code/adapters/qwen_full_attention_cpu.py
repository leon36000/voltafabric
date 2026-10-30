"""Qwen full-attention FP8 CPU reference; cache is caller-owned, not transactional."""
import math
import torch
from .onecat_cpu_fp8 import CpuStaticFp8Linear

class QwenFullAttentionCpuReference:
    def __init__(self, weights, *, max_tokens=4096):
        self.max_tokens = max_tokens
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
            raise ValueError('invalid cache bound')
        shapes = {'q_proj': (8192, 2048), 'k_proj': (512, 2048),
                  'v_proj': (512, 2048), 'o_proj': (2048, 4096)}
        for name, shape in shapes.items():
            if weights[name + '.weight'].shape != shape:
                raise ValueError('invalid projection shape')
            setattr(self, name, CpuStaticFp8Linear(weights[name+'.weight'],
                    weights[name+'.input_scale'], weights[name+'.weight_scale']))
        for name in ('q_norm.weight', 'k_norm.weight'):
            t = weights[name]
            if t.device.type != 'cpu' or t.shape != (256,) or t.dtype != torch.bfloat16 or not torch.isfinite(t.float()).all():
                raise ValueError('invalid norm weight')
        self.qnorm = weights['q_norm.weight'].float()
        self.knorm = weights['k_norm.weight'].float()

    @staticmethod
    def _norm(x, w):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + w)

    @staticmethod
    def _rope(x, positions):
        inv = 1.0 / (10_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        angles = positions.float().unsqueeze(-1) * inv
        angles = torch.cat((angles, angles), -1)[None, None, :, :]
        a, b = x[..., :32], x[..., 32:64]
        rot = torch.cat((-b, a), -1)
        return torch.cat((x[..., :64]*angles.cos() + rot*angles.sin(), x[..., 64:]), -1)

    def __call__(self, hidden, cache=None):
        if (not isinstance(hidden, torch.Tensor) or hidden.device.type != 'cpu' or hidden.dtype != torch.float32
                or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 2048
                or hidden.shape[1] < 1 or not torch.isfinite(hidden).all()):
            raise ValueError('invalid hidden states')
        n = hidden.shape[1]
        if cache is None:
            old = 0
        else:
            if not isinstance(cache, tuple) or len(cache) != 2:
                raise ValueError('invalid KV cache')
            k0, v0 = cache
            if (not isinstance(k0, torch.Tensor) or not isinstance(v0, torch.Tensor)
                    or k0.device.type != 'cpu' or v0.device.type != 'cpu'
                    or k0.dtype != torch.float32 or v0.dtype != torch.float32
                    or k0.ndim != 4 or k0.shape != v0.shape or k0.shape[:2] != (1, 2)
                    or k0.shape[-1] != 256 or not torch.isfinite(k0).all() or not torch.isfinite(v0).all()):
                raise ValueError('invalid KV cache')
            old = k0.shape[2]
        if old + n > self.max_tokens:
            raise ValueError('cache capacity exceeded')
        with torch.inference_mode():
            x = hidden.reshape(n, 2048)
            q, gate = self.q_proj(x).reshape(1,n,16,512).split(256, -1)
            q = self._norm(q, self.qnorm).transpose(1,2)
            k = self._norm(self.k_proj(x).reshape(1,n,2,256), self.knorm).transpose(1,2)
            v = self.v_proj(x).reshape(1,n,2,256).transpose(1,2)
            pos = torch.arange(old, old+n)
            q, k = self._rope(q, pos), self._rope(k, pos)
            if cache is not None:
                k, v = torch.cat((k0,k),2), torch.cat((v0,v),2)
            kr, vr = k.repeat_interleave(8,1), v.repeat_interleave(8,1)
            scores = q @ kr.transpose(-1,-2) / math.sqrt(256)
            mask = torch.arange(old+n)[None,:] <= pos[:,None]
            attn = torch.softmax(scores.masked_fill(~mask, float('-inf')), dim=-1) @ vr
            gated = attn.transpose(1,2).reshape(n,4096) * torch.sigmoid(gate.reshape(n,4096))
            out = self.o_proj(gated).reshape(1,n,2048)
            if not torch.isfinite(out).all():
                raise ValueError('nonfinite attention output')
            return out, (k.contiguous(), v.contiguous())
