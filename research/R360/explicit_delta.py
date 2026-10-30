"""Single-token explicit FP32 gated delta for the private CPU oracle.

Same arithmetic semantics as the donor recurrence, written independently;
not a high-precision reference or a GPU implementation.
"""
import math
import torch

def explicit_delta(q,k,v,g,beta,initial):
 if q.dtype!=torch.float32 or k.dtype!=torch.float32 or v.dtype!=torch.float32 or q.shape[1]!=1 or k.shape!=q.shape:
  raise ValueError('single-token FP32 inputs required')
 q=q*torch.rsqrt((q*q).sum(dim=-1,keepdim=True)+1e-6)
 k=k*torch.rsqrt((k*k).sum(dim=-1,keepdim=True)+1e-6)
 q=q.transpose(1,2).contiguous()* (1 / math.sqrt(q.shape[-1]))
 k=k.transpose(1,2).contiguous();v=v.transpose(1,2).contiguous();g=g.transpose(1,2).contiguous();beta=beta.transpose(1,2).contiguous()
 shape=(q.shape[0],q.shape[1],q.shape[-1],v.shape[-1]);state=torch.zeros(shape,dtype=torch.float32) if initial is None else initial.clone()
 state=state*g[:,:,0].exp().unsqueeze(-1).unsqueeze(-1)
 memory=(state*k[:,:,0].unsqueeze(-1)).sum(dim=-2)
 delta=(v[:,:,0]-memory)*beta[:,:,0].unsqueeze(-1)
 state=state+k[:,:,0].unsqueeze(-1)*delta.unsqueeze(-2)
 result=(state*q[:,:,0].unsqueeze(-1)).sum(dim=-2).unsqueeze(2).transpose(1,2).contiguous()
 return result,state
