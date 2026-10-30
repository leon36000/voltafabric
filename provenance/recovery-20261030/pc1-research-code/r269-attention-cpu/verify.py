import json, os, sys, torch
from safetensors import safe_open
from torch import nn
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention

import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
from vllm.config import VllmConfig, set_current_vllm_config
platforms._current_platform=CpuPlatform()
from adapters.qwen_full_attention_cpu import QwenFullAttentionCpuReference
ctx=set_current_vllm_config(VllmConfig());ctx.__enter__()
torch.set_num_threads(4)
p=os.environ['MODEL_SHARD']
prefix='model.language_model.layers.3.self_attn.'
with safe_open(p,framework='pt',device='cpu') as f:
    w={k[len(prefix):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
a=QwenFullAttentionCpuReference(w,max_tokens=3)
class Projection(nn.Module):
    def __init__(self, ref): super().__init__(); self.ref=ref
    def forward(self,x): return self.ref(x.reshape(-1,x.shape[-1])).reshape(*x.shape[:-1],-1)
c=Qwen3NextConfig(hidden_size=2048,num_attention_heads=16,num_key_value_heads=2,head_dim=256,attention_bias=False,rms_norm_eps=1e-6)
c._attn_implementation='eager'
ref=Qwen3NextAttention(c,layer_idx=0).eval()
for n in ('q_proj','k_proj','v_proj','o_proj'): setattr(ref,n,Projection(getattr(a,n)))
ref.q_norm.weight.data=w['q_norm.weight'].float();ref.k_norm.weight.data=w['k_norm.weight'].float()
torch.manual_seed(269)
x=torch.randn(1,2,2048)*0.1
positions=torch.arange(2)
inv=1/(10_000_000.0**(torch.arange(0,64,2,dtype=torch.float32)/64))
angles=positions[:,None]*inv; angles=torch.cat((angles,angles),-1)[None,:,:]
mask=torch.tensor([[[[0.0,float('-inf')],[0.0,0.0]]]])
with torch.inference_mode():
    got,cache=a(x);expected,_=ref(x,(angles.cos(),angles.sin()),mask)
    y0,c0=a(x[:,:1]);y1,c1=a(x[:,1:],c0)
err=(got-expected).abs().max().item(); cache_err=(got-torch.cat((y0,y1),1)).abs().max().item()
assert err < 1e-5,(err,'donor');assert cache_err < 1e-5,(cache_err,'cache')
assert c1[0].shape==(1,2,2,256)
failures=0
for bad in [(torch.zeros(1,2,1,256),torch.zeros(1,2,2,256)), (torch.zeros(1,2,1,256),torch.full((1,2,1,256),float('nan')))]:
    try: a(x[:,:1],bad)
    except ValueError: failures+=1
try: a(x,c1)
except ValueError: failures+=1
assert failures==3
print(json.dumps({'oracle_max_abs':err,'cached_vs_batch_max_abs':cache_err,'cache_shape':list(c1[0].shape),'refusals':failures,'checkpoint':'qwen layer 3','gpu':False}))
