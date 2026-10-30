import json,pathlib,torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule
from nvfp4_oracle import expert_forward
torch.set_num_threads(1);b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r294-twenty-layer-model-r1';r=json.loads((root/'receipt.json').read_text());cap=load_file(str(root/'capture.safetensors'));model=b/'r188-qwen-checkpoint-pc1-20261028-r1/model';weights=load_file(str(b/'r294-twenty-layer-fixture-r2/nonexpert.safetensors'));keys=['in_proj_qkv.weight','in_proj_qkv.weight_scale','in_proj_qkv.input_scale','in_proj_z.weight','in_proj_z.weight_scale','in_proj_z.input_scale','out_proj.weight','out_proj.weight_scale','out_proj.input_scale','in_proj_a.weight','in_proj_b.weight','conv1d.weight','A_log','dt_bias','norm.weight']
with safe_open(str(model/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:
 allw={i:{n:f.get_tensor(f'model.language_model.layers.{i}.linear_attn.'+n).contiguous() for n in keys} for i in (0,1,2,4,5,6,8,9,10,12,13,14,16)}
with safe_open(str(model/'model-00002-of-00003.safetensors'),framework='pt',device='cpu') as f:
 allw.update({i:{n:f.get_tensor(f'model.language_model.layers.{i}.linear_attn.'+n).contiguous() for n in keys} for i in (17,18)})
def fp8(w,name,x):
 scale=w[name+'.input_scale'];q=(x.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).float();return (q@w[name+'.weight'].float().T)*scale*w[name+'.weight_scale']
def norm(x,key):
 z=x.float();return (z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*(1+weights[key].float())).half()
def decode(pre,name):
 key=pre+'mlp.shared_expert.'+name+'.';c=weights[key+'weight'];s=weights[key+'weight_scale'].double();g=weights[key+'weight_scale_2'].double();n,k=c.shape;lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64);ix=torch.stack((c&15,c>>4),-1).reshape(n,k*2).long();return (lut[ix]*s.repeat_interleave(16,1)*g).half().double()
mats=[tuple(decode(f'model.language_model.layers.{i}.',name) for name in ('gate_proj','up_proj','down_proj')) for i in range(20)];hist={i:torch.zeros(1,8192,3) for i in (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18)};states={i:None for i in hist};results=[]
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform=CpuPlatform()
from vllm.config import VllmConfig,set_current_vllm_config
from adapters.onecat_cpu_fp8 import CpuStaticFp8Linear
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention,Qwen3NextRotaryEmbedding
ctx=set_current_vllm_config(VllmConfig());ctx.__enter__()
class Projection(torch.nn.Module):
 def __init__(self,n,aw):
  super().__init__();self.ref=CpuStaticFp8Linear(aw[n+'.weight'],aw[n+'.input_scale'],aw[n+'.weight_scale'])
 def forward(self,x):return self.ref(x.reshape(-1,x.shape[-1])).reshape(*x.shape[:-1],-1)
c=Qwen3NextConfig(hidden_size=2048,num_attention_heads=16,num_key_value_heads=2,head_dim=256,rope_theta=10_000_000.,partial_rotary_factor=.25,attention_bias=False,rms_norm_eps=1e-6);c._attn_implementation='eager';rotary=Qwen3NextRotaryEmbedding(c)
class Cache:
 def __init__(self):self.k=self.v=None
 def update(self,k,v,layer,kw):
  self.k=k if self.k is None else torch.cat((self.k,k),2);self.v=v if self.v is None else torch.cat((self.v,v),2);return self.k,self.v
full={};cache={}
with safe_open(str(model/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:
 for i in (3,7,11,15):
  pre=f'model.language_model.layers.{i}.self_attn.';aw={k[len(pre):]:f.get_tensor(k).contiguous() for k in f.keys() if k.startswith(pre)};module=Qwen3NextAttention(c,i).eval()
  for name in ('q_proj','k_proj','v_proj','o_proj'):setattr(module,name,Projection(name,aw))
  module.q_norm.weight.data=aw['q_norm.weight'].float();module.k_norm.weight.data=aw['k_norm.weight'].float();full[i]=module;cache[i]=Cache()
with safe_open(str(model/'model-00002-of-00003.safetensors'),framework='pt',device='cpu') as f:
 i=19;pre=f'model.language_model.layers.{i}.self_attn.';aw={k[len(pre):]:f.get_tensor(k).contiguous() for k in f.keys() if k.startswith(pre)};module=Qwen3NextAttention(c,i).eval()
 for name in ('q_proj','k_proj','v_proj','o_proj'):setattr(module,name,Projection(name,aw))
 module.q_norm.weight.data=aw['q_norm.weight'].float();module.k_norm.weight.data=aw['k_norm.weight'].float();full[i]=module;cache[i]=Cache()
for batch in (0,1):
 with safe_open(str(model/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:h=f.get_slice('model.language_model.embed_tokens.weight')[r['input_tokens'][batch]:r['input_tokens'][batch]+1].half()
 residual=None
 for i in range(20):
  pre=f'model.language_model.layers.{i}.';residual=h.float() if residual is None else h.float()+residual.float();z0=norm(residual,pre+'input_layernorm.weight')
  if i not in (3,7,11,15,19):
   w=allw[i];qkv=fp8(w,'in_proj_qkv',z0).reshape(1,1,8192);z=fp8(w,'in_proj_z',z0).reshape(1,1,32,128);a=F.linear(z0.float().unsqueeze(0),w['in_proj_a.weight'].float());bb=F.linear(z0.float().unsqueeze(0),w['in_proj_b.weight'].float());t=torch.cat([hist[i],qkv.transpose(1,2)],-1);conv=F.silu(F.conv1d(t,w['conv1d.weight'].float(),groups=8192)).transpose(1,2);hist[i]=t[:,:,-3:].clone();q,k,v=conv.split([2048,2048,4096],-1);q=q.reshape(1,1,16,128).repeat_interleave(2,2);k=k.reshape(1,1,16,128).repeat_interleave(2,2);v=v.reshape(1,1,32,128);g=-w['A_log'].float().exp()*F.softplus(a+w['dt_bias'].float());core,states[i]=torch_chunk_gated_delta_rule(q,k,v,g,bb.sigmoid(),chunk_size=4,initial_state=states[i],output_final_state=True,use_qk_l2norm_in_kernel=True);z1=core*torch.rsqrt(core.square().mean(-1,keepdim=True)+1e-6)*w['norm.weight'].float()*F.silu(z);attn=fp8(w,'out_proj',z1.reshape(1,4096)).reshape(1,2048).half()
  else:
   position=torch.tensor([[batch]]);cos,sin=rotary(z0.float().unsqueeze(0),position);attn,_=full[i](z0.float().unsqueeze(0),(cos,sin),None,past_key_values=cache[i]);attn=attn.reshape(1,2048).half()
  residual=attn.float()+residual.float();moe=norm(residual,pre+'post_attention_layernorm.weight');idx=batch*20+i;delta=(moe.float()-cap[f'batch{idx}.input'].float()).abs().max().item();assert delta<=.003,(batch,i,'moe',delta)
  ids=cap[f'batch{idx}.ids'][0].tolist();co=cap[f'batch{idx}.weights'][0].tolist();source=model/'model-00001-of-00003.safetensors';acc=torch.zeros(2048,dtype=torch.float64)
  for e,vv in zip(ids,co):acc+=expert_forward(source if i<16 or (i==16 and e<139) else source.with_name('model-00002-of-00003.safetensors'),i,e,moe[0]).double()*vv
  routed=acc.half().unsqueeze(0);gw,uw,dw=mats[i];a=(moe.double()@gw.T).half();u=(moe.double()@uw.T).half();mid=(F.silu(a)*u).half();sg=weights[pre+'mlp.shared_expert_gate.weight'].half();shared=(mid.double()@dw.T).half()*torch.sigmoid(F.linear(moe,sg));h=(routed+shared).half()
 out=norm(residual+h.float(),'model.language_model.norm.weight');err=(out.float()-cap[f'batch{batch}.output'].float()).abs().max().item();assert err<=.003,(batch,'output',err);results.append({'batch':batch,'max_output_error':err})
result={'status':'PASS','cases':results,'threshold':.003,'gpu':False};(root/'decoder-oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
