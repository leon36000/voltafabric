"""Offline replay of confirmed R344 expert rows through donor CPU primitives.

No session is reopened; results are diagnostic, not a new generation.
"""
import hashlib,json,pathlib,torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform=CpuPlatform()
from vllm.config import VllmConfig,set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.activation import SiluAndMul
from adapters.qwen_gdn_attention_cpu import QwenGdnCpuReference
from adapters.qwen_full_attention_cpu import QwenFullAttentionCpuReference
from adapters.onecat_cpu_nvfp4 import CpuNvfp4Reference
from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule

torch.set_num_threads(1);assert not torch.cuda.is_initialized()
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r344-single-load-forty-layer-r1';receipt=json.loads((root/'receipt.json').read_text());assert receipt['status']=='SUCCESS'
cap_path=root/'capture.safetensors';assert hashlib.sha256(cap_path.read_bytes()).hexdigest()==receipt['capture_sha256'];cap=load_file(str(cap_path));fixture=b/'r299-forty-layer-fixture-r1/nonexpert.safetensors';assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='adbf0ee2a8e67a89d178fdbe81ee0fe88bf6197ec541beaee045524ae9919663'
model=b/'r188-qwen-checkpoint-pc1-20261028-r1/model';shard=model/'model-00001-of-00003.safetensors';cfg=VllmConfig();gdn_names=['in_proj_qkv.weight','in_proj_qkv.weight_scale','in_proj_qkv.input_scale','in_proj_z.weight','in_proj_z.weight_scale','in_proj_z.input_scale','out_proj.weight','out_proj.weight_scale','out_proj.input_scale','in_proj_a.weight','in_proj_b.weight','conv1d.weight','A_log','dt_bias','norm.weight']
refs={};attn_cache={3:None,7:None,11:None};history={};state={};norms={};mats={};gates={}
lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
def decode(f,k):
 c=f.get_tensor(k+'weight');s=f.get_tensor(k+'weight_scale').float();g=f.get_tensor(k+'weight_scale_2').float();ix=torch.stack((c&15,c>>4),-1).reshape(c.shape[0],-1).long();return (lut[ix]*s.repeat_interleave(16,1)*g).half()
def projection(matrix,x):
 layer=type('Layer',(),{})();layer._vf_decoded_weight=matrix
 method=object.__new__(CpuNvfp4Reference);method._vf_layer=layer;method._vf_failed=False;method._vf_finalized=True;return method.apply(layer,x)
with set_current_vllm_config(cfg),safe_open(str(shard),framework='pt',device='cpu') as ck,safe_open(str(fixture),framework='pt',device='cpu') as fx:
 for i in range(12):
  pre=f'model.language_model.layers.{i}.'
  if i in (3,7,11):
   key=pre+'self_attn.';w={k[len(key):]:ck.get_tensor(k).contiguous() for k in ck.keys() if k.startswith(key)};refs[i]=QwenFullAttentionCpuReference(w)
  else:
   key=pre+'linear_attn.';w={n:ck.get_tensor(key+n).contiguous() for n in gdn_names};refs[i]=QwenGdnCpuReference(w);history[i]=torch.zeros((1,8192,3),dtype=torch.float32);state[i]=None
  norms[i]=[]
  for n in ('input_layernorm.weight','post_attention_layernorm.weight'):
   module=GemmaRMSNorm(2048,eps=1e-6);module.weight.data=fx.get_tensor(pre+n).float();norms[i].append(module)
  key=pre+'mlp.shared_expert.';mats[i]=tuple(decode(fx,key+n+'.') for n in ('gate_proj','up_proj','down_proj'));gates[i]=fx.get_tensor(pre+'mlp.shared_expert_gate.weight').half()
records=[];quant_records=[]
def check_quant(proj,x,layer,batch,label):
 q,scale=proj.quantizer.forward_native(x,proj.input_scale)
 direct=(x.float()/proj.input_scale).clamp(-448,448).to(torch.float8_e4m3fn)
 neq=q.float()!=direct.float();count=int(neq.sum());out_delta=0.0
 if count:
  a=(q.float()@proj.weight.float().T)*scale.float()*proj.weight_scale
  b=(direct.float()@proj.weight.float().T)*proj.input_scale.float()*proj.weight_scale
  out_delta=float((a-b).abs().max())
 quant_records.append({'batch':batch,'layer':layer,'projection':label,'different_values':count,'output_delta':out_delta})
class FullOutputSpy:
 def __init__(self,original,layer):self.original=original;self.layer=layer
 def __call__(self,x):
  check_quant(self.original,x,self.layer,batch,'attention.o_proj')
  return self.original(x)
for i in (3,7,11):refs[i].o_proj=FullOutputSpy(refs[i].o_proj,i)
alt_history={i:torch.zeros((1,8192,3),dtype=torch.float32) for i in range(12) if i not in (3,7,11)};alt_state={i:None for i in alt_history}
def manual_norm(module,x,residual):
 z=x.float() if residual is None else x.float()+residual.float()
 return (z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*(1+module.weight.float())).to(x.dtype)
def alt_gdn(i,z):
 ref=refs[i];flat=z.reshape(1,2048);qkv=ref.qkv(flat).reshape(1,1,8192);zz=ref.z(flat).reshape(1,1,32,128);a=F.linear(z,ref.a);bb=F.linear(z,ref.b);t=torch.cat([alt_history[i],qkv.transpose(1,2)],-1);conv=F.silu(F.conv1d(t,ref.conv.unsqueeze(1),groups=8192)).transpose(1,2);alt_history[i]=t[:,:,-3:].clone();q,k,v=conv.split([2048,2048,4096],-1);q=q.reshape(1,1,16,128).repeat_interleave(2,2);k=k.reshape(1,1,16,128).repeat_interleave(2,2);v=v.reshape(1,1,32,128);decay=-ref.a_log.exp()*F.softplus(a+ref.dt);core,alt_state[i]=torch_chunk_gated_delta_rule(q,k,v,decay,bb.sigmoid(),chunk_size=4,initial_state=alt_state[i],output_final_state=True,use_qk_l2norm_in_kernel=True);zz=core*torch.rsqrt(core.square().mean(-1,keepdim=True)+1e-6)*ref.norm*F.silu(zz);check_quant(ref.out,zz.reshape(1,4096),i,batch,'gdn.out');return ref.out(zz.reshape(1,4096)).reshape(1,1,2048)
with set_current_vllm_config(cfg),torch.inference_mode(),safe_open(str(shard),framework='pt',device='cpu') as ck:
 for batch,token in enumerate(receipt['input_tokens']):
  hidden=ck.get_slice('model.language_model.embed_tokens.weight')[token:token+1].half();residual=None
  for i in range(12):
   manual_input=manual_norm(norms[i][0],hidden,residual)
   if residual is None:residual=hidden;z0=norms[i][0].forward_native(hidden)
   else:z0,residual=norms[i][0].forward_native(hidden,residual)
   norm_input_delta=float((z0.float()-manual_input.float()).abs().max());gdn_delta=None
   for label in (('q_proj','k_proj','v_proj') if i in (3,7,11) else ('qkv','z')):check_quant(getattr(refs[i],label),z0.float(),i,batch,label)
   if i in (3,7,11):
    attention,attn_cache[i]=refs[i](z0.float().unsqueeze(0),attn_cache[i])
   else:
    attention,history[i],state[i]=refs[i](z0.float().unsqueeze(0),history[i],state[i]);alt=alt_gdn(i,z0.float().unsqueeze(0));gdn_delta=float((attention-alt).abs().max())
   manual_post=manual_norm(norms[i][1],attention.squeeze(0).half(),residual)
   moe,residual=norms[i][1].forward_native(attention.squeeze(0).half(),residual)
   norm_post_delta=float((moe.float()-manual_post.float()).abs().max())
   captured=cap[f'batch{batch*40+i}.input'];delta=float((moe.float()-captured.float()).abs().max())
   p=pathlib.Path(receipt['ledgers'][str(i)]);raw=(p/f'batch-{batch:020d}.fp16').read_bytes();done=json.loads((p/f'batch-{batch:020d}.done.json').read_text());assert hashlib.sha256(raw).hexdigest()==done['sha256'];routed=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(1,2048)
   gate,up,down=mats[i];fused=projection(torch.cat((gate,up),0),moe);mid=SiluAndMul.forward_native(fused);shared=projection(down,mid)*torch.sigmoid(F.linear(moe,gates[i]));hidden=(routed+shared).half()
   records.append({'batch':batch,'layer':i,'moe_input_delta':delta,'input_norm_delta':norm_input_delta,'post_norm_delta':norm_post_delta,'gdn_output_delta':gdn_delta})
out={'status':'DIAGNOSTIC_FP8_COMPARISON','records':records,'quant_records':quant_records,'gpu':False,'new_expert_execution':False};dest=b/'r357-fp8-comparison-r1';dest.mkdir(mode=0o700)
with (dest/'receipt.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
