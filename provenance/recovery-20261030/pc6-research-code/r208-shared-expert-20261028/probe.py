import os,sys,json,hashlib,pathlib
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import torch
from safetensors import safe_open
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform=CpuPlatform()
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
path=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r204-real-experts-20261028/shared-r208.safetensors'); assert hashlib.sha256(path.read_bytes()).hexdigest()=='6e83b12b99e891ce4eebd9adfb95960e7b40ae8d48811cd5499caa3ee6e09934';torch.set_num_threads(1)
prefix='model.language_model.layers.0.mlp.shared_expert.'
with safe_open(str(path),framework='pt',device='cpu') as f:
 def decode(proj):
  c=f.get_tensor(prefix+proj+'.weight');s=f.get_tensor(prefix+proj+'.weight_scale').float();g=f.get_tensor(prefix+proj+'.weight_scale_2').float();lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]);n,k=c.shape;ids=torch.stack((c&15,c>>4),dim=-1).reshape(n,k*2).long();return (lut[ids]*s.repeat_interleave(16,dim=1)*g).to(torch.float16).to(torch.float64)
 gate,up,down=[decode(k) for k in ('gate_proj','up_proj','down_proj')];linear_gate=f.get_tensor('model.language_model.layers.0.mlp.shared_expert_gate.weight').to(torch.float16)
class Project(torch.nn.Module):
 def __init__(self,w):super().__init__();self.w=w
 def forward(self,x):return torch.matmul(x.to(torch.float64),self.w.T).to(torch.float16),None
class Act(torch.nn.Module):
 def forward(self,x):a,b=x.chunk(2,dim=-1);return (torch.nn.functional.silu(a)*b).to(torch.float16)
class SharedGate(torch.nn.Module):
 def forward(self,x):return torch.nn.functional.linear(x,linear_gate),None
instance=object.__new__(Qwen2MoeMLP);torch.nn.Module.__init__(instance);instance.layer_idx=0;instance.gate_up_proj=Project(torch.cat([gate,up]));instance.down_proj=Project(down);instance.act_fn=Act();instance.expert_gate=SharedGate()
x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);out=Qwen2MoeMLP.forward(instance,x);assert out.shape==(1,2048) and torch.isfinite(out).all() and not torch.cuda.is_initialized();payload=out.contiguous().numpy().tobytes();root=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r208-shared-expert-20261028-r1');root.mkdir(mode=0o700);(root/'output.fp16').write_bytes(payload);receipt={'status':'SUCCESS','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(payload).hexdigest(),'cuda_initialized':False,'gpu':False};(root/'receipt.json').write_text(json.dumps(receipt,sort_keys=True)+'\n');print(json.dumps(receipt))
