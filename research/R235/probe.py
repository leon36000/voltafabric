import os,sys,json,hashlib,pathlib,torch
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1')
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform=CpuPlatform();torch.set_num_threads(1)
from safetensors import safe_open
from vllm.config import VllmConfig,set_current_vllm_config
ctx=set_current_vllm_config(VllmConfig());ctx.__enter__()
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.model_executor.layers.linear import ReplicatedLinear
from adapters.onecat_cpu_nvfp4 import CpuNvfp4Reference
from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');model=base/'r188-qwen-checkpoint-pc1-20261028-r1/model';q=json.loads((model/'config.json').read_text())['quantization_config'];conf=ModelOptMixedPrecisionConfig.from_config(q);conf.packed_modules_mapping=Qwen3_5MoeForConditionalGeneration.packed_modules_mapping
class Quant:
 def get_quant_method(self,layer,prefix):
  assert conf._resolve_quant_algo(prefix)=='W4A16_NVFP4'
  return CpuNvfp4Reference(conf.w4a16_nvfp4_config)
import vllm.model_executor.parameter as parammod
parammod.get_tensor_model_parallel_rank=lambda:0;parammod.get_tensor_model_parallel_world_size=lambda:1
expert_gate=ReplicatedLinear(2048,1,bias=False,params_dtype=torch.float16,disable_tp=True,prefix='model.language_model.layers.0.mlp.shared_expert_gate')
mlp=Qwen2MoeMLP(2048,512,'silu',quant_config=Quant(),expert_gate=expert_gate,is_sequence_parallel=True,prefix='model.language_model.layers.0.mlp.shared_expert')
prefix='model.language_model.layers.0.mlp.shared_expert.';path=base/'r204-real-experts-20261028/shared-r208.safetensors';assert hashlib.sha256(path.read_bytes()).hexdigest()=='6e83b12b99e891ce4eebd9adfb95960e7b40ae8d48811cd5499caa3ee6e09934'
with safe_open(str(path),framework='pt',device='cpu') as f:
 for proj,layer,shard in [('gate_proj',mlp.gate_up_proj,0),('up_proj',mlp.gate_up_proj,1),('down_proj',mlp.down_proj,None)]:
  for field in ('weight','weight_scale','weight_scale_2'):
   t=f.get_tensor(prefix+proj+'.'+field);p=getattr(layer,field)
   if shard is None: p.weight_loader(p,t)
   else: p.weight_loader(p,t,shard)
 expert_gate.weight_loader(expert_gate.weight,f.get_tensor('model.language_model.layers.0.mlp.shared_expert_gate.weight').to(torch.float16))
for layer in (mlp.gate_up_proj,mlp.down_proj):
 try:layer.quant_method.process_weights_after_loading(layer)
 except ValueError:pass
 else:raise AssertionError('unverified fields accepted')
 layer._vf_verified_fields=frozenset((shard,field) for shard in range(len(layer.logical_widths)) for field in ('weight','weight_scale','weight_scale_2'))
 saved=layer.weight_scale_2.detach().clone()
 layer.weight_scale_2.data[0]=float('nan')
 try:layer.quant_method.process_weights_after_loading(layer)
 except ValueError:pass
 else:raise AssertionError('nonfinite scale accepted')
 layer.weight_scale_2.data.copy_(saved)
 layer.quant_method.process_weights_after_loading(layer)
expert_gate.quant_method.process_weights_after_loading(expert_gate)
x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16)
out=mlp(x);raw=out.contiguous().numpy().tobytes();digest=hashlib.sha256(raw).hexdigest();assert digest=='929769740d0bc2c64c6ccef34e51aaa94e622a960882cbd5cb6d5836ae2eacef' and not torch.cuda.is_initialized();print(json.dumps({'status':'PASS','output_sha256':digest,'cuda_initialized':False,'output_bytes':len(raw),'unverified_fields_refused':True,'nonfinite_scales_refused':True}))
