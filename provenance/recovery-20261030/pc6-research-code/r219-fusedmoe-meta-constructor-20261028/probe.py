import os,json,pathlib,traceback,sys
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
root=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r219-fusedmoe-meta-constructor-20261028-r1');root.mkdir(mode=0o700);state={'operation_id':'r219-fusedmoe-meta-constructor-20261028-r1','gpu':False,'status':'STARTED'}
try:
 import torch,vllm.platforms as p
 from vllm.platforms.cpu import CpuPlatform
 p._current_platform=CpuPlatform()
 from vllm.config import VllmConfig,set_current_vllm_config
 import vllm.model_executor.parameter as parammod
 parammod.get_tensor_model_parallel_rank=lambda:0;parammod.get_tensor_model_parallel_world_size=lambda:1
 import vllm.model_executor.layers.fused_moe.config as fused_cfg
 fused_cfg.get_tensor_model_parallel_rank=lambda:0
 from vllm.model_executor.layers.fused_moe.layer import FusedMoE
 cfg=VllmConfig();state['stage']='CONFIG_BUILT'
 with set_current_vllm_config(cfg), torch.device('meta'):
  layer=FusedMoE(num_experts=256,top_k=8,hidden_size=2048,intermediate_size=512,params_dtype=torch.float16,tp_size=1,dp_size=1,pcp_size=1,prefix='model.language_model.layers.0.mlp.experts')
 params=list(layer.named_parameters());state.update(status='SUCCESS',parameters=len(params),elements=sum(p.numel() for _,p in params),meta=all(p.is_meta for _,p in params),shapes={n:list(p.shape) for n,p in params},cuda_initialized=torch.cuda.is_initialized(),quant=type(layer.quant_method).__name__)
except BaseException as e:
 state.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print(json.dumps(state,sort_keys=True));sys.exit(0 if state['status']=='SUCCESS' else 1)
