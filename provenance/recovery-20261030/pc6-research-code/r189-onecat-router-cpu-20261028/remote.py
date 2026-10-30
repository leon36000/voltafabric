import os,json,pathlib,traceback,sys
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
root=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r189-onecat-router-cpu-20261028-r1');root.mkdir(mode=0o700)
s={'operation_id':'r189-onecat-router-cpu-20261028-r1','status':'STARTED','gpu_execution':False,'cpu_platform_substituted':True}
try:
 import torch,vllm.platforms as platforms
 from vllm.platforms.cpu import CpuPlatform
 platforms._current_platform=CpuPlatform()
 from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
 router=FusedTopKRouter(top_k=2,global_num_experts=3,renormalize=False,scoring_func='softmax')
 x=torch.zeros((1,4),dtype=torch.float16);logits=torch.tensor([[0.0,1.0,-1.0]],dtype=torch.float32)
 weights,ids=router.select_experts(hidden_states=x,router_logits=logits)
 assert x.device.type=='cpu' and not torch.cuda.is_initialized();s.update(status='SUCCESS',weights=weights.tolist(),ids=ids.tolist(),cuda_initialized=torch.cuda.is_initialized())
except BaseException as e:
 s.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:600]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');print(json.dumps(s,sort_keys=True));sys.exit(0 if s['status']=='SUCCESS' else 1)
