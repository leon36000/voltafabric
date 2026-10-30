import os,sys,pathlib,json,hashlib,base64,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r222-onecat-live-session-20261028-r1';root.mkdir(mode=0o700);sys.path.insert(0,str(base/'r220-package'));state={'status':'STARTED','operation_id':'r222-onecat-live-session-20261028-r1','gpu':False}
try:
 import torch,vllm.platforms as platforms
 from vllm.platforms.cpu import CpuPlatform
 platforms._current_platform=CpuPlatform();torch.set_num_threads(1)
 from vllm.config import VllmConfig,set_current_vllm_config
 import vllm.model_executor.parameter as parammod
 parammod.get_tensor_model_parallel_rank=lambda:0;parammod.get_tensor_model_parallel_world_size=lambda:1
 import vllm.model_executor.layers.fused_moe.config as fused_cfg
 fused_cfg.get_tensor_model_parallel_rank=lambda:0
 from vllm.model_executor.layers.fused_moe.layer import FusedMoE
 from adapters.session_bridge import DurableSession
 from adapters.onecat_resident_method import ResidentMoEMethod,ResidentAdmissionError
 p=json.loads((base/'r222-input-r1.json').read_bytes());stamp=p['routing']['stamp'];session=DurableSession(base/'r222-proxy-r1.py',p,root/'ledger',deadline_seconds=30);state.update(run_id=stamp['run_id'],ledger=str(session.directory),stage='SESSION_READY');(root/'progress.json').write_text(json.dumps(state)+'\n')
 class Quant:
  def get_quant_method(self,layer,prefix):return ResidentMoEMethod(layer.moe_config,session,model_revision=stamp['model_revision'],plan_id=stamp['plan_id'],layer_id=stamp['layer'],epoch=stamp['epoch'],layer_name=prefix)
 cfg=VllmConfig()
 with set_current_vllm_config(cfg):layer=FusedMoE(num_experts=8,top_k=8,hidden_size=2048,intermediate_size=512,params_dtype=torch.float16,tp_size=1,dp_size=1,pcp_size=1,prefix='model.layers.0.mlp.experts',quant_config=Quant())
 assert len(list(layer.named_parameters()))==0
 x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);ids=torch.tensor([[5,0,7,2,4,1,6,3]]);weights=torch.tensor([[.125]*8]);outputs=[]
 for _ in range(2):
  out=layer.quant_method.apply(layer,x,weights,ids,None,None);outputs.append(base64.b64encode(out.numpy().tobytes()).decode())
 closed=session.close();assert all(c['exit_code']==0 and c['bye_execs']==2 for c in closed['close'])
 try:layer.ensure_moe_quant_config_init()
 except ResidentAdmissionError:pass
 else:raise AssertionError('closed session admitted')
 state.update(status='SUCCESS',outputs=outputs,closed_refused=True,zero_expert_parameters=True,cuda_initialized=torch.cuda.is_initialized())
except BaseException as e:
 state.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print('VF_RECEIPT '+json.dumps(state,sort_keys=True),flush=True);sys.exit(0 if state['status']=='SUCCESS' else 1)
