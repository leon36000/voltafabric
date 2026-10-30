import os,sys,pathlib,json,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
root=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r220-resident-method-constructor-20261028-r1');root.mkdir(mode=0o700);sys.path.insert(0,'/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r220-package');state={'operation_id':'r220-resident-method-constructor-20261028-r1','status':'STARTED','gpu':False}
try:
 import torch,vllm.platforms as platforms
 from vllm.platforms.cpu import CpuPlatform
 platforms._current_platform=CpuPlatform()
 from vllm.config import VllmConfig,set_current_vllm_config
 import vllm.model_executor.parameter as parammod
 parammod.get_tensor_model_parallel_rank=lambda:0;parammod.get_tensor_model_parallel_world_size=lambda:1
 import vllm.model_executor.layers.fused_moe.config as fused_cfg
 fused_cfg.get_tensor_model_parallel_rank=lambda:0
 from vllm.model_executor.layers.fused_moe.layer import FusedMoE
 from adapters.session_bridge import DurableSession
 from adapters.onecat_resident_method import ResidentMoEMethod,ResidentAdmissionError
 class Process:
  dead=False
  def poll(self):return 1 if self.dead else None
 session=object.__new__(DurableSession);session.proc=Process();session.terminal=False;session.identity={'run_id':'fixture-run','model_revision':'fixture-model','plan_id':'fixture-plan','layer':0,'epoch':1};session.experts=frozenset(range(8));session.hidden=2048
 class Quant:
  def get_quant_method(self,layer,prefix):return ResidentMoEMethod(layer.moe_config,session,model_revision='fixture-model',plan_id='fixture-plan',layer_id=0,epoch=1,layer_name=prefix)
 cfg=VllmConfig()
 def build(count,prefix):
  with set_current_vllm_config(cfg):return FusedMoE(num_experts=count,top_k=2,hidden_size=2048,intermediate_size=512,params_dtype=torch.float16,tp_size=1,dp_size=1,pcp_size=1,prefix=prefix,quant_config=Quant())
 layer=build(8,'model.layers.0.mlp.experts');assert isinstance(layer.quant_method,ResidentMoEMethod) and layer.quant_method._admitted and len(list(layer.named_parameters()))==0
 layer.ensure_moe_quant_config_init();assert layer.moe_quant_config is None
 try:build(256,'model.layers.1.mlp.experts')
 except ResidentAdmissionError:coverage_refused=True
 else:raise AssertionError('incomplete coverage accepted')
 session.proc.dead=True
 try:layer.ensure_moe_quant_config_init()
 except ResidentAdmissionError:dead_refused=True
 else:raise AssertionError('dead process accepted')
 state.update(status='SUCCESS',real_fused_constructor=True,parameter_count=len(list(layer.named_parameters())),coverage_refused=coverage_refused,dead_refused=dead_refused,fixture_session=True,cuda_initialized=torch.cuda.is_initialized())
except BaseException as e:
 state.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print(json.dumps(state,sort_keys=True));sys.exit(0 if state['status']=='SUCCESS' else 1)
