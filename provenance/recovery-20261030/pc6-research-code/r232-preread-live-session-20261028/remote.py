import os,sys,pathlib,json,hashlib,base64,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r232-preread-live-session-20261028-r1';root.mkdir(mode=0o700);sys.path.insert(0,str(base/'r232-package'));state={'status':'STARTED','operation_id':'r232-preread-live-session-20261028-r1','gpu':False}
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
 p=json.loads((base/'r232-input-r1.json').read_bytes());stamp=p['routing']['stamp'];session=DurableSession(base/'r232-proxy-r1.py',p,root/'ledger',deadline_seconds=30);state.update(run_id=stamp['run_id'],ledger=str(session.directory),stage='SESSION_READY');(root/'progress.json').write_text(json.dumps(state)+'\n')
 class Quant:
  def get_quant_method(self,layer,prefix):return ResidentMoEMethod(layer.moe_config,session,model_revision=stamp['model_revision'],plan_id=stamp['plan_id'],layer_id=stamp['layer'],epoch=stamp['epoch'],layer_name=prefix)
 def cpu_router(hidden_states,gating_output,topk,renormalize):
  assert topk==8 and renormalize
  ids=torch.tensor([[5,0,7,2,4,1,6,3]],dtype=torch.int32);weights=torch.full((1,8),.125,dtype=torch.float32)
  return weights,ids
 cfg=VllmConfig()
 with set_current_vllm_config(cfg):layer=FusedMoE(num_experts=8,top_k=8,hidden_size=2048,intermediate_size=512,params_dtype=torch.float16,tp_size=1,dp_size=1,pcp_size=1,prefix='model.language_model.layers.0.mlp.experts',quant_config=Quant(),custom_routing_function=cpu_router)
 assert len(list(layer.named_parameters()))==0
 from adapters.onecat_pre_read import ResidentPreReadFilter
 from safetensors.torch import safe_open
 reads=[]
 class TrackedOpen:
  def __init__(self,*a,**k):self.ctx=safe_open(*a,**k)
  def __enter__(self):self.f=self.ctx.__enter__();return self
  def __exit__(self,*a):return self.ctx.__exit__(*a)
  def keys(self):return self.f.keys()
  def get_tensor(self,name):reads.append(name);return self.f.get_tensor(name)
 from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
 from vllm.config import LoadConfig
 import types
 fixture=base/'r230-loader-fixture-r1';assert hashlib.sha256((fixture/'model.safetensors').read_bytes()).hexdigest()=='d5131737934574cb89f2a5cfdd11b1d56f4bc1bc9185f99de960357312d8db21'
 class CaptureModel(torch.nn.Module):
  def __init__(self):super().__init__();self.names=[]
  def load_weights(self,values):
   for name,_ in values:self.names.append(name)
   return set(self.names)
 model=CaptureModel();model_config=types.SimpleNamespace(model=str(fixture),revision=None,quantization='modelopt',is_moe=False)
 loader=DefaultModelLoader(LoadConfig(load_format='safetensors',use_tqdm_on_load=False));fil=ResidentPreReadFilter({0:layer.quant_method},ResidentMoEMethod,opener=TrackedOpen)
 with set_current_vllm_config(cfg):fil.load(loader,model_config,model)
 assert len(fil.seen[0])==96 and fil.finished and len(model.names)==3 and session.proc.poll() is None
 assert sorted(reads)==sorted(model.names) and len(reads)==3
 state['materialized_names']=sorted(reads)
 state['loader_filtered']=len(fil.seen[0]);state['loader_retained']=sorted(model.names);state['loader_session_alive']=True
 x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);ids=torch.tensor([[5,0,7,2,4,1,6,3]]);weights=torch.tensor([[.125]*8]);outputs=[]
 from vllm.forward_context import ForwardContext,override_forward_context
 context=ForwardContext(no_compile_layers={layer.layer_name:layer},attn_metadata={},slot_mapping={})
 with override_forward_context(context):
  for _ in range(2):
   out=layer(x,torch.zeros((1,8),dtype=torch.float16));outputs.append(base64.b64encode(out.numpy().tobytes()).decode())
 closed=session.close();assert all(c['exit_code']==0 and c['bye_execs']==2 for c in closed['close'])
 try:layer.ensure_moe_quant_config_init()
 except ResidentAdmissionError:pass
 else:raise AssertionError('closed session admitted')
 state.update(status='SUCCESS',outputs=outputs,closed_refused=True,zero_expert_parameters=True,cuda_initialized=torch.cuda.is_initialized())
except BaseException as e:
 state.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print('VF_RECEIPT '+json.dumps(state,sort_keys=True),flush=True);sys.exit(0 if state['status']=='SUCCESS' else 1)
