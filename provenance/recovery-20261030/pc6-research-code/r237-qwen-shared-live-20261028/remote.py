import os,sys,pathlib,json,hashlib,base64,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r237-qwen-shared-live-20261028-r1';root.mkdir(mode=0o700);sys.path.insert(0,str(base/'r237-package'));state={'status':'STARTED','operation_id':'r237-qwen-shared-live-20261028-r1','gpu':False}
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
 p=json.loads((base/'r237-input-r1.json').read_bytes());stamp=p['routing']['stamp'];session=DurableSession(base/'r237-proxy-r1.py',p,root/'ledger',deadline_seconds=30);state.update(run_id=stamp['run_id'],ledger=str(session.directory),stage='SESSION_READY');(root/'progress.json').write_text(json.dumps(state)+'\n')
 import types
 from adapters.onecat_quant_overlay import ResidentQuantOverlay
 import vllm.model_executor.models.qwen3_next as modelmod
 from vllm.transformers_utils.config import get_config
 checkpoint=base/'r188-qwen-checkpoint-pc1-20261028-r1/model'
 hf=get_config(str(checkpoint),trust_remote_code=False,config_format='hf',local_files_only=True)
 hf.text_config.num_experts=8
 from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
 from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
 qconfig=json.loads((checkpoint/'config.json').read_text())['quantization_config']
 original=ModelOptMixedPrecisionConfig.from_config(qconfig)
 overlay=ResidentQuantOverlay(original,{0:session},model_revision=stamp['model_revision'],plan_id=stamp['plan_id'],epoch=stamp['epoch'])
 overlay.packed_modules_mapping=Qwen3_5MoeForConditionalGeneration.packed_modules_mapping
 assert original._resolve_quant_algo('model.language_model.layers.0.mlp.shared_expert.gate_up_proj')=='W4A16_NVFP4'
 from adapters.onecat_cpu_shared_overlay import CpuSharedOverlay
 cpu_overlay=CpuSharedOverlay(overlay,[0])
 state['delegate']=original.get_name()
 cfg=VllmConfig()
 modelcfg=types.SimpleNamespace(model_config=types.SimpleNamespace(hf_text_config=hf.text_config),parallel_config=cfg.parallel_config,quant_config=cpu_overlay)
 class Group:
  device_group=types.SimpleNamespace(size=lambda:1)
  rank_in_group=0
 modelmod.get_tensor_model_parallel_world_size=lambda:1
 modelmod.get_ep_group=lambda:Group()
 modelmod.get_current_vllm_config=lambda:modelcfg
 import vllm.model_executor.layers.linear as linearmod
 linearmod.get_tensor_model_parallel_world_size=lambda:1
 linearmod.get_tensor_model_parallel_rank=lambda:0
 from vllm.model_executor.layers.linear import ReplicatedLinear
 modelmod.ReplicatedLinear=lambda *args,**kw:ReplicatedLinear(*args,**kw,params_dtype=torch.float16,disable_tp=True)
 with set_current_vllm_config(cfg):block=modelmod.Qwen3NextSparseMoeBlock(modelcfg,prefix='model.language_model.layers.0.mlp')
 layer=block.experts
 assert overlay.require_complete()[0] is layer.quant_method
 assert len(cpu_overlay.require_constructed())==2
 state['shared_projection_methods']=sorted(cpu_overlay.require_constructed())
 assert not layer._parameters
 state['constructed_block']=type(block).__name__
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
 closed=session.close();assert all(c['exit_code']==0 and c['bye_execs']==0 for c in closed['close'])
 try:overlay.require_complete()
 except ResidentAdmissionError:pass
 else:raise AssertionError('closed session admitted')
 state.update(status='SUCCESS',closed_refused=True,zero_expert_parameters=True,cuda_initialized=torch.cuda.is_initialized())
except BaseException as e:
 state.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print('VF_RECEIPT '+json.dumps(state,sort_keys=True),flush=True);sys.exit(0 if state['status']=='SUCCESS' else 1)
