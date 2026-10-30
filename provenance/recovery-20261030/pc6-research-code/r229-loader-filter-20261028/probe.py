import os,sys,pathlib,json,traceback,types
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r229-loader-filter-20261028-r1';root.mkdir(mode=0o700);sys.path.insert(0,str(base/'r229-package'));state={'operation_id':'r229-loader-filter-20261028-r1','status':'STARTED','gpu':False}
try:
 import torch,vllm.platforms as platforms
 from vllm.platforms.cpu import CpuPlatform
 platforms._current_platform=CpuPlatform()
 from safetensors.torch import save_file
 from vllm.config import LoadConfig,VllmConfig,set_current_vllm_config
 from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
 from adapters.session_bridge import DurableSession
 from adapters.onecat_resident_method import ResidentMoEMethod
 from adapters.onecat_weight_filter import ResidentWeightFilter,ResidentFilterError
 class Process:
  def poll(self):return None
 session=object.__new__(DurableSession);session.proc=Process();session.terminal=False;session.identity={'run_id':'fixture','model_revision':'m','plan_id':'p','layer':0,'epoch':1};session.experts=frozenset(range(8));session.hidden=2048
 method=ResidentMoEMethod(types.SimpleNamespace(num_experts=8),session,model_revision='m',plan_id='p',layer_id=0,epoch=1,layer_name='model.language_model.layers.0.mlp.experts');method.create_weights(types.SimpleNamespace(layer_name=method.layer_name),8,2048,512,torch.float16,global_num_experts=8)
 fields=[f'{proj}.{field}' for proj in ('gate_proj','up_proj','down_proj') for field in ('weight','weight_scale','weight_scale_2','input_scale')]
 def fixture(name,missing=False):
  path=root/name;path.mkdir();t={f'model.language_model.layers.0.mlp.experts.{e}.{field}':torch.tensor([e],dtype=torch.float32) for e in range(8) for field in fields}
  if missing:t.pop(next(iter(t)))
  t['model.language_model.layers.0.mlp.shared_expert.gate_proj.weight']=torch.tensor([2.]);t['model.language_model.layers.1.mlp.experts.0.gate_proj.weight']=torch.tensor([3.]);save_file(t,str(path/'model.safetensors'));return path
 class Model(torch.nn.Module):
  def __init__(self):super().__init__();self.names=[]
  def load_weights(self,values):
   for name,_ in values:self.names.append(name)
   return set(self.names)
 def attempt(path):
  loader=DefaultModelLoader(LoadConfig(load_format='safetensors',use_tqdm_on_load=False));cfg=types.SimpleNamespace(model=str(path),revision=None,quantization='modelopt',is_moe=False);model=Model();f=ResidentWeightFilter({0:method},ResidentMoEMethod)
  with set_current_vllm_config(VllmConfig()):f.load(loader,cfg,model)
  assert 'get_all_weights' not in vars(loader);return model.names
 names=attempt(fixture('positive'));assert sorted(names)==['model.language_model.layers.0.mlp.shared_expert.gate_proj.weight','model.language_model.layers.1.mlp.experts.0.gate_proj.weight']
 try:attempt(fixture('missing',True))
 except ResidentFilterError:missing_refused=True
 else:raise AssertionError('missing tensor accepted')
 state.update(status='SUCCESS',loader='DefaultModelLoader.load_weights',actual_iterator=True,fixture_session=True,filtered_tensors=96,retained_names=names,missing_refused=missing_refused,cuda_initialized=torch.cuda.is_initialized())
except BaseException as e:
 state.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print(json.dumps(state,sort_keys=True));sys.exit(0 if state['status']=='SUCCESS' else 1)
