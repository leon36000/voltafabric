import os,sys,json,pathlib,hashlib,traceback
root=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r195-onecat-expert-filter-20261028-r1');root.mkdir(mode=0o700);(root/'cache').mkdir();os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',VLLM_CACHE_ROOT=str(root/'cache'),XDG_CACHE_HOME=str(root/'cache'))
s={'operation_id':'r195-onecat-expert-filter-20261028-r1','status':'STARTED','gpu_execution':False,'cpu_platform_substituted':True}
try:
 import torch,vllm.platforms as platforms
 from vllm.platforms.cpu import CpuPlatform
 platforms._current_platform=CpuPlatform()
 from safetensors.torch import save_file
 from vllm.model_executor.model_loader import weight_utils as u
 p=root/'fixture.safetensors'; values={'model.layers.0.mlp.experts.0.gate_proj.weight':torch.arange(6,dtype=torch.float32),'model.layers.0.mlp.experts.0.gate_proj.weight_scale':torch.ones(2),'model.layers.0.mlp.experts.0.gate_proj.weight_scale_2':torch.ones(1),'model.layers.0.mlp.experts.1.down_proj.weight':torch.ones(4),'model.layers.0.mlp.shared_expert.gate_proj.weight':torch.ones(3),'model.layers.0.self_attn.q_proj.weight':torch.ones(5),'model.layers.0.mlp.experts.gate_up_proj.weight':torch.ones(7)};save_file(values,str(p));orig=u.safe_open;reads=[]
 class TraceOpen:
  def __init__(self,*a,**kw):self.ctx=orig(*a,**kw)
  def __enter__(self):self.h=self.ctx.__enter__();return self
  def __exit__(self,*a):return self.ctx.__exit__(*a)
  def keys(self):return self.h.keys()
  def get_tensor(self,name):reads.append(name);return self.h.get_tensor(name)
  def metadata(self):return self.h.metadata()
 u.safe_open=TraceOpen
 empty=dict(u.safetensors_weights_iterator([str(p)],False,safetensors_load_strategy='lazy',local_expert_ids=set()));zero_reads=reads[:];reads.clear()
 only0=dict(u.safetensors_weights_iterator([str(p)],False,safetensors_load_strategy='lazy',local_expert_ids={0}));selected_reads=reads[:]
 excluded={k for k in values if '.experts.0.gate_proj.weight'==k[-len('.experts.0.gate_proj.weight'):] or '.experts.1.down_proj.weight'==k[-len('.experts.1.down_proj.weight'): ]};assert set(empty)==set(values)-excluded;assert set(zero_reads)==set(empty);assert set(only0)==set(values)-{'model.layers.0.mlp.experts.1.down_proj.weight'};assert set(selected_reads)==set(only0);assert all(torch.equal(v,values[k]) for k,v in empty.items());assert not torch.cuda.is_initialized()
 s.update(status='SUCCESS',fixture_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),excluded=sorted(excluded),zero_reads=zero_reads,selected_reads=selected_reads,cuda_initialized=False)
except BaseException as e:
 s.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(s,sort_keys=True,indent=2)+'\n');print(json.dumps(s,sort_keys=True));sys.exit(0 if s['status']=='SUCCESS' else 1)
