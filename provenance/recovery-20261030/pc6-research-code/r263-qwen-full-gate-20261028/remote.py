import os,sys,pathlib,json,hashlib,base64,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r263-qwen-full-gate-20261028-r1';root.mkdir(mode=0o700);sys.path.insert(0,str(base/'r250-package'));state={'status':'STARTED','operation_id':'r263-qwen-full-gate-20261028-r1','gpu':False}
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
 p=json.loads((base/'r263-input-r1.json').read_bytes());stamp=p['routing']['stamp'];session=DurableSession(base/'r263-proxy-r1.py',p,root/'ledger',deadline_seconds=120);state.update(run_id=stamp['run_id'],ledger=str(session.directory),stage='SESSION_READY');(root/'progress.json').write_text(json.dumps(state)+'\n')
 import types
 from adapters.onecat_quant_overlay import ResidentQuantOverlay
 import vllm.model_executor.models.qwen3_next as modelmod
 from vllm.transformers_utils.config import get_config
 checkpoint=base/'r188-qwen-checkpoint-pc1-20261028-r1/model'
 hf=get_config(str(checkpoint),trust_remote_code=False,config_format='hf',local_files_only=True)
 hf.text_config.num_experts=256
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
 import vllm.model_executor.layers.fused_moe.layer as fused_layer
 fused_layer.get_tensor_model_parallel_world_size=lambda:1
 fused_layer.get_dp_group=lambda:types.SimpleNamespace(world_size=1)
 fused_layer.get_pcp_group=lambda:types.SimpleNamespace(world_size=1)
 modelmod.get_tensor_model_parallel_world_size=lambda:1
 modelmod.get_ep_group=lambda:Group()
 modelmod.get_current_vllm_config=lambda:modelcfg
 import vllm.model_executor.layers.linear as linearmod
 linearmod.get_tensor_model_parallel_world_size=lambda:1
 linearmod.get_tensor_model_parallel_rank=lambda:0
 from vllm.model_executor.layers.linear import ReplicatedLinear
 modelmod.ReplicatedLinear=lambda *args,**kw:ReplicatedLinear(*args,**kw,params_dtype=torch.float16,disable_tp=True)
 import vllm.model_executor.models.qwen3_5 as qwenmod
 hf.text_config.num_hidden_layers=1;hf.text_config.layer_types=hf.text_config.layer_types[:1];hf.text_config.vocab_size=16
 modelcfg.cache_config=cfg.cache_config
 modelcfg.compilation_config=cfg.compilation_config
 from vllm.config import CompilationMode
 modelcfg.compilation_config.mode=CompilationMode.NONE
 from safetensors import safe_open as checkpoint_open
 from adapters.qwen_gdn_attention_cpu import QwenGdnCpuReference
 from adapters.qwen_gdn_transaction import TransactionalGdnAttention
 gdn_names=['in_proj_qkv.weight','in_proj_qkv.weight_scale','in_proj_qkv.input_scale','in_proj_z.weight','in_proj_z.weight_scale','in_proj_z.input_scale','out_proj.weight','out_proj.weight_scale','out_proj.input_scale','in_proj_a.weight','in_proj_b.weight','conv1d.weight','A_log','dt_bias','norm.weight']
 with checkpoint_open(str(checkpoint/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:
  gdn_weights={name:f.get_tensor('model.language_model.layers.0.linear_attn.'+name).contiguous() for name in gdn_names}
 with set_current_vllm_config(cfg):gdn_ref=QwenGdnCpuReference(gdn_weights)
 identity={k:stamp[k] for k in ('run_id','model_revision','plan_id','layer','epoch')}
 def real_attention(**kw):return TransactionalGdnAttention(gdn_ref,identity)
 qwenmod.Qwen3_5GatedDeltaNet=real_attention
 qwenmod.VocabParallelEmbedding=lambda count,hidden:torch.nn.Embedding(count,hidden,dtype=torch.float16)
 qwenmod.get_pp_group=lambda:types.SimpleNamespace(is_first_rank=True,is_last_rank=True)
 def make_one(num,fn,prefix):
  assert num==1
  return 0,1,torch.nn.ModuleList([fn(prefix=prefix+'.0')])
 qwenmod.make_layers=make_one
 with set_current_vllm_config(cfg):shell=qwenmod.Qwen3_5Model(vllm_config=modelcfg,prefix='model.language_model')
 block=shell.layers[0].mlp
 state['model_constructor']=type(shell).__name__;state['decoder_constructor']=type(shell.layers[0]).__name__
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
 fixture=base/'r262-full-model-fixture-r1';assert hashlib.sha256((fixture/'nonexpert.safetensors').read_bytes()).hexdigest()=='35c4d00d2dcbc09db0ac89494ceee34820b2ca52d0805b9ac0959418113f4f15'
 class ModelWrapper(torch.nn.Module):
  def __init__(self):super().__init__();self.qwen=shell;self.names=[]
  def load_weights(self,values):
   def normalized():
    for name,tensor in values:
     self.names.append(name);prefix='model.language_model.'
     if not name.startswith(prefix):raise ValueError('unexpected fixture prefix')
     yield name[len(prefix):],tensor
   return self.qwen.load_weights(normalized())
 model=ModelWrapper();model_config=types.SimpleNamespace(model=str(fixture),revision=None,quantization='modelopt',is_moe=False)
 loader=DefaultModelLoader(LoadConfig(load_format='safetensors',use_tqdm_on_load=False));fil=ResidentPreReadFilter({0:layer.quant_method},ResidentMoEMethod,opener=TrackedOpen)
 with set_current_vllm_config(cfg):fil.load(loader,model_config,model)
 assert len(fil.seen[0])==3072 and fil.finished and len(model.names)==14 and session.proc.poll() is None
 assert sorted(reads)==sorted(model.names) and len(reads)==14
 state['materialized_names']=sorted(reads);state['loader_filtered']=len(fil.seen[0]);state['loader_session_alive']=True;state['qwen_weight_loader']=True
 for target in (block.shared_expert.gate_up_proj,block.shared_expert.down_proj,block.gate,block.shared_expert_gate):target.quant_method.process_weights_after_loading(target)
 from safetensors.torch import load_file
 actual_norms=load_file(str(fixture/'nonexpert.safetensors'))
 norm_pairs=[('layers.0.input_layernorm.weight',shell.layers[0].input_layernorm.weight),('layers.0.post_attention_layernorm.weight',shell.layers[0].post_attention_layernorm.weight),('norm.weight',shell.norm.weight)]
 for name,param in norm_pairs:
  expected=actual_norms['model.language_model.'+name]
  assert torch.equal(param.detach().float(),expected.float()),name
 state['verified_checkpoint_norms']=[name for name,_ in norm_pairs]
 captured=[]
 class Router:
  def select_experts(self,hidden_states,router_logits,input_ids):
   probs=torch.softmax(router_logits.float(),dim=-1)
   weights,ids=torch.topk(probs,8,dim=-1)
   if getattr(hf.text_config,'norm_topk_prob',True):weights=weights/weights.sum(dim=-1,keepdim=True)
   captured.append({'input':hidden_states.detach().clone(),'logits':router_logits.detach().clone(),'weights':weights.detach().clone(),'ids':ids.detach().clone()})
   return weights,ids.to(torch.int32)
 layer.runner.router=Router()
 x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16)
 from vllm.forward_context import ForwardContext,override_forward_context
 context=ForwardContext(no_compile_layers={layer.layer_name:layer},attn_metadata={},slot_mapping={})
 outputs=[];output_tensors=[];attention=shell.layers[0].linear_attn
 with set_current_vllm_config(cfg),override_forward_context(context),torch.inference_mode():
  for batch in range(2):
   attention.begin({**identity,'batch':batch,'attempt':1})
   try:
    model_out=shell(input_ids=None,positions=torch.tensor([batch]),inputs_embeds=x)
    assert model_out.shape==x.shape and torch.isfinite(model_out).all()
    outputs.append(hashlib.sha256(model_out.contiguous().numpy().tobytes()).hexdigest());output_tensors.append(model_out.detach().clone())
    attention.commit()
   except BaseException:
    attention.mark_unknown();raise
 assert attention.committed_batch==2 and len(captured)==2
 from safetensors.torch import save_file
 save_file({**{f'batch{i}.output':v for i,v in enumerate(output_tensors)},**{f'batch{i}.{k}':v for i,row in enumerate(captured) for k,v in row.items()}},str(root/'capture.safetensors'))
 state['capture_sha256']=hashlib.sha256((root/'capture.safetensors').read_bytes()).hexdigest();state['router']='full_256_checkpoint_gate_rows_softmax_topk';state['route_ids']=[v['ids'].tolist() for v in captured];state['route_weights']=[v['weights'].tolist() for v in captured]
 state['model_fixture_output_sha256']=outputs;state['model_fixture_forward']=True;state['gdn_real_weights']=True;state['attention_committed_batches']=2;state['shared_weights_loaded']=True
 closed=session.close();assert len(closed['close'])==16 and all(c['exit_code']==0 for c in closed['close']) and sum(c['bye_execs'] for c in closed['close'])>0
 try:overlay.require_complete()
 except ResidentAdmissionError:pass
 else:raise AssertionError('closed session admitted')
 state.update(status='SUCCESS',closed_refused=True,zero_expert_parameters=True,cuda_initialized=torch.cuda.is_initialized())
except BaseException as e:
 state.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print('VF_RECEIPT '+json.dumps(state,sort_keys=True),flush=True);sys.exit(0 if state['status']=='SUCCESS' else 1)
