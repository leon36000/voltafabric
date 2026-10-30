import os,sys,pathlib,json,hashlib,base64,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r292-sixteen-layer-model-r1';root.mkdir(mode=0o700);sys.path.insert(0,str(base/'r276-package-r1'));sys.path.insert(0,str(base));state={'status':'STARTED','operation_id':'r292-sixteen-layer-model-r1','gpu':False}
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
 import uuid,copy,subprocess
 code=base/'r292-code-r1';receiver=root/'receiver';receiver.mkdir();daemons={};sockets={};logs=[]
 approvals={}
 for idx in range(16):
  folder=(base/'r283-four-layer-owners-r1'/f'layer{idx}') if idx<4 else (base/f'r284-layer{idx}-owners-r1' if idx<8 else (base/f'r289-layer{idx}-owners-r1' if idx==8 else (base/f'r290-layer{idx}-owners-r1' if idx<12 else base/f'r292-layer{idx}-owners-r1')))
  manifest=json.loads((folder/'manifest.json').read_text())
  for rank in range(4):
   a=manifest['ranks'][str(rank)];pinned=pathlib.Path(a['path']);approvals[f'{idx}:{rank}']={'path':str(pinned),'sha256':a['sha256'],'bytes':pinned.stat().st_size}
 (root/'reuse-map.json').write_text(json.dumps(approvals,sort_keys=True)+'\n')
 for rank in range(4):
  sock=f'/run/media/pc1/Infrastructure/Naos/.r292-rank{rank}-r1.sock';sockets[rank]=sock;err=(root/f'daemon{rank}.stderr').open('wb');logs.append(err)
  proc=subprocess.Popen([sys.executable,'-B',str(code/'multiplex.py'),sock,str(receiver),'16'],stdout=subprocess.PIPE,stderr=err,bufsize=0);daemons[rank]=proc
  if proc.stdout.readline()!=b'READY\n':raise RuntimeError('multiplexer unavailable')
 state['daemon_pids']={i:p.pid for i,p in daemons.items()};(root/'daemon-pids.json').write_text(json.dumps(state['daemon_pids'])+'\n')
 p=json.loads((base/'r281-layer1-session-r1/input.json').read_text());run='r292-'+uuid.uuid4().hex;sessions={};identities={}
 for idx in range(16):
  q=copy.deepcopy(p);folder=(base/'r283-four-layer-owners-r1'/f'layer{idx}') if idx<4 else (base/f'r284-layer{idx}-owners-r1' if idx<8 else (base/f'r289-layer{idx}-owners-r1' if idx==8 else (base/f'r290-layer{idx}-owners-r1' if idx<12 else base/f'r292-layer{idx}-owners-r1')));manifest=json.loads((folder/'manifest.json').read_text());q['artifacts']=[manifest['ranks'][str(i)]['path'] for i in range(4)];q['workers']=q['workers'][:4];q['routing']['registry']={'epoch':1,'copies':[{'expert':e,'rank':e//64,'local':e%64} for e in range(256)]};q['routing']['resident_order']={str(i):list(range(64)) for i in range(4)};q['routing']['stamp'].update(run_id=run,plan_id='r292-local-reuse',layer=idx,batch=0);
  for worker in q['workers']:worker['argv']=['/usr/bin/python3','-B',str(code/'proxy.py'),sockets[worker['rank']]]
  sessions[idx]=DurableSession(base/'r280-rust-x86-r1/target/debug/examples/r261_multipart_cpu_session',q,root/f'ledger{idx}',deadline_seconds=180)
  identities[idx]={k:q['routing']['stamp'][k] for k in ('run_id','model_revision','plan_id','layer','epoch')}
 stamp={**identities[0]};state.update(run_id=run,ledgers={i:str(x.directory) for i,x in sessions.items()},stage='SESSIONS_READY');(root/'progress.json').write_text(json.dumps(state)+'\n')
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
 overlay=ResidentQuantOverlay(original,sessions,model_revision=stamp['model_revision'],plan_id=stamp['plan_id'],epoch=stamp['epoch'])
 overlay.packed_modules_mapping=Qwen3_5MoeForConditionalGeneration.packed_modules_mapping
 assert original._resolve_quant_algo('model.language_model.layers.0.mlp.shared_expert.gate_up_proj')=='W4A16_NVFP4'
 from adapters.onecat_cpu_shared_overlay import CpuSharedOverlay
 cpu_overlay=CpuSharedOverlay(overlay,list(range(16)))
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
 hf.text_config.num_hidden_layers=16;hf.text_config.layer_types=hf.text_config.layer_types[:16];hf.text_config.vocab_size=16
 modelcfg.cache_config=cfg.cache_config
 modelcfg.compilation_config=cfg.compilation_config
 from vllm.config import CompilationMode
 modelcfg.compilation_config.mode=CompilationMode.NONE
 from safetensors import safe_open as checkpoint_open
 from adapters.qwen_gdn_attention_cpu import QwenGdnCpuReference
 from adapters.qwen_gdn_transaction import TransactionalGdnAttention
 gdn_names=['in_proj_qkv.weight','in_proj_qkv.weight_scale','in_proj_qkv.input_scale','in_proj_z.weight','in_proj_z.weight_scale','in_proj_z.input_scale','out_proj.weight','out_proj.weight_scale','out_proj.input_scale','in_proj_a.weight','in_proj_b.weight','conv1d.weight','A_log','dt_bias','norm.weight']
 gdn_refs={}
 with checkpoint_open(str(checkpoint/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:
  for i in (0,1,2,4,5,6,8,9,10,12,13,14):
   weights={name:f.get_tensor(f'model.language_model.layers.{i}.linear_attn.'+name).contiguous() for name in gdn_names}
   with set_current_vllm_config(cfg):gdn_refs[i]=QwenGdnCpuReference(weights)
 def real_attention(**kw):
  idx=int(kw['prefix'].split('.layers.')[1].split('.')[0]);return TransactionalGdnAttention(gdn_refs[idx],identities[idx])
 qwenmod.Qwen3_5GatedDeltaNet=real_attention
 from adapters.qwen_full_attention_cpu import QwenFullAttentionCpuReference
 from adapters.qwen_kv_transaction import TransactionalKvAttention
 full_refs={}
 with checkpoint_open(str(checkpoint/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:
  for idx in (3,7,11,15):
   pre=f'model.language_model.layers.{idx}.self_attn.';w={k[len(pre):]:f.get_tensor(k).contiguous() for k in f.keys() if k.startswith(pre)}
   with set_current_vllm_config(cfg):full_refs[idx]=QwenFullAttentionCpuReference(w)
 class Attention(TransactionalKvAttention):
  def forward(self,hidden_states,output,positions=None):
   expected=0 if self._cache is None else self._cache[0].shape[2]
   if positions is None or positions.tolist()!=list(range(expected,expected+hidden_states.shape[0])):raise ValueError('position mismatch')
   return super().forward(hidden_states,output)
 def full_attention(*args,**kw):
  idx=int(kw['prefix'].split('.layers.')[1].split('.')[0]);return Attention(full_refs[idx],identities[idx])
 qwenmod.Qwen3NextAttention=full_attention
 qwenmod.VocabParallelEmbedding=lambda count,hidden:torch.nn.Embedding(count,hidden,dtype=torch.float16)
 qwenmod.get_pp_group=lambda:types.SimpleNamespace(is_first_rank=True,is_last_rank=True)
 def make_one(num,fn,prefix):
  assert num==16
  return 0,16,torch.nn.ModuleList([fn(prefix=prefix+'.'+str(i)) for i in range(16)])
 qwenmod.make_layers=make_one
 with set_current_vllm_config(cfg):shell=qwenmod.Qwen3_5Model(vllm_config=modelcfg,prefix='model.language_model')
 blocks=[shell.layers[i].mlp for i in range(16)];block=blocks[0]
 state['model_constructor']=type(shell).__name__;state['decoder_constructor']=type(shell.layers[0]).__name__
 layers=[b.experts for b in blocks];layer=layers[0]
 assert all(overlay.require_complete()[i] is layers[i].quant_method for i in range(16))
 assert len(cpu_overlay.require_constructed())==32
 state['shared_projection_methods']=sorted(cpu_overlay.require_constructed())
 assert all(not l._parameters for l in layers)
 state['constructed_block']=type(block).__name__
 from adapters.onecat_pre_read import ResidentPreReadFilter
 from safetensors.torch import safe_open
 reads=[]
 class TrackedOpen:
  def __init__(self,*a,**k):self.is_source=pathlib.Path(a[0]).name=='expert_source.safetensors';self.ctx=safe_open(*a,**k)
  def __enter__(self):self.f=self.ctx.__enter__();return self
  def __exit__(self,*a):return self.ctx.__exit__(*a)
  def keys(self):
   keys=self.f.keys()
   return [k for k in keys if any(k.startswith(f'model.language_model.layers.{i}.mlp.experts.') for i in range(16))] if self.is_source else keys
  def get_tensor(self,name):reads.append(name);return self.f.get_tensor(name)
 from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
 from vllm.config import LoadConfig
 import types
 fixture=base/'r292-sixteen-layer-fixture-r1';assert hashlib.sha256((fixture/'nonexpert.safetensors').read_bytes()).hexdigest()=='f0c94c20c0a5e08c9401ed4bad637013707a7364a3e33399816d2483dbb86127'
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
 loader=DefaultModelLoader(LoadConfig(load_format='safetensors',use_tqdm_on_load=False));fil=ResidentPreReadFilter({i:layers[i].quant_method for i in range(16)},ResidentMoEMethod,opener=TrackedOpen)
 with set_current_vllm_config(cfg):fil.load(loader,model_config,model)
 assert all(len(fil.seen[i])==3072 for i in range(16)) and fil.finished and len(model.names)==209 and all(session.proc.poll() is None for session in sessions.values())
 assert sorted(reads)==sorted(model.names) and len(reads)==209
 state['materialized_names']=sorted(reads);state['loader_filtered']=sum(len(fil.seen[i]) for i in range(16));state['loader_session_alive']=True;state['qwen_weight_loader']=True
 for block in blocks:
  for target in (block.shared_expert.gate_up_proj,block.shared_expert.down_proj,block.gate,block.shared_expert_gate):target.quant_method.process_weights_after_loading(target)
 from safetensors.torch import load_file
 actual_norms=load_file(str(fixture/'nonexpert.safetensors'))
 norm_pairs=[(f'layers.{i}.{name}.weight',getattr(shell.layers[i],name).weight) for i in range(16) for name in ('input_layernorm','post_attention_layernorm')]+[('norm.weight',shell.norm.weight)]
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
 for layer in layers:layer.runner.router=Router()
 from transformers import AutoTokenizer
 from qwen_cpu_head import greedy_nvfp4_head
 assert hashlib.sha256((base/'qwen_cpu_head.py').read_bytes()).hexdigest()=='786a6f2a136b82adfca81bf7a08b74ad6910af5fef82422b89e6120df35e0ae0'
 tokenizer=AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,trust_remote_code=False)
 tokens=tokenizer.encode('Bonjour',add_special_tokens=False);assert len(tokens)==1
 token_inputs=[];token_outputs=[];embedding_shard=checkpoint/'model-00001-of-00003.safetensors';head_shard=checkpoint/'model-00003-of-00003.safetensors'
 def embedding(token):
  with checkpoint_open(str(embedding_shard),framework='pt',device='cpu') as f:
   rows=f.get_slice('model.language_model.embed_tokens.weight');assert rows.get_shape()==[248320,2048]
   return rows[token:token+1].to(torch.float16)

 from vllm.forward_context import ForwardContext,override_forward_context
 context=ForwardContext(no_compile_layers={layer.layer_name:layer for layer in layers},attn_metadata={},slot_mapping={})
 outputs=[];output_tensors=[];attentions=[shell.layers[i].self_attn if i in (3,7,11,15) else shell.layers[i].linear_attn for i in range(16)]
 with set_current_vllm_config(cfg),override_forward_context(context),torch.inference_mode():
  for batch in range(2):
   token_inputs.append(int(tokens[-1]));x=embedding(tokens[-1]);[attentions[i].begin({**identities[i],'batch':batch,'attempt':1}) for i in range(16)]
   try:
    model_out=shell(input_ids=None,positions=torch.tensor([batch]),inputs_embeds=x)
    assert model_out.shape==x.shape and torch.isfinite(model_out).all()
    outputs.append(hashlib.sha256(model_out.contiguous().numpy().tobytes()).hexdigest());output_tensors.append(model_out.detach().clone())
    prediction=greedy_nvfp4_head(head_shard,model_out[0],vocab=248320,width=2048,chunk_rows=512);token_outputs.append(prediction);tokens.append(prediction['token_id']);[attn.commit() for attn in attentions]
   except BaseException:
    [attn.mark_unknown() for attn in attentions];raise
 assert all(attn.committed_batch==2 for attn in attentions) and len(captured)==32
 from safetensors.torch import save_file
 save_file({**{f'batch{i}.output':v for i,v in enumerate(output_tensors)},**{f'batch{i}.{k}':v for i,row in enumerate(captured) for k,v in row.items()}},str(root/'capture.safetensors'))
 state['capture_sha256']=hashlib.sha256((root/'capture.safetensors').read_bytes()).hexdigest();state['router']='full_256_checkpoint_gate_rows_softmax_topk';state['route_ids']=[v['ids'].tolist() for v in captured];state['route_weights']=[v['weights'].tolist() for v in captured]
 state['input_tokens']=token_inputs;state['predicted_tokens']=token_outputs;state['model_fixture_output_sha256']=outputs;state['model_fixture_forward']=True;state['gdn_real_weights']=True;state['attention_committed_batches']=[attn.committed_batch for attn in attentions];state['shared_weights_loaded']=True
 closed=[session.close() for session in sessions.values()];assert all(len(c['close'])==4 and all(v['exit_code']==0 for v in c['close']) for c in closed)
 for proc in daemons.values():assert proc.stdout.readline()==b'CLOSED\n' and proc.wait(timeout=30)==0
 for err in logs:err.close()
 assert len(list(receiver.glob('*/resident.json')))==64 and not list(receiver.glob('*/staged.bin')) and not list(receiver.glob('*/resident.bin'))
 state['receiver_copy_files']=0;state['transferred_bytes']=sum(a['bytes'] for a in approvals.values())
 state['daemon_exit_codes']={i:p.returncode for i,p in daemons.items()}
 try:overlay.require_complete()
 except ResidentAdmissionError:pass
 else:raise AssertionError('closed session admitted')
 state.update(status='SUCCESS',closed_refused=True,zero_expert_parameters=True,cuda_initialized=torch.cuda.is_initialized())
except BaseException as e:
 state.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print('VF_RECEIPT '+json.dumps(state,sort_keys=True),flush=True);sys.exit(0 if state['status']=='SUCCESS' else 1)
