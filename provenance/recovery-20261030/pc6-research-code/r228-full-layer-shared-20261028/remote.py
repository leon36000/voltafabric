import os,sys,pathlib,json,hashlib,base64,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r228-full-layer-shared-20261028-r1';root.mkdir(mode=0o700);sys.path.insert(0,str(base/'r226-package'));state={'status':'STARTED','operation_id':'r228-full-layer-shared-20261028-r1','gpu':False}
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
 p=json.loads((base/'r228-input-r1.json').read_bytes());stamp=p['routing']['stamp'];session=DurableSession(base/'r228-proxy-r1.py',p,root/'ledger',deadline_seconds=180);state.update(run_id=stamp['run_id'],ledger=str(session.directory),stage='SESSION_READY');(root/'progress.json').write_text(json.dumps(state)+'\n')
 class Quant:
  def get_quant_method(self,layer,prefix):return ResidentMoEMethod(layer.moe_config,session,model_revision=stamp['model_revision'],plan_id=stamp['plan_id'],layer_id=stamp['layer'],epoch=stamp['epoch'],layer_name=prefix)
 def cpu_router(hidden_states,gating_output,topk,renormalize):
  assert topk==8 and renormalize
  prob=torch.softmax(gating_output.float(),dim=-1);weights,ids=torch.topk(prob,8,dim=-1);weights=weights/weights.sum(dim=-1,keepdim=True);state['route_ids']=ids[0].tolist();state['route_weights']=weights[0].tolist()
  return weights,ids.to(torch.int32)
 from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
 from safetensors import safe_open
 path=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r204-real-experts-20261028/shared-r208.safetensors'); assert hashlib.sha256(path.read_bytes()).hexdigest()=='6e83b12b99e891ce4eebd9adfb95960e7b40ae8d48811cd5499caa3ee6e09934';torch.set_num_threads(1)
 prefix='model.language_model.layers.0.mlp.shared_expert.'
 with safe_open(str(path),framework='pt',device='cpu') as f:
  def decode(proj):
   c=f.get_tensor(prefix+proj+'.weight');s=f.get_tensor(prefix+proj+'.weight_scale').float();g=f.get_tensor(prefix+proj+'.weight_scale_2').float();lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]);n,k=c.shape;ids=torch.stack((c&15,c>>4),dim=-1).reshape(n,k*2).long();return (lut[ids]*s.repeat_interleave(16,dim=1)*g).to(torch.float16).to(torch.float64)
  gate,up,down=[decode(k) for k in ('gate_proj','up_proj','down_proj')];linear_gate=f.get_tensor('model.language_model.layers.0.mlp.shared_expert_gate.weight').to(torch.float16)
 class Project(torch.nn.Module):
  def __init__(self,w):super().__init__();self.w=w
  def forward(self,x):return torch.matmul(x.to(torch.float64),self.w.T).to(torch.float16),None
 class Act(torch.nn.Module):
  def forward(self,x):a,b=x.chunk(2,dim=-1);return (torch.nn.functional.silu(a)*b).to(torch.float16)
 class SharedGate(torch.nn.Module):
  def forward(self,x):return torch.nn.functional.linear(x,linear_gate),None
 instance=object.__new__(Qwen2MoeMLP);torch.nn.Module.__init__(instance);instance.layer_idx=0;instance.gate_up_proj=Project(torch.cat([gate,up]));instance.down_proj=Project(down);instance.act_fn=Act();instance.expert_gate=SharedGate()
 instance._vllm_disable_shared_experts_stream=True
 cfg=VllmConfig()
 with set_current_vllm_config(cfg):layer=FusedMoE(num_experts=256,top_k=8,hidden_size=2048,intermediate_size=512,params_dtype=torch.float16,tp_size=1,dp_size=1,pcp_size=1,prefix='model.layers.0.mlp.experts',quant_config=Quant(),custom_routing_function=cpu_router,shared_experts=instance)
 assert not any(name.startswith(('w13_weight','w2_weight')) for name,_ in layer.named_parameters())
 x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);ids=torch.tensor([[5,0,7,2,4,1,6,3]]);weights=torch.tensor([[.125]*8]);outputs=[]
 from vllm.forward_context import ForwardContext,override_forward_context
 from safetensors import safe_open
 gate_path=base/'r204-real-experts-20261028/gate-r206.safetensors'
 assert hashlib.sha256(gate_path.read_bytes()).hexdigest()=='7cf4a08f30c939094008eeddedfdfbf43d1f8b780f742e02d45f7dc566fc25a5'
 with safe_open(str(gate_path),framework='pt',device='cpu') as f:gate=f.get_tensor('model.language_model.layers.0.mlp.gate.weight').to(torch.float16)
 logits=torch.nn.functional.linear(x,gate)
 context=ForwardContext(no_compile_layers={layer.layer_name:layer},attn_metadata={},slot_mapping={})
 with override_forward_context(context):
  for _ in range(2):
   out=layer(x,logits);outputs.append(base64.b64encode(out.numpy().tobytes()).decode())
 closed=session.close();expected_ranks={e//8 for e in state['route_ids']};assert len(closed['close'])==32 and all(c['exit_code']==0 and c['bye_execs']==(2 if rank in expected_ranks else 0) for rank,c in enumerate(closed['close']));state['selected_ranks']=sorted(expected_ranks);state['owner_exec_counts']=[c['bye_execs'] for c in closed['close']]
 try:layer.ensure_moe_quant_config_init()
 except ResidentAdmissionError:pass
 else:raise AssertionError('closed session admitted')
 state.update(status='SUCCESS',outputs=outputs,closed_refused=True,zero_expert_parameters=True,cuda_initialized=torch.cuda.is_initialized())
except BaseException as e:
 state.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print('VF_RECEIPT '+json.dumps(state,sort_keys=True),flush=True);sys.exit(0 if state['status']=='SUCCESS' else 1)
