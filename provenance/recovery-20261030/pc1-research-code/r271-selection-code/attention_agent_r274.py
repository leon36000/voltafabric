"""Private PC1 CPU stage: no cache commit until coordinator acknowledges experts."""
import contextlib,json,os,sys,torch
with contextlib.redirect_stdout(sys.stderr):
 from safetensors import safe_open
 import vllm.platforms as platforms
 from vllm.platforms.cpu import CpuPlatform
 platforms._current_platform=CpuPlatform()
 from vllm.config import VllmConfig,set_current_vllm_config
 from adapters.qwen_full_attention_cpu import QwenFullAttentionCpuReference
 from adapters.qwen_kv_transaction import TransactionalKvAttention
 torch.set_num_threads(4)
 source=os.environ['MODEL_SHARD'];prefix='model.language_model.layers.3.'
 with safe_open(source,framework='pt',device='cpu') as f:
  w={k[len(prefix+'self_attn.'):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix+'self_attn.')};gate=f.get_tensor(prefix+'mlp.gate.weight').float();n1=f.get_tensor(prefix+'input_layernorm.weight').float();n2=f.get_tensor(prefix+'post_attention_layernorm.weight').float();shared={k[len(prefix+'mlp.shared_expert.'):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix+'mlp.shared_expert.')};shared_gate=f.get_tensor(prefix+'mlp.shared_expert_gate.weight').half()
 ctx=set_current_vllm_config(VllmConfig());ctx.__enter__();ref=QwenFullAttentionCpuReference(w,max_tokens=4)
lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
def decode(name):
 codes=shared[name+'.weight']; scales=shared[name+'.weight_scale'].float(); factor=shared[name+'.weight_scale_2'].float(); rows,columns=codes.shape
 assert codes.dtype==torch.uint8 and scales.shape==(rows,columns//8) and torch.isfinite(scales).all() and torch.isfinite(factor).all() and (scales>=0).all() and factor>0
 ix=torch.stack((codes&15,codes>>4),-1).reshape(rows,columns*2).long();return (lut[ix]*scales.repeat_interleave(16,1)*factor).half().double()
gw,uw,dw=(decode(n) for n in ('gate_proj','up_proj','down_proj'))
identity=json.loads(sys.stdin.readline());attention=TransactionalKvAttention(ref,identity)
torch.manual_seed(271);inputs=(torch.randn(2,2048)*0.1).half()
def rms(x,w):
 z=x.float();return (z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*(1+w)).half()
def emit(v):print(json.dumps(v,allow_nan=False,separators=(',',':')),flush=True)
emit({'state':'ready'})
for b in range(2):
 cmd=json.loads(sys.stdin.readline());assert cmd=={'op':'stage','batch':b}
 x=inputs[b:b+1];hidden=rms(x,n1);output=torch.empty_like(hidden);attention.begin(dict(identity,batch=b,attempt=0));attention(hidden,output);res=(x.float()+output.float()).half();moe=rms(res,n2);logits=(moe.float()@gate.T).half().float();vals,ids=torch.topk(logits,8,dim=-1);weights=torch.softmax(vals,dim=-1)
 emit({'state':'staged','batch':b,'x':moe.tolist(),'ids':ids.tolist(),'weights':weights.tolist()})
 try:
  cmd=json.loads(sys.stdin.readline())
  if set(cmd)!={'op','batch','routed'} or cmd['op']!='commit' or cmd['batch']!=b:attention.mark_unknown();raise RuntimeError('collector not confirmed')
  routed=torch.tensor(cmd['routed'],dtype=torch.float16);assert routed.shape==(1,2048) and torch.isfinite(routed).all()
  gx=(moe.double()@gw.T).half();ux=(moe.double()@uw.T).half();mid=(torch.nn.functional.silu(gx)*ux).half();shared_out=(mid.double()@dw.T).half()*torch.sigmoid(torch.nn.functional.linear(moe,shared_gate));combined=(routed+shared_out).half();layer_out=(res+combined).half()
  assert torch.isfinite(layer_out).all();attention.commit();emit({'state':'committed','batch':b,'committed':attention.committed_batch,'layer_output':layer_out.tolist(),'shared_output':shared_out.tolist()})
 except BaseException:
  # EOF/invalid command is ambiguous: no future replay in this process.
  try:attention.mark_unknown()
  except BaseException:pass
  raise
emit({'state':'closed','committed':attention.committed_batch})
