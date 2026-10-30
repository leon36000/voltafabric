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
  w={k[len(prefix+'self_attn.'):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix+'self_attn.')};gate=f.get_tensor(prefix+'mlp.gate.weight').float();n1=f.get_tensor(prefix+'input_layernorm.weight').float();n2=f.get_tensor(prefix+'post_attention_layernorm.weight').float()
 ctx=set_current_vllm_config(VllmConfig());ctx.__enter__();ref=QwenFullAttentionCpuReference(w,max_tokens=4)
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
  if cmd!={'op':'commit','batch':b}:attention.mark_unknown();raise RuntimeError('collector not confirmed')
  attention.commit();emit({'state':'committed','batch':b,'committed':attention.committed_batch})
 except BaseException:
  # EOF/invalid command is ambiguous: no future replay in this process.
  try:attention.mark_unknown()
  except BaseException:pass
  raise
emit({'state':'closed','committed':attention.committed_batch})
