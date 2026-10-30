import json, os, torch
from safetensors import safe_open
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform=CpuPlatform()
from vllm.config import VllmConfig,set_current_vllm_config
from adapters.qwen_full_attention_cpu import QwenFullAttentionCpuReference
from adapters.qwen_kv_transaction import TransactionalKvAttention,KvTransactionError
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding,apply_rotary_pos_emb

torch.set_num_threads(4)
ctx=set_current_vllm_config(VllmConfig());ctx.__enter__()
prefix='model.language_model.layers.3.self_attn.'
with safe_open(os.environ['MODEL_SHARD'],framework='pt',device='cpu') as f:
 w={k[len(prefix):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
ref=QwenFullAttentionCpuReference(w,max_tokens=4)
base={'run_id':'r270-cpu','model_revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5','plan_id':'r270-cpu-only','layer':3,'epoch':0}
def ident(batch,attempt): return dict(base,batch=batch,attempt=attempt)
torch.manual_seed(270)
x=(torch.randn(2,2048)*0.1).half()
a=TransactionalKvAttention(ref,base)
def run(batch,attempt,row):
 a.begin(ident(batch,attempt));o=torch.empty_like(row);a(row,o);return o
expected0,c0=ref(x[0:1].float().unsqueeze(0));expected1,c1=ref(x[1:2].float().unsqueeze(0),c0)
o0=run(0,0,x[0:1]); assert torch.equal(o0,expected0.squeeze(0).half());a.commit();assert a.committed_batch==1
aborted=run(1,0,x[1:2]);a.abort();assert a.committed_batch==1
refused=0
try:a.begin(ident(1,0))
except KvTransactionError:refused+=1
retry=run(1,1,x[1:2]);assert torch.equal(aborted,retry) and torch.equal(retry,expected1.squeeze(0).half());a.commit();assert a.committed_batch==2
try:a.begin(dict(ident(2,0),layer=4))
except KvTransactionError:refused+=1
try:a.begin(ident(1,2))
except KvTransactionError:refused+=1
a.begin(ident(2,0));a.mark_unknown()
try:a.begin(ident(2,1))
except KvTransactionError:refused+=1
assert refused==4
# Independent Transformers default RoPE frequencies for text, checked beyond position 0.
c=Qwen3NextConfig(hidden_size=2048,num_attention_heads=16,num_key_value_heads=2,head_dim=256,rope_theta=10_000_000.0,partial_rotary_factor=0.25)
r=Qwen3NextRotaryEmbedding(c); pos=torch.tensor([[0,1,17,1023]])
q=torch.randn(1,16,4,256);k=torch.randn(1,2,4,256)
cos,sin=r(q,pos);qr,kr=apply_rotary_pos_emb(q,k,cos,sin)
rope_error=max((ref._rope(q,pos[0])-qr).abs().max().item(),(ref._rope(k,pos[0])-kr).abs().max().item())
assert rope_error < 1e-6,rope_error
print(json.dumps({'status':'PASS','committed_batches':2,'aborted_retry_equal':True,'refusals':refused,'rope_max_abs':rope_error,'gpu':False}))
