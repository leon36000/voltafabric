"""Select actual layer-3 experts using the real attention and gate on CPU."""
import os,json,hashlib,pathlib,torch
from safetensors import safe_open
from safetensors.torch import save_file
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform=CpuPlatform()
from vllm.config import VllmConfig,set_current_vllm_config
from adapters.qwen_full_attention_cpu import QwenFullAttentionCpuReference
from adapters.qwen_kv_transaction import TransactionalKvAttention
from extract_layer import extract

torch.set_num_threads(4)
root=pathlib.Path(os.environ['OUTPUT']);root.mkdir(mode=0o700)
source=pathlib.Path(os.environ['MODEL_SHARD']);prefix='model.language_model.layers.3.'
with safe_open(str(source),framework='pt',device='cpu') as f:
 w={k[len(prefix+'self_attn.'):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix+'self_attn.')}
 gate=f.get_tensor(prefix+'mlp.gate.weight').float()
 norm1=f.get_tensor(prefix+'input_layernorm.weight').float()
 norm2=f.get_tensor(prefix+'post_attention_layernorm.weight').float()
assert gate.shape==(256,2048) and norm1.shape==norm2.shape==(2048,)
ctx=set_current_vllm_config(VllmConfig());ctx.__enter__();ref=QwenFullAttentionCpuReference(w,max_tokens=4)
identity={'run_id':'r271-selection-cpu','model_revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5','plan_id':'r271-selection','layer':3,'epoch':0}
attn=TransactionalKvAttention(ref,identity)
torch.manual_seed(271);inputs=(torch.randn(2,2048)*0.1).half();selected=set();capture={};routes=[]
def rms(x,w):
 z=x.float();return (z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*(1+w)).half()
for b in range(2):
 x=inputs[b:b+1];n1=rms(x,norm1);out=torch.empty_like(n1);attn.begin(dict(identity,batch=b,attempt=0));attn(n1,out);res=(x.float()+out.float()).half();moe=rms(res,norm2)
 logits=(moe.float()@gate.T).half().float();vals,ids=torch.topk(logits,8,dim=-1);probs=torch.softmax(vals,dim=-1)
 ids=ids[0].tolist();probs=probs[0].tolist();selected.update(ids);routes.append({'ids':ids,'weights':probs});capture.update({f'batch{b}.input':moe,f'batch{b}.hidden':x,f'batch{b}.attention':out});attn.commit()
assert attn.committed_batch==2 and 8<=len(selected)<=16
save_file(capture,str(root/'capture.safetensors'))
result=extract(source,'07141c2db92e47bc08777132cd1a0323faf300eab3a7d7c111bc2bf075fda050',root/'experts.safetensors',sorted(selected),3)
# Compare selected tensor bytes against pinned source independently of extractor.
with safe_open(str(source),framework='pt',device='cpu') as src,safe_open(str(root/'experts.safetensors'),framework='pt',device='cpu') as dst:
 names=dst.keys(); assert len(names)==12*len(selected)
 for name in names: assert torch.equal(src.get_tensor(name),dst.get_tensor(name)),name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result.update(status='PASS',routes=routes,capture_sha256=sha(root/'capture.safetensors'),capture_inputs='synthetic',gpu=False,verified_tensors=len(names))
(root/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result),flush=True)
