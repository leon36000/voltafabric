"""Admit real checkpoint layer-0 full 256-expert gate and compare an independent scalar oracle."""
import hashlib,json,math,os,pathlib
os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file,save_file
torch.set_num_threads(1)
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');old=base/'r252-qwen-norm-live-20261028-r1';model=base/'r188-qwen-checkpoint-pc1-20261028-r1/model';r=json.loads((old/'receipt.json').read_text());capture=old/'capture.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert r['status']=='SUCCESS' and sha(capture)==r['capture_sha256'];c=load_file(str(capture));name='model.language_model.layers.0.mlp.gate.weight'
with safe_open(str(model/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:gate=f.get_tensor(name).contiguous()
assert gate.shape==(256,2048) and torch.isfinite(gate).all();g=gate.half();rows=g.tolist();results=[]
for batch in range(2):
 x=c[f'batch{batch}.input'];logits=F.linear(x,g).float()[0];values,idx=torch.topk(logits,8);weights=torch.softmax(values,dim=-1)
 # Softmax over all 256 followed by top-k normalization is equivalent to softmax over top-k.
 scalar=[math.fsum(float(a)*float(b) for a,b in zip(x[0].tolist(),row)) for row in rows];scalar=torch.tensor(scalar,dtype=torch.float16).float().tolist();order=sorted(range(256),key=lambda i:-scalar[i])[:8];assert order==idx.tolist();p=max(scalar[i] for i in order);e=[math.exp(scalar[i]-p) for i in order];den=math.fsum(e);ref=[v/den for v in e];err=max(abs(a-b) for a,b in zip(weights.tolist(),ref));assert err<1e-6
 results.append({'batch':batch,'expert_ids':order,'weights':weights.tolist(),'max_weight_error':err,'max_logit_error':max(abs(a-b) for a,b in zip(scalar,logits.tolist()))})
root=base/'r254-full-gate-r1';root.mkdir(mode=0o700);save_file({name:gate},str(root/'gate.safetensors'));result={'status':'PASS','source_revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5','gate_sha256':sha(root/'gate.safetensors'),'capture_sha256':r['capture_sha256'],'cases':results,'union':sorted(set(i for z in results for i in z['expert_ids'])),'gpu':False,'scope':'full gate on R252 synthetic post-attention hidden inputs; no full model generation'};(root/'receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result))
