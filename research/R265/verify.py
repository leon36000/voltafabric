import hashlib,json,math,os,pathlib,sys
os.environ.update(CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1')
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from nvfp4_oracle import expert_forward
torch.set_num_threads(1)
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r265-token-path-20261028-r1';r=json.loads((root/'receipt.json').read_text());source=base/'r262-full-layer-r1/experts.safetensors';gate=base/'r254-full-gate-r1/gate.safetensors';cap=root/'capture.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert r['status']=='SUCCESS' and r['loader_filtered']==3072 and r['cuda_initialized'] is False and sha(cap)==r['capture_sha256'] and sha(gate)=='64835dd620ae1d75e8b5eb0822c7a7753821caed2a935a0f0b4444ab9e7c0c1f';x=load_file(str(cap));g=load_file(str(gate))['model.language_model.layers.0.mlp.gate.weight'].half().tolist();ledger=pathlib.Path(r['ledger']);results=[]
for b in range(2):
 inp=x[f'batch{b}.input'][0];ids=x[f'batch{b}.ids'][0].tolist();weights=x[f'batch{b}.weights'][0].tolist();expected_logits=torch.tensor([math.fsum(float(a)*float(v) for a,v in zip(inp.tolist(),row)) for row in g],dtype=torch.float16).float().tolist();assert len(ids)==8 and len(set(ids))==8 and all(expected_logits[ids[j]]>=expected_logits[ids[j+1]] for j in range(7));assert min(expected_logits[j] for j in ids)>=max(expected_logits[j] for j in range(256) if j not in ids);maxv=max(expected_logits[j] for j in ids);e=[math.exp(expected_logits[j]-maxv) for j in ids];den=math.fsum(e);weight_error=max(abs(a/den-b) for a,b in zip(e,weights));assert weight_error<1e-6
 acc=torch.zeros(2048,dtype=torch.float64)
 for expert,weight in zip(ids,weights):acc+=expert_forward(source,0,expert,inp).double()*float(weight)
 raw=(ledger/f'batch-{b:020d}.fp16').read_bytes();done=json.loads((ledger/f'batch-{b:020d}.done.json').read_text());assert sha(ledger/f'batch-{b:020d}.fp16')==done['sha256'];actual=torch.frombuffer(bytearray(raw),dtype=torch.float16);err=(actual.float()-acc.half().float()).abs().max().item();assert err<0.0001,(b,err);results.append({'batch':b,'route_ids':ids,'max_weight_error':weight_error,'expert_max_abs_error':err})
assert (ledger/'closed.json').exists();out={'status':'PASS','batches':results,'expert_tolerance':0.0001,'gpu':False,'scope':'two embedding-driven inputs through actual 1Cat one-layer fixture, full gate, 16 CPU owners and Rust Collector; no full model/generation'};(root/'oracle.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
