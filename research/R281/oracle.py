import json,pathlib,math,hashlib,torch
from safetensors import safe_open
from safetensors.torch import load_file
from nvfp4_oracle import expert_forward
torch.set_num_threads(1);base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r281-layer1-session-r1';r=json.loads((root/'receipt.json').read_text());capture=load_file(str(root/'capture.safetensors'));ledger=pathlib.Path(r['ledger']);source=base/'r280-layer1-r1/experts.safetensors'
with safe_open(str(base/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:gate=f.get_tensor('model.language_model.layers.1.mlp.gate.weight').half().tolist()
result=[]
for b in range(2):
 x=capture['x'][b];ids=capture['ids'][b].tolist();weights=capture['weights'][b].tolist();logits=[float(torch.tensor(math.fsum(float(a)*float(v) for a,v in zip(x.tolist(),row)),dtype=torch.float16)) for row in gate];assert len(set(ids))==8 and min(logits[i] for i in ids)>=max(logits[i] for i in range(256) if i not in ids);ev=[math.exp(logits[i]-max(logits[j] for j in ids)) for i in ids];den=math.fsum(ev);we=max(abs(ev[i]/den-weights[i]) for i in range(8));assert we<1e-6
 acc=torch.zeros(2048,dtype=torch.float64)
 for e,w in zip(ids,weights):acc+=expert_forward(source,1,e,x).double()*w
 raw=(ledger/f'batch-{b:020d}.fp16').read_bytes();done=json.loads((ledger/f'batch-{b:020d}.done.json').read_text());assert hashlib.sha256(raw).hexdigest()==done['sha256'];out=torch.frombuffer(bytearray(raw),dtype=torch.float16);err=(out.float()-acc.half().float()).abs().max().item();assert err<1e-4;(result.append({'batch':b,'weight_error':we,'expert_max_abs':err}))
assert (ledger/'closed.json').exists();out={'status':'PASS','cases':result,'gpu':False};(root/'oracle.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
