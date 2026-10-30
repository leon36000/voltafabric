import json,math,pathlib,hashlib,torch
from safetensors import safe_open
from safetensors.torch import load_file
from nvfp4_oracle import expert_forward
torch.set_num_threads(1);b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r284-eight-layer-model-r1';r=json.loads((root/'receipt.json').read_text());cap=root/'capture.safetensors';assert hashlib.sha256(cap.read_bytes()).hexdigest()==r['capture_sha256'];x=load_file(str(cap));shard=b/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors';results=[]
with safe_open(str(shard),framework='pt',device='cpu') as f:gates={l:f.get_tensor(f'model.language_model.layers.{l}.mlp.gate.weight').half().tolist() for l in range(8)}
for batch in (0,1):
 for layer in range(8):
  index=batch*8+layer;inp=x[f'batch{index}.input'][0];ids=x[f'batch{index}.ids'][0].tolist();weights=x[f'batch{index}.weights'][0].tolist();logits=[float(torch.tensor(math.fsum(float(a)*float(v) for a,v in zip(inp.tolist(),row)),dtype=torch.float16)) for row in gates[layer]];assert len(set(ids))==8 and min(logits[i] for i in ids)>=max(logits[i] for i in range(256) if i not in ids);ev=[math.exp(logits[i]-max(logits[j] for j in ids)) for i in ids];den=math.fsum(ev);we=max(abs(ev[i]/den-weights[i]) for i in range(8));assert we<1e-6
  source=b/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors';acc=torch.zeros(2048,dtype=torch.float64)
  for e,w in zip(ids,weights):acc+=expert_forward(source,layer,e,inp).double()*w
  ledger=pathlib.Path(r['ledgers'][str(layer)]);raw=(ledger/f'batch-{batch:020d}.fp16').read_bytes();done=json.loads((ledger/f'batch-{batch:020d}.done.json').read_text());assert hashlib.sha256(raw).hexdigest()==done['sha256'];out=torch.frombuffer(bytearray(raw),dtype=torch.float16);err=(out.float()-acc.half().float()).abs().max().item();assert err<1e-4,(batch,layer,err);results.append({'batch':batch,'layer':layer,'weight_error':we,'expert_max_abs':err})
assert all((pathlib.Path(p)/'closed.json').exists() for p in r['ledgers'].values());out={'status':'PASS','cases':results,'gpu':False};(root/'expert-oracle.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
