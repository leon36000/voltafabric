import json,pathlib,sys,hashlib,torch
from safetensors.torch import load_file
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');repo=B/'.runs/gitlab-import-r113-20261028/repository';sys.path.insert(0,str(repo/'research/R199'));from nvfp4_oracle import expert_forward
root=B/'.runs/r274-live-decoder-20261028-r1';data=B/'.runs/r272-layer3-owners-20261028';r=json.loads((root/'receipt.json').read_text());capture=load_file(str(data/'capture.safetensors'));ledger=pathlib.Path(r['ledger']);out=[];torch.set_num_threads(1)
for item in r['batches']:
 b=item['batch'];x=capture[f'batch{b}.input'][0];acc=torch.zeros(2048,dtype=torch.float64)
 for e,w in zip(item['ids'],item['weights']):acc+=expert_forward(data/'experts.safetensors',3,e,x).double()*float(w)
 claim=json.loads((ledger/f'batch-{b:020d}.claim.json').read_text());done=json.loads((ledger/f'batch-{b:020d}.done.json').read_text());raw=(ledger/f'batch-{b:020d}.fp16').read_bytes();assert claim['batch']==done['batch']==b and hashlib.sha256(raw).hexdigest()==done['sha256'];actual=torch.frombuffer(bytearray(raw),dtype=torch.float16);assert torch.equal(actual,torch.tensor(item['output'],dtype=torch.float16));error=(actual.float()-acc.half().float()).abs().max().item();assert error<1e-4,(b,error);out.append({'batch':b,'max_abs_error':error})
assert (ledger/'closed.json').exists();result={'status':'PASS','cases':out,'tolerance':0.0001,'gpu':False};(root/'oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
