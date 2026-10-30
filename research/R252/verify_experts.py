import hashlib,json,os,pathlib,sys
os.environ.update(CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1')
import torch
from safetensors.torch import load_file
from nvfp4_oracle import expert_forward
torch.set_num_threads(1)
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs')
root=base/'r252-qwen-norm-live-20261028-r1';fixture=base/'r251-qwen-norm-fixture-r1/model.safetensors';receipt=json.loads((root/'receipt.json').read_text())
assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='5857afc34b1343321f6a87ddaf8716f49891efe240ba76c05fee3d229adc9219'
assert hashlib.sha256((root/'capture.safetensors').read_bytes()).hexdigest()==receipt['capture_sha256']
capture=load_file(str(root/'capture.safetensors'));ledger=pathlib.Path(receipt['ledger']);results=[]
for b in range(2):
 x=capture[f'batch{b}.input'][0]; ids=capture[f'batch{b}.ids'][0].tolist(); w=capture[f'batch{b}.weights'][0].tolist();assert x.dtype==torch.float16 and len(set(ids))==8
 acc=torch.zeros(2048,dtype=torch.float64)
 for expert,weight in zip(ids,w):acc+=expert_forward(fixture,0,expert,x).double()*float(weight)
 expected=acc.half(); out=ledger/f'batch-{b:020d}.fp16';done=json.loads((ledger/f'batch-{b:020d}.done.json').read_text());raw=out.read_bytes();assert hashlib.sha256(raw).hexdigest()==done['sha256']
 actual=torch.frombuffer(bytearray(raw),dtype=torch.float16);err=(actual.float()-expected.float()).abs();maximum=err.max().item();assert maximum<=0.0001,(b,maximum)
 results.append({'batch':b,'max_abs_error':maximum,'tolerance':0.0001,'actual_sha256':done['sha256']})
out={'status':'PASS','batches':results,'scope':'8 experts only, CPU float16 oracle compared to live Rust Collector; no full decoder oracle','gpu':False}
(root/'expert-oracle-receipt.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
