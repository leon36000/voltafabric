import hashlib,json,math,os,pathlib
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
from safetensors.torch import load_file
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs')
root=base/'r252-qwen-norm-live-20261028-r1'
receipt=json.loads((root/'receipt.json').read_text())
assert receipt['status']=='SUCCESS' and receipt['gpu'] is False and receipt['cuda_initialized'] is False
capture=root/'capture.safetensors'
assert hashlib.sha256(capture.read_bytes()).hexdigest()==receipt['capture_sha256']
fixture=base/'r251-qwen-norm-fixture-r1/model.safetensors'
assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='5857afc34b1343321f6a87ddaf8716f49891efe240ba76c05fee3d229adc9219'
t=load_file(str(capture));weights=load_file(str(fixture))['model.language_model.layers.0.mlp.gate.weight'].half().tolist()
results=[]
for i in range(2):
 x=t[f'batch{i}.input'][0].tolist(); actual_logits=t[f'batch{i}.logits'][0].tolist(); actual_ids=t[f'batch{i}.ids'][0].tolist(); actual_w=t[f'batch{i}.weights'][0].tolist()
 logits=[math.fsum(float(a)*float(b) for a,b in zip(x,row)) for row in weights]
 # Actual gate output is stored in FP16, so round each independent sum once.
 logits=torch.tensor(logits,dtype=torch.float16).float().tolist()
 assert max(abs(a-b) for a,b in zip(logits,actual_logits))<=0.001953125
 order=sorted(range(8),key=lambda j:-logits[j]); assert order==actual_ids
 peak=max(logits); exp=[math.exp(v-peak) for v in logits]; total=math.fsum(exp); ref=[exp[j]/total for j in order]
 err=max(abs(a-b) for a,b in zip(ref,actual_w)); assert err<0.0001
 assert len(set(actual_ids))==8 and all(w>=0 and math.isfinite(w) for w in actual_w) and abs(math.fsum(actual_w)-1)<1e-6
 results.append({'batch':i,'ids':order,'max_abs_weight_error':err,'max_abs_logit_error':max(abs(a-b) for a,b in zip(logits,actual_logits))})
ledger=pathlib.Path(receipt['ledger']);assert (ledger/'closed.json').exists()
for i in range(2):
 claim=ledger/f'batch-{i:020d}.claim.json'; done=ledger/f'batch-{i:020d}.done.json'; output=ledger/f'batch-{i:020d}.fp16'
 assert claim.exists() and done.exists() and output.exists()
 d=json.loads(done.read_text());assert hashlib.sha256(output.read_bytes()).hexdigest()==d['sha256'] and d['batch']==i
out={'status':'PASS','cases':results,'completed_ledger_batches':2,'closed':True,'gpu':False,'scope':'8 of 256 expert gate rows only, synthetic hidden inputs'}
(root/'route-oracle-receipt.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
