import os,json,math,pathlib,torch
from safetensors import safe_open
from safetensors.torch import load_file
root=pathlib.Path(os.environ['OUTPUT']); receipt=json.loads((root/'receipt.json').read_text()); capture=load_file(str(root/'capture.safetensors'))
with safe_open(os.environ['MODEL_SHARD'],framework='pt',device='cpu') as f: gate=f.get_tensor('model.language_model.layers.3.mlp.gate.weight').half().tolist()
errors=[]
for b,route in enumerate(receipt['routes']):
 x=capture[f'batch{b}.input'][0].tolist(); logits=[float(torch.tensor(math.fsum(float(a)*float(v) for a,v in zip(x,row)),dtype=torch.float16)) for row in gate];ids=route['ids']; assert len(set(ids))==8 and min(logits[i] for i in ids)>=max(logits[i] for i in range(256) if i not in ids)
 ex=[math.exp(logits[i]-max(logits[j] for j in ids)) for i in ids];den=math.fsum(ex);err=max(abs(ex[i]/den-route['weights'][i]) for i in range(8));assert err<1e-6,err;errors.append(err)
print(json.dumps({'status':'PASS','cases':2,'weight_errors':errors,'gpu':False}))
