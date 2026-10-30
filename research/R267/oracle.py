import hashlib,json,pathlib,sys,torch
from safetensors.torch import load_file
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';sys.path.insert(0,str(R/'research/R199'));from nvfp4_oracle import expert_forward
torch.set_num_threads(1);root=B/'.runs/r267-exact-provider-20261028';c=load_file(str(root/'capture.safetensors'));fixture=B/'.runs/r262-full-layer-20261028/experts.safetensors';assert hashlib.sha256((root/'capture.safetensors').read_bytes()).hexdigest()=='8444bce6a7ecdc76a21463b01e62ddf3c108d18887037ff6a2c6dfb1503adcd5';results=[]
for b in range(2):
 x=c[f'batch{b}.input'][0];ids=c[f'batch{b}.ids'][0].tolist();weights=c[f'batch{b}.weights'][0].tolist();acc=torch.zeros(2048,dtype=torch.float64)
 for e,w in zip(ids,weights):acc+=expert_forward(fixture,0,e,x).double()*float(w)
 obj=json.loads((root/f'batch{b}.json').read_text());assert obj['ok'] and obj['result']['ok'];actual=torch.tensor(obj['result']['output'][0],dtype=torch.float64);err=(actual-acc).abs().max().item();assert err<=1e-7,(b,err);results.append({'batch':b,'max_abs_error':err,'ranks':[(r['rank'],r['expected_slots']) for r in obj['result']['receipts']]})
out={'status':'PASS','tolerance':1e-7,'cases':results,'gpu':False,'scope':'exact-scale CPU provider; same captured R265 inputs, not actual SM70 GPU'};(root/'oracle.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
