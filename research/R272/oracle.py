import json,pathlib,sys,torch
from safetensors.torch import load_file
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';sys.path.insert(0,str(R/'research/R199'))
from nvfp4_oracle import expert_forward
torch.set_num_threads(1);root=B/'.runs/r272-layer3-session-20261028-r2';data=B/'.runs/r272-layer3-owners-20261028';receipt=json.loads((B/'.runs/r271-layer3-selection-20261028/receipt.json').read_text());capture=load_file(str(data/'capture.safetensors'));results=[]
for b in range(2):
 route=receipt['routes'][b];x=capture[f'batch{b}.input'][0];acc=torch.zeros(2048,dtype=torch.float64)
 for e,w in zip(route['ids'],route['weights']):acc+=expert_forward(data/'experts.safetensors',3,e,x).double()*float(w)
 output=torch.tensor(json.loads((root/f'batch{b}.json').read_text())['result']['output'][0],dtype=torch.float64);err=(output-acc).abs().max().item();assert err<0.0001,(b,err);results.append({'batch':b,'max_abs_error':err})
out={'status':'PASS','tolerance':0.0001,'cases':results,'gpu':False};(root/'oracle.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
