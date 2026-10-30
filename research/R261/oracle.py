import json,pathlib,sys,torch
from safetensors.torch import load_file
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';sys.path.insert(0,str(R/'research/R199'));from nvfp4_oracle import expert_forward
torch.set_num_threads(1);root=pathlib.Path(sys.argv[1]);c=load_file(str(root/'capture.safetensors'));gate=json.loads((B/'.runs/r254-full-routing-20261028/gate-receipt.json').read_text());fixture=B/'.runs/r254-full-routing-20261028/experts.safetensors';results=[]
for b in range(2):
 case=gate['cases'][b];x=c[f'batch{b}.input'][0];acc=torch.zeros(2048,dtype=torch.float64)
 for e,w in zip(case['expert_ids'],case['weights']):acc+=expert_forward(fixture,0,e,x).double()*float(w)
 output=torch.tensor(json.loads((root/f'batch{b}.json').read_text())['result']['output'][0],dtype=torch.float64);err=(output-acc).abs().max().item();assert err<0.0001,(b,err);results.append({'batch':b,'max_abs_error':err})
out={'status':'PASS','tolerance':0.0001,'cases':results,'gpu':False,'scope':'CPU FP16 expert outputs versus independent original NVFP4 oracle, two captured synthetic inputs'};(root/'oracle.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
