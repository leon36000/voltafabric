import os,json,pathlib,torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
torch.set_num_threads(2);root=pathlib.Path(os.environ['ROOT']);r=json.loads((root/'receipt.json').read_text());capture=load_file(os.environ['CAPTURE']);prefix='model.language_model.layers.3.mlp.'
with safe_open(os.environ['MODEL_SHARD'],framework='pt',device='cpu') as f:
 def mat(name):
  key=prefix+'shared_expert.'+name+'.';codes=f.get_tensor(key+'weight');scale=f.get_tensor(key+'weight_scale').double();g=float(f.get_tensor(key+'weight_scale_2').item());n,k=codes.shape;lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64);idx=torch.stack((codes&15,codes>>4),dim=-1).reshape(n,k*2).long();return (lut[idx]*scale.repeat_interleave(16,1)*g).half().double()
 gate,up,down=(mat(n) for n in ('gate_proj','up_proj','down_proj'));shared_gate=f.get_tensor(prefix+'shared_expert_gate.weight').half()
results=[]
for row in r['batches']:
 b=row['batch'];x=capture[f'batch{b}.input'];h=capture[f'batch{b}.hidden'];attn=capture[f'batch{b}.attention'];a=(x.double()@gate.T).half();u=(x.double()@up.T).half();mid=(F.silu(a)*u).half();shared=(mid.double()@down.T).half()*torch.sigmoid(F.linear(x,shared_gate));routed=torch.tensor(row['output'],dtype=torch.float16).reshape(1,-1);expected=((h.float()+attn.float()).half()+(routed+shared).half()).half();actual=torch.tensor(row['layer_output'],dtype=torch.float16).reshape(1,-1);se=(shared.float()-torch.tensor(row['shared_output']).reshape(1,-1)).abs().max().item();err=(expected.float()-actual.float()).abs().max().item();assert se<=.0001 and err<=.003,(b,se,err);results.append({'batch':b,'shared_max_abs':se,'layer_max_abs':err})
out={'status':'PASS','cases':results,'gpu':False,'limits':'shared expert and residual only; attention and routed output separately checked in R269/R273'};(root/'oracle.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
