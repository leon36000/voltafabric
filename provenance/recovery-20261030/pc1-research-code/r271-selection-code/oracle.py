import os,json,pathlib,sys,torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from nvfp4_oracle import expert_forward
torch.set_num_threads(1);base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r276-onecat-layer3-r1';r=json.loads((root/'receipt.json').read_text());actual=load_file(str(root/'capture.safetensors'));ref=load_file(str(base/'r271-selection-data/capture.safetensors'));source=base/'r275-layer3-full-r1/experts.safetensors';model=base/'r188-qwen-checkpoint-pc1-20261028-r1/model';pre='model.language_model.layers.3.mlp.'
with safe_open(str(model/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:
 def mat(name):
  key=pre+'shared_expert.'+name+'.';codes=f.get_tensor(key+'weight');scales=f.get_tensor(key+'weight_scale').double();g=float(f.get_tensor(key+'weight_scale_2').item());n,k=codes.shape;lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64);ix=torch.stack((codes&15,codes>>4),-1).reshape(n,k*2).long();return (lut[ix]*scales.repeat_interleave(16,1)*g).half().double()
 gate,up,down=(mat(n) for n in ('gate_proj','up_proj','down_proj'));shared_gate=f.get_tensor(pre+'shared_expert_gate.weight').half()
with safe_open(str(model/'model-00003-of-00003.safetensors'),framework='pt',device='cpu') as f:final_norm=f.get_tensor('model.language_model.norm.weight').float()
ledger=pathlib.Path(r['ledger']);results=[]
for b in range(2):
 x=actual[f'batch{b}.input'];nerr=(x.float()-ref[f'batch{b}.input'].float()).abs().max().item();assert nerr<=.003,(b,'normalized',nerr)
 ids=actual[f'batch{b}.ids'][0].tolist();weights=actual[f'batch{b}.weights'][0].tolist();acc=torch.zeros(2048,dtype=torch.float64)
 for e,w in zip(ids,weights):acc+=expert_forward(source,3,e,x[0]).double()*float(w)
 routed=torch.frombuffer(bytearray((ledger/f'batch-{b:020d}.fp16').read_bytes()),dtype=torch.float16).reshape(1,-1);eerr=(routed.float()-acc.half().float()).abs().max().item();assert eerr<1e-4,(b,'expert',eerr)
 gx=(x.double()@gate.T).half();ux=(x.double()@up.T).half();mid=(F.silu(gx)*ux).half();shared=(mid.double()@down.T).half()*torch.sigmoid(F.linear(x,shared_gate));res=(ref[f'batch{b}.hidden'].float()+ref[f'batch{b}.attention'].float()).half();combined=(routed+shared).half();full=(res.float()+combined.float());expected=(full*torch.rsqrt(full.square().mean(-1,keepdim=True)+1e-6)*(1+final_norm)).half();err=(expected.float()-actual[f'batch{b}.output'].float()).abs().max().item();assert err<=.003,(b,'output',err);results.append({'batch':b,'post_norm_max_abs':nerr,'expert_max_abs':eerr,'decoder_max_abs':err})
out={'status':'PASS','cases':results,'tolerance_decoder':.003,'gpu':False,'limits':'attention activations reuse R271 capture; expert oracle independent, not full model'};(root/'oracle.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
