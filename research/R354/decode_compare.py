"""Compare float32/float64 NVFP4 shared decoding on R344 fixtures."""
import hashlib,json,pathlib,torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
torch.set_num_threads(1);assert not torch.cuda.is_initialized()
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r344-single-load-forty-layer-r1';receipt=json.loads((root/'receipt.json').read_text());capture=root/'capture.safetensors';assert hashlib.sha256(capture.read_bytes()).hexdigest()==receipt['capture_sha256'];cap=load_file(str(capture));fixture=b/'r299-forty-layer-fixture-r1/nonexpert.safetensors';assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='adbf0ee2a8e67a89d178fdbe81ee0fe88bf6197ec541beaee045524ae9919663'
base=[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.];records=[]
def decode(f,key,dtype):
 c=f.get_tensor(key+'weight');s=f.get_tensor(key+'weight_scale').to(dtype);g=f.get_tensor(key+'weight_scale_2').to(dtype);idx=torch.stack((c&15,c>>4),-1).reshape(c.shape[0],-1).long();return (torch.tensor(base,dtype=dtype)[idx]*s.repeat_interleave(16,1)*g).half()
def shared(x,mats,gate):
 a=(x.double()@mats[0].double().T).half();u=(x.double()@mats[1].double().T).half();mid=(F.silu(a)*u).half();return ((mid.double()@mats[2].double().T).half()*torch.sigmoid(F.linear(x,gate))).half()
with torch.inference_mode(),safe_open(str(fixture),framework='pt',device='cpu') as f:
 for layer in range(5):
  pre=f'model.language_model.layers.{layer}.mlp.';keys=[pre+'shared_expert.'+n+'.' for n in ('gate_proj','up_proj','down_proj')];a=[decode(f,k,torch.float32) for k in keys];b64=[decode(f,k,torch.float64) for k in keys];gate=f.get_tensor(pre+'shared_expert_gate.weight').half();counts=[int((x!=y).sum()) for x,y in zip(a,b64)]
  for batch in range(3):
   x=cap[f'batch{batch*40+layer}.input'];d=float((shared(x,a,gate)-shared(x,b64,gate)).abs().max());records.append({'batch':batch,'layer':layer,'weight_difference_counts':counts,'shared_output_max':d})
out={'status':'DIAGNOSTIC_NOT_PASS','records':records,'gpu':False};dest=root/'decode-comparison-r354.json'
with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
