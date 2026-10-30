"""Compare real CPU projection implementation to scalar-layout reconstruction.

Offline R344 capture; this does not capture the actual shared-branch output.
"""
import hashlib,json,pathlib,types,torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform=CpuPlatform()
from vllm.config import VllmConfig,set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from adapters.onecat_cpu_nvfp4 import CpuNvfp4Reference

torch.set_num_threads(1); assert not torch.cuda.is_initialized()
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs')
root=b/'r344-single-load-forty-layer-r1';r=json.loads((root/'receipt.json').read_text());capture=root/'capture.safetensors';assert hashlib.sha256(capture.read_bytes()).hexdigest()==r['capture_sha256']
fixture=b/'r299-forty-layer-fixture-r1/nonexpert.safetensors';assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='adbf0ee2a8e67a89d178fdbe81ee0fe88bf6197ec541beaee045524ae9919663'
pre='model.language_model.layers.3.';names=('gate_proj','up_proj','down_proj');weights={}
with safe_open(str(fixture),framework='pt',device='cpu') as f:
 for n in names:
  k=pre+'mlp.shared_expert.'+n+'.'
  for field in ('weight','weight_scale','weight_scale_2'):weights[n+'.'+field]=f.get_tensor(k+field)
 weights['shared_gate']=f.get_tensor(pre+'mlp.shared_expert_gate.weight').half()
lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
def decode(n):
 c=weights[n+'.weight'];s=weights[n+'.weight_scale'].float();g=weights[n+'.weight_scale_2'].float();ix=torch.stack((c&15,c>>4),-1).reshape(c.shape[0],-1).long()
 return (lut[ix]*s.repeat_interleave(16,1)*g).half()
mat={n:decode(n) for n in names};fused=torch.cat((mat['gate_proj'],mat['up_proj']),0)
def run_projection(matrix,x):
 layer=types.SimpleNamespace(_vf_decoded_weight=matrix)
 obj=object.__new__(CpuNvfp4Reference);obj._vf_layer=layer;obj._vf_failed=False;obj._vf_finalized=True
 return obj.apply(layer,x)
cap=load_file(str(capture));results=[]
with set_current_vllm_config(VllmConfig()),torch.inference_mode():
 for batch in range(3):
  x=cap[f'batch{batch*40+3}.input'];z=run_projection(fused,x);act=SiluAndMul.forward_native(z);down=run_projection(mat['down_proj'],act);gate=torch.sigmoid(F.linear(x,weights['shared_gate']));actual=(down*gate).half()
  a=(x.double()@mat['gate_proj'].double().T).half();u=(x.double()@mat['up_proj'].double().T).half();mid=(F.silu(a)*u).half();reference=((mid.double()@mat['down_proj'].double().T).half()*torch.sigmoid(F.linear(x,weights['shared_gate']))).half()
  results.append({'batch':batch,'gate_projection_max':float((z[:,:a.shape[1]]-a).abs().max()),'up_projection_max':float((z[:,a.shape[1]:]-u).abs().max()),'activation_max':float((act-mid).abs().max()),'shared_max':float((actual-reference).abs().max())})
out={'status':'DIAGNOSTIC_NOT_CAPTURED_BRANCH','layer':3,'cases':results,'actual_shared_branch_captured':False,'gpu':False};p=root/'shared-comparison-r351.json'
with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
