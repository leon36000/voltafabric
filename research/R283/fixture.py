import pathlib,os,json,hashlib
from safetensors import safe_open
from safetensors.torch import save_file
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r283-four-layer-fixture-r1';root.mkdir();old=b/'r282-two-layer-fixture-r1/nonexpert.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(old)=='4f3c2f338a7be338c391c703f058545732477c98a0de5b73a284ad8c3ab30329'
with safe_open(str(old),framework='pt',device='cpu') as f:weights={k:f.get_tensor(k).contiguous() for k in f.keys()}
with safe_open(str(b/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:
 for i in (2,3):
  pre=f'model.language_model.layers.{i}.';names=[pre+'input_layernorm.weight',pre+'post_attention_layernorm.weight',pre+'mlp.gate.weight',pre+'mlp.shared_expert_gate.weight']+[pre+'mlp.shared_expert.'+p+'.'+v for p in ('gate_proj','up_proj','down_proj') for v in ('weight','weight_scale','weight_scale_2')];weights.update({k:f.get_tensor(k).contiguous() for k in names})
assert len(weights)==53;save_file(weights,str(root/'nonexpert.safetensors'));groups=['r262-full-layer-r1','r280-layer1-r1','r283-layer2-r1','r275-layer3-full-r1']
for i,g in enumerate(groups):os.symlink(b/g/'experts.safetensors',root/f'experts{i}.safetensors')
result={'status':'PASS','layers':4,'nonexpert_tensors':53,'nonexpert_sha256':sha(root/'nonexpert.safetensors'),'gpu':False};(root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
