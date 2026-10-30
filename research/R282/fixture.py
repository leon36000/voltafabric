import pathlib,os,json,hashlib
from safetensors import safe_open
from safetensors.torch import save_file
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r282-two-layer-fixture-r1';root.mkdir();layer0=b/'r262-full-model-fixture-r1/nonexpert.safetensors';shard=b/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(layer0)=='35c4d00d2dcbc09db0ac89494ceee34820b2ca52d0805b9ac0959418113f4f15'
with safe_open(str(layer0),framework='pt',device='cpu') as f:tensors={k:f.get_tensor(k).contiguous() for k in f.keys()}
pre='model.language_model.layers.1.';fields=[pre+'input_layernorm.weight',pre+'post_attention_layernorm.weight',pre+'mlp.gate.weight',pre+'mlp.shared_expert_gate.weight']+[pre+'mlp.shared_expert.'+proj+'.'+field for proj in ('gate_proj','up_proj','down_proj') for field in ('weight','weight_scale','weight_scale_2')]
with safe_open(str(shard),framework='pt',device='cpu') as f:tensors.update({k:f.get_tensor(k).contiguous() for k in fields})
assert len(tensors)==27;save_file(tensors,str(root/'nonexpert.safetensors'));os.symlink(b/'r262-full-layer-r1/experts.safetensors',root/'experts0.safetensors');os.symlink(b/'r280-layer1-r1/experts.safetensors',root/'experts1.safetensors')
for i in (0,1):
 with safe_open(str(root/f'experts{i}.safetensors'),framework='pt',device='cpu') as f:assert len(f.keys())==3072 and all(k.startswith(f'model.language_model.layers.{i}.mlp.experts.') for k in f.keys())
result={'status':'PASS','nonexpert_tensors':len(tensors),'expert_tensors_per_layer':3072,'nonexpert_sha256':sha(root/'nonexpert.safetensors'),'gpu':False};(root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
