import os,pathlib,json,hashlib
from safetensors import safe_open
from safetensors.torch import save_file
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');shard=base/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors';selected=base/'r275-layer3-full-r1/experts.safetensors';root=base/'r276-layer3-model-fixture-r2';root.mkdir(mode=0o700)
prefix='model.language_model.layers.3.';fields=[prefix+'input_layernorm.weight',prefix+'post_attention_layernorm.weight',prefix+'mlp.gate.weight',prefix+'mlp.shared_expert_gate.weight']
for n in ('gate_proj','up_proj','down_proj'):
 for field in ('weight','weight_scale','weight_scale_2'):fields.append(prefix+'mlp.shared_expert.'+n+'.'+field)
with safe_open(str(shard),framework='pt',device='cpu') as f: tensors={name:f.get_tensor(name).contiguous() for name in fields}
with safe_open(str(shard.with_name('model-00003-of-00003.safetensors')),framework='pt',device='cpu') as f: tensors['model.language_model.norm.weight']=f.get_tensor('model.language_model.norm.weight').contiguous()
save_file(tensors,str(root/'nonexpert.safetensors'));os.symlink(selected,root/'experts.safetensors')
with safe_open(str(selected),framework='pt',device='cpu') as f: assert len(f.keys())==3072 and all(k.startswith(prefix+'mlp.experts.') for k in f.keys())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();result={'status':'PASS','tensors':len(tensors),'expert_tensors':3072,'nonexpert_sha256':sha(root/'nonexpert.safetensors'),'expert_sha256':sha(selected),'gpu':False};assert result['expert_sha256']=='2802aa69c3254c8b7c8944448ec3e9f6116e9633c504681ea5b8b4726bed864f';(root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
