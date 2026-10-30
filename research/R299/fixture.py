import pathlib,os,json,hashlib,contextlib
from safetensors import safe_open
from safetensors.torch import save_file
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r299-forty-layer-fixture-r1';root.mkdir();old=b/'r298-thirtytwo-layer-fixture-r1/nonexpert.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(old)=='e91a72754f4b0c06b25921fd9dc70c805786ac0d01af53685450c2c6be869117'
with safe_open(str(old),framework='pt',device='cpu') as f:weights={k:f.get_tensor(k).contiguous() for k in f.keys()}
model=b/'r188-qwen-checkpoint-pc1-20261028-r1/model';index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];counts={}
with contextlib.ExitStack() as stack:
 files={i:stack.enter_context(safe_open(str(model/f'model-{i:05d}-of-00003.safetensors'),framework='pt',device='cpu')) for i in (1,2,3)}
 for i in range(32,40):
  pre=f'model.language_model.layers.{i}.';names=[pre+'input_layernorm.weight',pre+'post_attention_layernorm.weight',pre+'mlp.gate.weight',pre+'mlp.shared_expert_gate.weight']+[pre+'mlp.shared_expert.'+p+'.'+v for p in ('gate_proj','up_proj','down_proj') for v in ('weight','weight_scale','weight_scale_2')]
  for name in names:
   shard=index[name];number={'model-00001-of-00003.safetensors':1,'model-00002-of-00003.safetensors':2,'model-00003-of-00003.safetensors':3}[shard];counts[shard]=counts.get(shard,0)+1;weights[name]=files[number].get_tensor(name).contiguous()
assert len(weights)==521;save_file(weights,str(root/'nonexpert.safetensors'))
for i in (1,2,3):os.symlink(model/f'model-{i:05d}-of-00003.safetensors',root/f'expert_source_{i}.safetensors')
result={'status':'PASS','layers':40,'nonexpert_tensors':521,'new_weights_by_shard':counts,'nonexpert_sha256':sha(root/'nonexpert.safetensors'),'gpu':False};(root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
