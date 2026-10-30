import pathlib,os,json,hashlib
from safetensors import safe_open
from safetensors.torch import save_file
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r294-twenty-layer-fixture-r1';root.mkdir();old=b/'r292-sixteen-layer-fixture-r1/nonexpert.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(old)=='f0c94c20c0a5e08c9401ed4bad637013707a7364a3e33399816d2483dbb86127'
with safe_open(str(old),framework='pt',device='cpu') as f:weights={k:f.get_tensor(k).contiguous() for k in f.keys()}
shard=b/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00002-of-00003.safetensors'
with safe_open(str(shard),framework='pt',device='cpu') as f:
 for i in range(16,20):
  pre=f'model.language_model.layers.{i}.';names=[pre+'input_layernorm.weight',pre+'post_attention_layernorm.weight',pre+'mlp.gate.weight',pre+'mlp.shared_expert_gate.weight']+[pre+'mlp.shared_expert.'+p+'.'+v for p in ('gate_proj','up_proj','down_proj') for v in ('weight','weight_scale','weight_scale_2')];weights.update({k:f.get_tensor(k).contiguous() for k in names})
assert len(weights)==261;save_file(weights,str(root/'nonexpert.safetensors'));os.symlink(shard,root/'expert_source_2.safetensors');os.symlink(shard.with_name('model-00001-of-00003.safetensors'),root/'expert_source_1.safetensors')
result={'status':'PASS','layers':20,'nonexpert_tensors':261,'nonexpert_sha256':sha(root/'nonexpert.safetensors'),'expert_source_sha256':'6dea9c759a0f941cf9e1cc1501216b0e107966a66c90425050e212efbd053f02','gpu':False};(root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
