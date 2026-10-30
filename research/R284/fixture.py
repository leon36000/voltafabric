import pathlib,os,json,hashlib
from safetensors import safe_open
from safetensors.torch import save_file
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r284-eight-layer-fixture-r1';root.mkdir();old=b/'r283-four-layer-fixture-r1/nonexpert.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(old)=='f9eaea8cf2111d9982c769802acf4fb24259c82e9a57b9d04662e9e30beb241f'
with safe_open(str(old),framework='pt',device='cpu') as f:weights={k:f.get_tensor(k).contiguous() for k in f.keys()}
shard=b/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors'
with safe_open(str(shard),framework='pt',device='cpu') as f:
 for i in range(4,8):
  pre=f'model.language_model.layers.{i}.';names=[pre+'input_layernorm.weight',pre+'post_attention_layernorm.weight',pre+'mlp.gate.weight',pre+'mlp.shared_expert_gate.weight']+[pre+'mlp.shared_expert.'+p+'.'+v for p in ('gate_proj','up_proj','down_proj') for v in ('weight','weight_scale','weight_scale_2')];weights.update({k:f.get_tensor(k).contiguous() for k in names})
assert len(weights)==105;save_file(weights,str(root/'nonexpert.safetensors'));os.symlink(shard,root/'expert_source.safetensors')
result={'status':'PASS','layers':8,'nonexpert_tensors':105,'nonexpert_sha256':sha(root/'nonexpert.safetensors'),'expert_source_sha256':'07141c2db92e47bc08777132cd1a0323faf300eab3a7d7c111bc2bf075fda050','gpu':False};(root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
