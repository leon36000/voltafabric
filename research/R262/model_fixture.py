import hashlib,json,os,pathlib
os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
from safetensors import safe_open
from safetensors.torch import save_file
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');old=base/'r251-qwen-norm-fixture-r1/model.safetensors';gate=base/'r254-full-gate-r1/gate.safetensors';experts=base/'r262-full-layer-r1/experts.safetensors';root=base/'r262-full-model-fixture-r1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old)=='5857afc34b1343321f6a87ddaf8716f49891efe240ba76c05fee3d229adc9219' and sha(gate)=='64835dd620ae1d75e8b5eb0822c7a7753821caed2a935a0f0b4444ab9e7c0c1f'
name='model.language_model.layers.0.mlp.gate.weight'
with safe_open(str(old),framework='pt',device='cpu') as f:other={k:f.get_tensor(k).contiguous() for k in f.keys() if '.mlp.experts.' not in k}
assert len(other)==14
with safe_open(str(gate),framework='pt',device='cpu') as f:other[name]=f.get_tensor(name).contiguous()
assert other[name].shape==(256,2048)
root.mkdir(mode=0o700);save_file(other,str(root/'nonexpert.safetensors'));os.symlink(experts,root/'experts.safetensors')
with safe_open(str(root/'experts.safetensors'),framework='pt',device='cpu') as f:keys=f.keys();assert len(keys)==3072
out={'status':'PASS','nonexpert_tensors':len(other),'expert_tensors':len(keys),'nonexpert_sha256':sha(root/'nonexpert.safetensors'),'expert_source_sha256':'9b03db3babfc11af78d6c02fef2bba3435223f58a0f6d27bb952e60d41a11ddc','gpu':False};(root/'receipt.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
