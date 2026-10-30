import hashlib,json,pathlib,os
os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1')
from safetensors.torch import load_file,save_file
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');original=base/'r230-loader-fixture-r1/model.safetensors';shared=base/'r204-real-experts-20261028/shared-r208.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(original)=='d5131737934574cb89f2a5cfdd11b1d56f4bc1bc9185f99de960357312d8db21';assert sha(shared)=='6e83b12b99e891ce4eebd9adfb95960e7b40ae8d48811cd5499caa3ee6e09934'
t=load_file(str(original));s=load_file(str(shared));t.pop('model.language_model.layers.1.mlp.gate.weight');gate='model.language_model.layers.0.mlp.gate.weight';t[gate]=t[gate][:8].contiguous()
for k,v in s.items():
 if k in t:
  assert k.endswith('shared_expert_gate.weight') and (t[k]==v).all()
 else:t[k]=v
root=base/'r241-qwen-loader-fixture-r1';root.mkdir(mode=0o700);save_file(t,str(root/'model.safetensors'));print(json.dumps({'tensors':len(t),'bytes':(root/'model.safetensors').stat().st_size,'sha256':sha(root/'model.safetensors')}))
