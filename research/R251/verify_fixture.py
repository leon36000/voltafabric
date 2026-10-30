import hashlib, json, os, pathlib
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
from safetensors import safe_open
from safetensors.torch import load_file
base = pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs')
new = base/'r251-qwen-norm-fixture-r1/model.safetensors'
old = base/'r241-qwen-loader-fixture-r1/model.safetensors'
checkpoint = base/'r188-qwen-checkpoint-pc1-20261028-r1/model'
assert hashlib.sha256(new.read_bytes()).hexdigest() == '5857afc34b1343321f6a87ddaf8716f49891efe240ba76c05fee3d229adc9219'
a, b = load_file(str(new)), load_file(str(old))
assert len(a)==110 and len(b)==107
assert all(torch.equal(a[k], v) for k,v in b.items())
extra = set(a)-set(b)
assert len(extra)==3
for shard, names in [('model-00001-of-00003.safetensors', ['model.language_model.layers.0.input_layernorm.weight','model.language_model.layers.0.post_attention_layernorm.weight']), ('model-00003-of-00003.safetensors',['model.language_model.norm.weight'])]:
    with safe_open(str(checkpoint/shard), framework='pt', device='cpu') as f:
        for name in names:
            assert name in extra and torch.equal(a[name],f.get_tensor(name))
print(json.dumps({'status':'PASS','prior_tensors_equal':len(b),'checkpoint_norm_tensors_equal':len(extra),'gpu':False}))
