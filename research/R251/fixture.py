import hashlib, json, os, pathlib
os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1')
from safetensors import safe_open
from safetensors.torch import load_file, save_file

base = pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs')
fixture = base / 'r241-qwen-loader-fixture-r1/model.safetensors'
checkpoint = base / 'r188-qwen-checkpoint-pc1-20261028-r1/model'
out = base / 'r251-qwen-norm-fixture-r1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(fixture) == '35fe78b51d7f0f0f7dcdefddb023bc9a1aa1b9a13ccfecf190437261da1e3836'
tensors = load_file(str(fixture))
assert len(tensors) == 107
names = {'model-00001-of-00003.safetensors': ['model.language_model.layers.0.input_layernorm.weight', 'model.language_model.layers.0.post_attention_layernorm.weight'], 'model-00003-of-00003.safetensors': ['model.language_model.norm.weight']}
for shard, keys in names.items():
    with safe_open(str(checkpoint / shard), framework='pt', device='cpu') as f:
        for key in keys:
            assert key not in tensors
            tensors[key] = f.get_tensor(key).contiguous()
            assert tensors[key].shape == (2048,) and tensors[key].isfinite().all()
assert len(tensors) == 110
out.mkdir(mode=0o700)
save_file(tensors, str(out / 'model.safetensors'))
result = {'sha256': sha(out / 'model.safetensors'), 'bytes': (out / 'model.safetensors').stat().st_size, 'tensors': len(tensors), 'norm_names': sum(names.values(), []), 'gpu': False}
(out / 'receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True))
