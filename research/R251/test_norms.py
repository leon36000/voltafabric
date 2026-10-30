import hashlib, json, os, pathlib
os.environ.update(CUDA_VISIBLE_DEVICES='-1', VLLM_TARGET_DEVICE='cpu', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
import torch
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform = CpuPlatform()
torch.set_num_threads(1)
from safetensors.torch import load_file
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.config import VllmConfig, set_current_vllm_config
base = pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs')
p = base / 'r251-qwen-norm-fixture-r1/model.safetensors'
assert hashlib.sha256(p.read_bytes()).hexdigest() == '5857afc34b1343321f6a87ddaf8716f49891efe240ba76c05fee3d229adc9219'
tensors = load_file(str(p))
cfg = json.loads((base / 'r188-qwen-checkpoint-pc1-20261028-r1/model/config.json').read_text())
eps = cfg['text_config']['rms_norm_eps']
x = torch.tensor([[((i % 31) - 15) / 16 for i in range(2048)]], dtype=torch.float16)
r = torch.tensor([[((i % 17) - 8) / 32 for i in range(2048)]], dtype=torch.float32)
keys = ['model.language_model.layers.0.input_layernorm.weight', 'model.language_model.layers.0.post_attention_layernorm.weight', 'model.language_model.norm.weight']
results = []
for key in keys:
    weight = tensors[key]
    with set_current_vllm_config(VllmConfig()): m = GemmaRMSNorm(2048, eps=eps)
    with torch.no_grad(): m.weight.copy_(weight)
    for residual in (None, r):
        actual = m.forward_native(x, residual)
        out, out_residual = actual if residual is not None else (actual, None)
        z = x.double() if residual is None else x.double() + residual.double()
        expected = (z / torch.sqrt(torch.mean(z*z, dim=-1, keepdim=True) + eps) * (1 + weight.double())).half()
        diff = (out.float() - expected.float()).abs().max().item()
        assert diff <= 0.001953125, (key, residual is not None, diff)
        if residual is not None:
            assert out_residual.dtype == torch.float32
            assert torch.equal(out_residual, x.float() + residual)
        results.append({'name': key, 'residual': residual is not None, 'max_abs_fp16_vs_float64': diff, 'output_sha256': hashlib.sha256(out.contiguous().numpy().tobytes()).hexdigest()})
result = {'status': 'PASS', 'fixture_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'eps': eps, 'cases': results, 'gpu': False, 'cuda_initialized': torch.cuda.is_initialized()}
(base / 'r251-qwen-norm-fixture-r1/norm-test-receipt.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
print(json.dumps(result, sort_keys=True))
