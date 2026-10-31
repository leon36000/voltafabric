"""Independent reconstruction of layer-preflight, batch and affinity domains."""
import hashlib
import json
import runpy
from pathlib import Path
p = Path(__file__).parent
baseline = runpy.run_path(str(p.parent / 'R571' / 'verify_plan.py'))
base = baseline['base']
def bind(prefix, domain, parent):
    return prefix + hashlib.sha256(domain + parent.encode('ascii')).hexdigest()
admission = bind('r579-', b'vf-sc117-layer-admission-preflight-v1\0', base)
batch = bind('r570-', b'vf-prompt-batch-two-local-cpu-v1\0', admission)
affinity = bind('r576-', b'vf-expert-rank-affinity-v1\0', batch)
prepared = json.loads((p / 'PREPARED.json').read_bytes())
assert prepared['plan_id'] == affinity
assert prepared['prompt_token_ids'] == baseline['tokens']
assert prepared['prompt_tokens_sha256'] == baseline['ph']
print('PASS', affinity)
