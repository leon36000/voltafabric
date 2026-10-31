"""Independently reconstruct the local odd-length grouped prompt plan."""
import hashlib
import json
import struct
from pathlib import Path

tokens = [248045, 846, 198, 12675, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
ph = hashlib.sha256(b''.join(struct.pack('<I', x) for x in tokens)).hexdigest()
worker = '/home/pc1/NAOS/vf-r629-work/target/release/vf-rank-session'
worker_sha = 'f5eba82eb034a99b7948218adccfa9bb36314c03425098c4092de22993ef6132'
plan = {'version': 'vf-sc117-worker-plan-v16', 'wire_policy': 'native-f32-raw-if-advertised-otherwise-json-v1',
        'eviction_policy': 'rank-lru-then-size-v2', 'origin_policy': 'stable-source-digest-cache-v1',
        'model_revision': 'sc117-local-observation',
        'index_sha256': 'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24',
        'ple_manifest_sha256': '436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824',
        'epoch': 1, 'layers': 48, 'prompt_length': len(tokens), 'prompt_tokens_sha256': ph,
        'max_new_tokens': 8, 'eos_token_id': 248046, 'prompt_mode': 'chat',
        'payload_budget': 4294967296, 'rank_limits': None,
        'endpoints': {str(rank): {'rank': rank, 'argv': [worker, '--guarded-local',
                       '--expected-exe-sha256', worker_sha]} for rank in (2, 3)},
        'known_hosts_sha256': {}, 'shared_lease': None, 'r53_namespace': None}
base = 'r531-' + hashlib.sha256(json.dumps(plan, separators=(',', ':')).encode()).hexdigest()
admission = 'r579-' + hashlib.sha256(b'vf-sc117-layer-admission-preflight-v1\0' + base.encode()).hexdigest()
batch = 'r570-' + hashlib.sha256(b'vf-prompt-batch-two-local-cpu-v1\0' + admission.encode()).hexdigest()
pid = 'r576-' + hashlib.sha256(b'vf-expert-rank-affinity-v1\0' + batch.encode()).hexdigest()
pid = 'r600-' + hashlib.sha256(b'vf-sc117-scalar-reference-backend-v1\0' + pid.encode()).hexdigest()
prepared = json.loads((Path(__file__).parent / 'PREPARED.json').read_text())
assert prepared == {'plan_id': pid, 'prompt_tokens_sha256': ph, 'prompt_token_ids': tokens}
print('PASS', pid)
