"""Independently reconstruct the 73-position local CPU plan."""
import hashlib
import json
import struct
from pathlib import Path

tokens = [248045, 846, 198, 12675, 248046, 198, 248045, 74455, 198,
          248068, 271, 248069, 271, 9419, 248046, 198, 248045, 846, 198,
          12675] + [20375] * 44 + [248046, 198, 248045, 74455, 198,
                                  248068, 271, 248069, 271]
ph = hashlib.sha256(b''.join(struct.pack('<I', x) for x in tokens)).hexdigest()
worker = '/home/pc1/NAOS/vf-r525-work/target/release/vf-rank-session'
worker_sha = '49216f6ce76a637be323dd0aa88c69e7788b4fa8ff13f5663ee103c5e4a22409'
plan = {'version': 'vf-sc117-worker-plan-v16',
        'wire_policy': 'native-f32-raw-if-advertised-otherwise-json-v1',
        'eviction_policy': 'rank-lru-then-size-v2',
        'origin_policy': 'stable-source-digest-cache-v1',
        'model_revision': 'sc117-local-observation',
        'index_sha256': 'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24',
        'ple_manifest_sha256': '436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824',
        'epoch': 1, 'layers': 48, 'prompt_length': 73,
        'prompt_tokens_sha256': ph, 'max_new_tokens': 20, 'eos_token_id': 248046,
        'prompt_mode': 'chat', 'payload_budget': 3221225472,
        'rank_limits': {'2': 1073741824, '3': 2147483648},
        'endpoints': {str(rank): {'rank': rank, 'argv': [worker, '--guarded-local',
                     '--expected-exe-sha256', worker_sha]} for rank in (2, 3)},
        'known_hosts_sha256': {}, 'shared_lease': None, 'r53_namespace': None}
base = 'r531-' + hashlib.sha256(json.dumps(plan, separators=(',', ':')).encode()).hexdigest()
pid = 'r544-' + hashlib.sha256(b'vf-qsa-trace-v2-layer3-input-96-f32le\0' + base.encode()).hexdigest()
actual = json.loads((Path(__file__).parent / 'PLAN.json').read_text())
assert actual == {'plan_id': pid, 'prompt_tokens_sha256': ph, 'prompt_token_ids': tokens}
print('PASS:', pid)
