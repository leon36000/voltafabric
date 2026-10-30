"""Independent Python reproduction of the R494 fixed Rust test vector."""
import hashlib
import json

plan = {
    'version': 'vf-sc117-worker-plan-v5',
    'wire_policy': 'native-f32-raw-if-advertised-otherwise-json-v1',
    'model_revision': 'sc117-local-observation',
    'index_sha256': 'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24',
    'epoch': 1, 'layers': 48, 'prompt_length': 2, 'max_new_tokens': 16,
    'eos_token_id': 248046, 'prompt_mode': 'plain', 'payload_budget': 300,
    'rank_limits': {'2': 100, '3': 200},
    'endpoints': {
        '2': {'rank': 2, 'argv': ['/worker']},
        '3': {'rank': 3, 'argv': ['/usr/bin/ssh', '-T', '-o', 'BatchMode=yes',
                                  '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
                                  'pc1-lan', '/worker', '--guarded-remote',
                                  '--expected-exe-sha256', 'a' * 64]},
    },
}
def digest(value):
    return 'r494-' + hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()
expected = 'r494-612a5812f810c8749c39d64a1ca68a949d298f8663292c8b3f2456abeace36ae'
assert digest(plan) == expected
plan['prompt_mode'] = 'chat'
assert digest(plan) != expected
print('EOS-bound plan vector OK')
