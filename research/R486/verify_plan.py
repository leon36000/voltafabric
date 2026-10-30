#!/usr/bin/env python3
"""Independent canonical JSON/SHA256 vector for the Rust worker-plan test."""
import hashlib
import json
plan = {
    'version': 'vf-sc117-worker-plan-v2',
    'model_revision': 'sc117-local-observation',
    'index_sha256': 'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24',
    'epoch': 1, 'layers': 48, 'prompt_length': 2, 'max_new_tokens': 16, 'payload_budget': 300,
    'rank_limits': {'2': 100, '3': 200},
    'endpoints': {
        '2': {'rank': 2, 'argv': ['/worker']},
        '3': {'rank': 3, 'argv': ['/usr/bin/ssh', '-T', '-o', 'BatchMode=yes',
                '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
                'pc1-lan', '/worker']},
    },
}
encoded = json.dumps(plan, separators=(',', ':'), ensure_ascii=False).encode()
digest = hashlib.sha256(encoded).hexdigest()
assert digest == 'd10a0397d45343a51bf3ff2a7c77f39de8b286d9ba46e81875ffde9e53473aa1'
print('PASS: r486-' + digest)
