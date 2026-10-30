#!/usr/bin/env python3
"""Reconstruct the observed local rank plan independently of the Rust binary."""
import hashlib
import json
from pathlib import Path
worker = '/home/pc1/NAOS/vf-r486-work/target/release/vf-rank-session'
plan = {
    'version': 'vf-sc117-worker-plan-v2',
    'model_revision': 'sc117-local-observation',
    'index_sha256': 'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24',
    'epoch': 1, 'layers': 48, 'prompt_length': 2, 'max_new_tokens': 16, 'payload_budget': 3221225472,
    'rank_limits': {'2': 1073741824, '3': 2147483648},
    'endpoints': {'2': {'rank': 2, 'argv': [worker]}, '3': {'rank': 3, 'argv': [worker]}},
}
digest = 'r486-' + hashlib.sha256(json.dumps(plan, separators=(',', ':')).encode()).hexdigest()
receipt = json.loads((Path(__file__).parent / 'RESULT.json').read_text())['receipt']
assert receipt['plan_id'] == digest
assert receipt['worker_transports'] == {'2': 'local', '3': 'local'}
print('PASS: ' + digest)
