"""Independent canonical v8 plan for the R516 no-worker preflight."""
import hashlib, json, struct
from pathlib import Path
p = Path(__file__).parent
actual = json.loads((p/'PLAN.json').read_text())
tokens = [248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271]
prompt_hash = hashlib.sha256(b''.join(struct.pack('<I', i) for i in tokens)).hexdigest()
worker = '/home/pc1/NAOS/vf-r495-work/target/release/vf-rank-session'
digest = '9dd3b08169e3ddce9228241b3c72a18f885cd8433db4d2806fa10aebc7b386c9'
plan = {'version':'vf-sc117-worker-plan-v8', 'wire_policy':'native-f32-raw-if-advertised-otherwise-json-v1',
        'eviction_policy':'rank-lru-then-size-v2', 'model_revision':'sc117-local-observation',
        'index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24',
        'epoch':1, 'layers':48, 'prompt_length':13, 'prompt_tokens_sha256':prompt_hash,
        'max_new_tokens':1, 'eos_token_id':248046, 'prompt_mode':'chat', 'payload_budget':3221225472,
        'rank_limits':{'2':1073741824,'3':2147483648},
        'endpoints':{'2':{'rank':2,'argv':[worker]},'3':{'rank':3,'argv':['/usr/bin/ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','pc1-lan',worker,'--guarded-remote','--expected-exe-sha256',digest]}}}
expected = 'r512-'+hashlib.sha256(json.dumps(plan,separators=(',',':')).encode()).hexdigest()
assert actual == {'plan_id':expected,'prompt_tokens_sha256':prompt_hash}, (actual, expected)
print('PASS: preflight plan independently reconstructed:', expected)
