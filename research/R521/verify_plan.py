"""Reconstruct the R521 plan from fixed configuration and token IDs."""
import hashlib,json,struct
from pathlib import Path
p=Path(__file__).parent
actual=json.loads((p/'PLAN.json').read_text())
tokens=[248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271]
ph=hashlib.sha256(b''.join(struct.pack('<I',x) for x in tokens)).hexdigest()
worker='/home/pc1/NAOS/vf-r495-work/target/release/vf-rank-session'
plan={'version':'vf-sc117-worker-plan-v10','wire_policy':'native-f32-raw-if-advertised-otherwise-json-v1','eviction_policy':'rank-lru-then-size-v2','origin_policy':'stable-source-digest-cache-v1','model_revision':'sc117-local-observation','index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24','epoch':1,'layers':48,'prompt_length':13,'prompt_tokens_sha256':ph,'max_new_tokens':1,'eos_token_id':248046,'prompt_mode':'chat','payload_budget':3221225472,'rank_limits':{'2':1073741824,'3':2147483648},'endpoints':{'2':{'rank':2,'argv':[worker]},'3':{'rank':3,'argv':[worker]}}}
pid='r520-'+hashlib.sha256(json.dumps(plan,separators=(',',':')).encode()).hexdigest()
assert actual=={'plan_id':pid,'prompt_tokens_sha256':ph,'prompt_token_ids':tokens}
print('PASS: independent v10 plan',pid)
