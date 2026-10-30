"""Reconstruct the R523 plans from fixed configuration and token IDs."""
import hashlib,json,struct
from pathlib import Path
p=Path(__file__).parent

tokens=[248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271]
ph=hashlib.sha256(b''.join(struct.pack('<I',x) for x in tokens)).hexdigest()
worker='/home/pc1/NAOS/vf-r525-work/target/release/vf-rank-session'
plan={'version':'vf-sc117-worker-plan-v12','wire_policy':'native-f32-raw-if-advertised-otherwise-json-v1','eviction_policy':'rank-lru-then-size-v2','origin_policy':'stable-source-digest-cache-v1','model_revision':'sc117-local-observation','index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24','ple_manifest_sha256':None,'epoch':1,'layers':48,'prompt_length':13,'prompt_tokens_sha256':ph,'max_new_tokens':16,'eos_token_id':248046,'prompt_mode':'chat','payload_budget':3221225472,'rank_limits':{'2':1073741824,'3':2147483648},'endpoints':{'2':{'rank':2,'argv':[worker,'--guarded-local','--expected-exe-sha256',None]},'3':{'rank':3,'argv':[worker,'--guarded-local','--expected-exe-sha256',None]}}}
plan['ple_manifest_sha256']='436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824'
for rank in ('2','3'): plan['endpoints'][rank]['argv'][-1]='49216f6ce76a637be323dd0aa88c69e7788b4fa8ff13f5663ee103c5e4a22409'
pid='r525-'+hashlib.sha256(json.dumps(plan,separators=(',',':')).encode()).hexdigest()
actual=json.loads((p/'PLAN.json').read_text())
assert actual=={'plan_id':pid,'prompt_tokens_sha256':ph,'prompt_token_ids':tokens}
print('PASS:',pid)
