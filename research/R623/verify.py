"""Verify receipt consistency and the generated head's scalar comparison."""
import hashlib,json,math
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=root/'research/R623';w=root/'.runs/vf-r623-work'
h=lambda data:hashlib.sha256(data).hexdigest();plan=(p/'PLAN.json').read_bytes();cfg=json.loads(plan)
assert (w/'SUBMITTED').read_text().strip()==h(plan)
assert (w/'EXIT_STATUS').read_bytes()==b'0\n' and (w/'STDERR').read_bytes()==b''
x=json.loads((w/'STDOUT').read_bytes());assert x['rows']==248320 and x['violations']==0 and x['observed_argmax']==x['reference_argmax']==9419
assert all(math.isfinite(x[k]) and x[k]>=0 for k in ('max_abs','max_tolerance_fraction')) and x['max_tolerance_fraction']<=1
r=json.loads((root/'research/R622/RESULT.json').read_text());head=(root/'.runs/vf-r622-work/head.bin').read_bytes()
assert h(head)==r['head_sha256']==cfg['file_sha256'][cfg['head']] and r['generated_token_ids']==[9419]
x['plan_sha256']=h(plan);x['head_sha256']=h(head);x['generation_receipt_sha256']=r['receipt_sha256'];x['generation_plan_id']=r['plan_id']
print(json.dumps(x,indent=2,sort_keys=True))
