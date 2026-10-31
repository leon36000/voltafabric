"""Validate the bounded full-head diagnostic receipt, not the GGML arithmetic."""
import hashlib,json,math
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'research/R621';w=root/'.runs/vf-r621-work'
sha=lambda b:hashlib.sha256(b).hexdigest()
plan=(p/'PLAN.json').read_bytes()
assert (w/'SUBMITTED').read_text().strip()==sha(plan)
assert (w/'EXIT_STATUS').read_bytes()==b'0\n' and (w/'STDERR').read_bytes()==b''
raw=(w/'STDOUT').read_bytes();assert sha(raw)=='e8ebeee1e60fb6a2f7ff74cfb5c474fbb6e6dc829f0f676269a314b01af73f4f'
x=json.loads(raw)
assert set(x)=={'rows','observed_argmax','reference_argmax','violations','max_abs','max_tolerance_fraction'}
assert x['rows']==248320 and x['observed_argmax']==x['reference_argmax']==710 and x['violations']==0
assert all(math.isfinite(x[k]) and x[k]>=0 for k in ('max_abs','max_tolerance_fraction'))
assert x['max_tolerance_fraction']<=1
negative=(w/'NEGATIVE_RESULT').read_bytes()
assert sha(negative)=='0fac05d73fc669558eb7a0d6c2c02ecdb46b85a94242058309e116aee5e15289'
n=json.loads(negative)
assert n['rows']==248320 and n['violations']==1 and n['observed_argmax']==n['reference_argmax']==710
assert (w/'NEGATIVE_SUBMITTED').read_text().strip()==sha(plan)+' 89b9aab70a949b92eff6354f620dbebdbbaafa3887ea7f6c0e1f03660b9351c8'
x['negative_control_violations']=n['violations'];x['negative_result_sha256']=sha(negative)
x['plan_sha256']=sha(plan);x['stdout_sha256']=sha(raw)
print(json.dumps(x,indent=2,sort_keys=True))
