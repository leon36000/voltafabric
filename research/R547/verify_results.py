"""Verify recorded sweep identities and preserve strict vs explained outcomes."""
import copy,json
from pathlib import Path
p=Path(__file__).parent
result=json.loads((p/'RESULT.json').read_text()); routes=json.loads((p/'ROUTES.json').read_text()); obs=json.loads((p.parent/'R428/OBSERVATION.json').read_text())
experts={x['expert'] for x in obs['experts']}
def validate(x):
 assert x['status']=='DIAGNOSTIC_COMPLETE' and len(experts)==12
 assert x['activation_sha256']==[r['activation_sha'] for r in routes] and len(routes)==2
 s=x['samples']; assert len(s)==24 and {(r['activation'],r['expert']) for r in s}=={(a,e) for a in range(2) for e in experts}
 for r in s:
  assert r['gate_violations']==r['up_violations']==r['mid_violations']==r['envelope_violations']==0
  assert r['status'] in ('STRICT_PASS','STRICT_FAIL_BOUND_EXPLAINED')
  assert (r['status']=='STRICT_PASS')==(r['exit']==0 and r['strict_violations']==0)
  if r['status']!='STRICT_PASS': assert r['exit']==19 and r['strict_violations']>0
 assert {(r['activation'],r['expert'],r['strict_violations']) for r in s if r['status']!='STRICT_PASS'}=={(0,255,1169)}
 assert sum(r['quant_bytes_differ']>0 for r in s)==1
 assert all(len(r['ids'])==10 and set(r['ids']).issubset(experts) for r in routes)
validate(result)
bad=copy.deepcopy(result);bad['samples']=[r for r in bad['samples'] if not (r['activation']==0 and r['expert']==255)]
try:validate(bad)
except AssertionError:pass
else:raise AssertionError('missing failure accepted')
bad=copy.deepcopy(result);r=next(r for r in bad['samples'] if r['activation']==0 and r['expert']==255);r.update(status='STRICT_PASS',exit=0,strict_violations=0)
try:validate(bad)
except AssertionError:pass
else:raise AssertionError('reclassified failure accepted')
print('PASS: 24 samples, 20 routed, strict failure retained, two negative controls rejected')
