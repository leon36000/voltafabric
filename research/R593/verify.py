"""Verify paired CPU scalar output captures, formats and observed source identities."""
import hashlib,json,math,struct
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=root/'research/R593';w=root/'.runs/vf-r593-work'
sha=lambda b:hashlib.sha256(b).hexdigest()
plan=json.loads((p/'PLAN.json').read_bytes());cases_raw=(p/'CASES.json').read_bytes();cases=json.loads(cases_raw)
assert sha(cases_raw)==plan['cases_sha256'] and sha((p/'run.py').read_bytes())==plan['runner_sha256']
assert sha((root/'adapters/native_ggml/scalar_reference.cpp').read_bytes())==plan['reference_sha256']
assert sha((root/'adapters/native_ggml/scalar_reference.hpp').read_bytes())==plan['header_sha256']
assert sha((p/'capture_reference.cpp').read_bytes())==plan['driver_sha256']
assert {(c['gate_up'],c['down']) for c in cases}=={(18,20),(18,42),(21,20),(21,42),(22,20),(22,42),(23,20)}
seed=json.loads((root/'research/R588/RESULT.json').read_bytes())['receipt']['seed_hashes']
records=[]
for c in cases:
    assert c['expert']==0 and c['sha256']==seed[c['layer']][0]
    assert sha((w/'experts'/('layer'+str(c['layer'])+'.bin')).read_bytes())==c['sha256']
    for name,digest in plan['inputs'].items():
        assert sha((root/'.runs/vf-r591-work/inputs'/name).read_bytes())==digest
        filename=f"l{c['layer']}-"+name[:-4]+'.bin'
        a=(w/'results'/filename).read_bytes();b=(w/'pc1-results'/filename).read_bytes()
        assert len(a)==len(b)==20480 and a==b
        vals=struct.unpack('<2560d',a);assert all(map(math.isfinite,vals)) and any(v!=0 for v in vals)
        records.append({'layer':c['layer'],'gate_up':c['gate_up'],'down':c['down'],'input':name,'sha256':sha(a),'pc1_equals_pc6':True})
assert len(records)==28
corrupt=bytearray((w/'results'/'l2-b0-v0.bin').read_bytes());corrupt[0]^=1
assert bytes(corrupt)!=(w/'pc1-results'/'l2-b0-v0.bin').read_bytes()
print(json.dumps({'pairs':7,'inputs':4,'captures':len(records),'all_host_outputs_identical':True,'records':records},sort_keys=True,indent=2))
