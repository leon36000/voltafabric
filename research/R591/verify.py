"""Verify fresh local captures and reject any cross-host byte difference."""
import hashlib
import json
import math
import struct
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'research/R591';work=root/'.runs/vf-r591-work'
plan=json.loads((p/'PLAN.json').read_bytes()); raw=(p/'INPUTS.json').read_bytes(); records=json.loads(raw)
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha(raw)==plan['input_manifest_sha256'] and sha((p/'run.py').read_bytes())==plan['runner_sha256']
assert sha((root/'research/R590/scalar_capture_executed.cpp').read_bytes())==plan['source_sha256']
assert len(records)==16 and len({r['file'] for r in records})==16
files=[]
for rec in records:
    assert sha((work/'inputs'/rec['file']).read_bytes()) == rec['sha256']
    for expert in (12,75,255):
        name=f"{rec['file'][:-4]}-e{expert}.bin"
        a=(work/'results'/name).read_bytes();b=(work/'pc1-results'/name).read_bytes()
        assert len(a)==len(b)==31760 and a==b, name
        gate=struct.unpack_from('<640f',a,2920);up=struct.unpack_from('<640f',a,5480)
        mid=struct.unpack_from('<640f',a,8040);out=struct.unpack_from('<2560d',a,11280)
        assert all(math.isfinite(x) for x in gate+up+mid+out)
        assert any(x!=0 for x in out)
        if rec['file']=='b0-v0.f32':
            prev='pc6.bin' if expert==255 else f'pc6-{expert}.bin'
            assert a==(root/'.runs/vf-r590-work'/prev).read_bytes()
        files.append({'file':name,'sha256':sha(a),'bytes':len(a),'pc1_equals_pc6':True})
assert len(files)==48
broken=bytearray((work/'results'/files[0]['file']).read_bytes());broken[8040]^=1
assert bytes(broken)!=(work/'pc1-results'/files[0]['file']).read_bytes()
print(json.dumps({'cases':len(files),'input_count':len(records),'expert_count':3,'all_bit_identical':True,'cases_sha256':sha(json.dumps(files,sort_keys=True,separators=(',',':')).encode()),'captures':files},indent=2,sort_keys=True))
