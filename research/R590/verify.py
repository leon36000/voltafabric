"""Independent parsing of bounded scalar captures; no model or worker execution."""
import hashlib
import json
import math
import struct
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parent.parent
plan = json.loads((p/'PLAN.json').read_bytes())
multi = json.loads((p/'MULTI_PLAN.json').read_bytes())
def sha(data): return hashlib.sha256(data).hexdigest()
assert sha((p/'scalar_capture_executed.cpp').read_bytes()) == plan['source_sha256'] == multi['source_sha256']
assert (p/'scalar_capture_executed.cpp').read_text().replace('one IQ3_S gate/up + IQ4_NL down expert','one IQ2_S gate/up + IQ4_NL down expert') == (p/'scalar_capture.cpp').read_text()
for expert in (12, 75):
    assert sha((root/f'.runs/vf-r590-work/expert{expert}.bin').read_bytes()) == multi['expert_sha256_by_id'][str(expert)]
assert sha((root/'.runs/vf-r574-input/expert255.bin').read_bytes()) == plan['expert_sha256']
assert sha((root/'.runs/vf-r574-input/activation0.f32').read_bytes()) == plan['activation_sha256']
result = {}
for expert, files in {255: ('pc1.bin','pc6.bin'), 12: ('pc1-12.bin','pc6-12.bin'), 75: ('pc1-75.bin','pc6-75.bin')}.items():
    a, b = [(root/'.runs/vf-r590-work'/name).read_bytes() for name in files]
    assert len(a)==len(b)==31760
    assert a==b, f'cross-host scalar capture differs for {expert}'
    gate=struct.unpack_from('<640f', a, 2920)
    up=struct.unpack_from('<640f', a, 5480)
    mid=struct.unpack_from('<640f', a, 8040)
    out=struct.unpack_from('<2560d', a, 11280)
    assert all(math.isfinite(x) for x in gate+up+mid+out)
    assert any(x != 0 for x in out)
    result[str(expert)]={'capture_sha256':sha(a),'bytes':len(a),'host_bytes_identical':True,'gate_count':len(gate),'up_count':len(up),'mid_count':len(mid),'output_count':len(out)}
    if expert==255:
        assert mid[541] == -0.056449014693498611
        assert struct.unpack_from('<b',a,10600+16*34+2+29)[0] == -121
        for host in ('pc1','pc6'):
            raw=(root/f'.runs/vf-r575-work/{host}.f32').read_bytes()
            original={'pc1':'4e0c48562ba620b32901b61ee1cf3b35050e7f68f675365d2fe2a7ea3b200bb9','pc6':'777c1f4027aeed7dff8f35962c4ba520dcacad2efc2d60d66e1fc0f54e988e98'}[host]
            assert sha(raw)==original
            vector=struct.unpack('<2560f',raw); diff=[abs(x-y) for x,y in zip(out,vector)]
            result[str(expert)][host+'_vector_vs_scalar']={'violations':sum(d>1e-7+1e-3*abs(x) for d,x in zip(diff,out)),'max_abs':max(diff)}
    corrupt=bytearray(a);corrupt[8040]^=1
    assert bytes(corrupt)!=b
print(json.dumps(result,sort_keys=True,indent=2))
