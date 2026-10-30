"""Verify the stored numerical threshold witness, not the shared GGML decoder."""
import hashlib,math,struct
from pathlib import Path
p=Path(__file__).parent
assert hashlib.sha256((p/'threshold_probe.cpp').read_bytes()).hexdigest()=='c04c7bf956f4d1f8d24ceb93370fcaec2774edbb82136037965152d59c3cd19a'
raw=(p/'RESULT.txt').read_bytes(); assert hashlib.sha256(raw).hexdigest()=='ff8c8a6ca7f1e5e5fc0637e59cbc9b8ff8e9e3e3cbac14de28404f7b18dc57df'
lines=raw.decode().splitlines()
def fields(line): return dict(item.split('=',1) for item in line.split() if '=' in item)
a,b,c,d=fields(lines[0]),fields(lines[1]),fields(lines[2]),fields(lines[4])
assert int(a['quant_bytes_differ'])==1 and b['difference_kind']=='code' and int(b['block'])==16 and int(b['index'])==541
assert int(b['scalar_code'])==-121 and int(b['vector_code'])==-120
assert float(b['scalar_max'])==float(b['vector_max'])
def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
scale=f32(127.0/float(b['scalar_max']))
for kind in ('scalar','vector'):
    assert f32(float(b[kind+'_mid'])*scale)==float(b[kind+'_scaled'])
assert float(b['scalar_scaled']) < -120.5 < float(b['vector_scaled'])
assert int(c['changed_activations'])==1 and int(c['changed_index'])==541 and int(c['original_scalar_violations'])==1169
assert float(c['single_delta_linearity_max_error'])<=1e-12 and math.isclose(float(c['scalar_vs_staged_max_abs']),5.401861017162446e-5,rel_tol=1e-12)
assert int(d['violations'])==0 and float(d['max_abs'])<3e-9
print('PASS: one code straddles -120.5; one activation explains the scalar/staged difference; original failures remain')
