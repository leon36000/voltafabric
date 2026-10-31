"""Independent IEEE conversions and ordered f64 Collector oracle for R594."""
import hashlib, math, struct, sys
from pathlib import Path
W = [0.3420320749282837,0.1739722639322281,0.09913824498653412,0.08050233870744705,0.06733449548482895,0.061283160001039505,0.05222782492637634,0.042672742158174515,0.04069479927420616,0.04014204069972038]
paths = {0:'.runs/vf-r590-work/pc6-75.bin',1:'.runs/vf-r590-work/pc6-12.bin',9:'.runs/vf-r590-work/pc6.bin'}
rows={}
for slot,path in paths.items():
    capture=Path(path).read_bytes()
    assert len(capture)==31760
    vals=struct.unpack('<2560d',capture[11280:])
    rows[slot]=[struct.unpack('<e',struct.pack('<e',struct.unpack('<f',struct.pack('<f',v))[0]))[0] for v in vals]
actual=Path(sys.argv[1]).read_bytes()
assert len(actual)==2560*8
want=[]
for i in range(2560):
    value=0.0
    for slot in range(10): value+=rows[slot][i]*W[slot] if slot in rows else 0.0
    assert math.isfinite(value)
    want.append(value)
expected=struct.pack('<2560d',*want)
if actual != expected:
    differences=sum(a!=b for a,b in zip(struct.unpack('<2560Q',actual),struct.unpack('<2560Q',expected)))
    raise SystemExit(f'FAIL {differences} unequal f64 rows')
assert any(v!=0 for v in want)
print('PASS 2560 exact rows; sha256='+hashlib.sha256(actual).hexdigest())
