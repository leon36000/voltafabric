"""Generate 16 bounded deterministic f32 activation variants from R428 inputs."""
import hashlib
import json
import math
import struct
from pathlib import Path
root = Path(__file__).resolve().parents[2]
work = root/'.runs/vf-r591-work'
first = (root/'.runs/vf-r574-input/activation0.f32').read_bytes()
second_ref = (work/'ref-1-12.bin').read_bytes()
assert hashlib.sha256(first).hexdigest() == 'c8a0744a87fade2652acfb34c802a49354379f558791bdf63e53f1f124bcc88a'
assert hashlib.sha256(second_ref).hexdigest() == 'b0236fa9c0ddbbd15a26d3f05718104e42dfd0fcc2ceb31a1776f06ba79c63fb'
assert len(first)==10240 and len(second_ref)==30720
bases = [struct.unpack('<2560f', first), struct.unpack('<2560f', second_ref[:10240])]
scales = (1.0, 0.5, -1.0, 1.0001, 0.9999, 2.0, 0.125, -0.75)
records = []
for base_index, base in enumerate(bases):
    for variant, scale in enumerate(scales):
        vals = [x*scale for x in base]
        assert all(math.isfinite(x) for x in vals)
        data = struct.pack('<2560f', *vals)
        name=f'b{base_index}-v{variant}.f32'
        with (work/'inputs'/name).open('xb') as f:
            f.write(data)
        records.append({'file':name,'source':base_index,'scale':scale,'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps(records,indent=2,sort_keys=True))
