"""Check the recorded cross-build threshold witness without executing a worker."""
import hashlib
import json
import math
import struct
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parent.parent
plan = json.loads((p / 'PLAN.json').read_bytes())
assert hashlib.sha256((p / 'scalar_probe_initial.cpp').read_bytes()).hexdigest() == plan['input_hashes']['research/R589/scalar_witness.cpp']
detail_plan = json.loads((p / 'DETAIL_PLAN.json').read_bytes())
assert hashlib.sha256((p / 'scalar_witness.cpp').read_bytes()).hexdigest() == detail_plan['sha256']['research/R589/scalar_witness.cpp']
assert hashlib.sha256((root / 'research/R539/RESULT.txt').read_bytes()).hexdigest() == detail_plan['sha256']['research/R539/RESULT.txt']
def fields(line):
    return dict(word.split('=', 1) for word in line.split() if '=' in word)
pc1_lines = (root / 'research/R539/RESULT.txt').read_text().splitlines()
pc6_lines = (p / 'DETAIL_RESULT.txt').read_text().splitlines()
pc1 = fields(next(x for x in pc1_lines if x.startswith('difference_kind=code ')))
pc6 = fields(next(x for x in pc6_lines if x.startswith('witness_index=')))
assert pc1['index'] == pc6['witness_index'] == '541'
assert pc1['scalar_mid'] == pc6['scalar_mid']
assert pc1['scalar_max'] == pc6['scalar_max'] == pc1['vector_max'] == pc6['vector_max']
assert (pc1['scalar_code'], pc1['vector_code'], pc6['scalar_code'], pc6['vector_code']) == ('-121', '-120', '-121', '-121')
def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]
for row in (pc1, pc6):
    for name in ('scalar', 'vector'):
        scaled = f32(f32(127.0 / float(row[name + '_max'])) * float(row[name + '_mid']))
        assert scaled == float(row[name + '_scaled'])
        rounded = math.copysign(math.floor(abs(scaled) + 0.5), scaled)
        assert int(rounded) == int(row[name + '_code'])
assert fields(pc6_lines[0])['quant_bytes_differ'] == '0'
assert fields(pc1_lines[0])['quant_bytes_differ'] == '1'
print('PASS: same scalar witness; vector paths cross opposite sides of the Q8 threshold')
