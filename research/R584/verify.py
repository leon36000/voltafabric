"""Read-only comparison of the retained real-witness output bytes; no workers."""
import hashlib
import json
import math
import struct
from pathlib import Path

root = Path(__file__).resolve().parents[2]
plan = json.loads((root / 'research/R584/PLAN.json').read_text())
for name, expected in plan['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
ref = (root / '.runs/vf-r574-input/ref-0-255.bin').read_bytes()
assert hashlib.sha256(ref).hexdigest() == '29d8307a5f839c4c44fe9098622ba7faa18cb8a2a0f21781329da1ed36923ddc'
reference = struct.unpack('<2560d', ref[10240:])
files = {'normal': '.runs/vf-r584-work/normal.f32', 'nocontract': '.runs/vf-r584-work/nocontract.f32', 'pc1_previous': '.runs/vf-r575-work/pc1.f32', 'pc6_previous': '.runs/vf-r575-work/pc6.f32'}
values = {}
result = {}
for name, filename in files.items():
    raw = (root / filename).read_bytes()
    values[name] = struct.unpack('<2560f', raw)
    assert all(map(math.isfinite, values[name]))
    errors = [abs(float(x) - y) for x, y in zip(values[name], reference)]
    result[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'strict_violations': sum(e > 1e-7 + 1e-3 * abs(y) for e, y in zip(errors, reference)), 'max_abs_vs_staged': max(errors)}
assert result['pc1_previous']['sha256'] == '4e0c48562ba620b32901b61ee1cf3b35050e7f68f675365d2fe2a7ea3b200bb9'
assert result['pc6_previous']['sha256'] == '777c1f4027aeed7dff8f35962c4ba520dcacad2efc2d60d66e1fc0f54e988e98'
a, b = values['normal'], values['nocontract']
result['same_host_comparison'] = {'different_f32_values': sum(x != y for x, y in zip(a, b)), 'max_abs': max(abs(float(x)-float(y)) for x, y in zip(a, b))}
print(json.dumps(result, sort_keys=True, indent=2))
