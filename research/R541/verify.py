"""Validate recorded full-row check; raw trace is needed to rerun it."""
import hashlib,json,math
from pathlib import Path
p=Path(__file__).parent
hashes={'qsa_all_rows.cpp':'c076af506fffdf391d32df79e3d8ad5d8e102452248c51353343b2677b9ad891',
'result.jsonl':'e8cac11bf0bbc0037f3a15fce1a8af04963fdf3263d2e625a10174f3e6e50c67',
'spans.jsonl':'e7cd5ca9e08705075a55afe2caf01475c53ea21ad9127095d8326a2d9d468bde',
'negative-verified.err':'a15e1c234e8ab3bc4aa2ca39a0cc02ffee45d2a3f75730d5d1a38abe5b415010'}
for name,digest in hashes.items(): assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
rows=[json.loads(x) for x in (p/'result.jsonl').read_text().splitlines()]
assert len(rows)==4
for i,(row,count) in enumerate(zip(rows,(12288,512,512,2560),strict=True)):
    assert row['projection']==i and row['rows']==count and row['positions']==92 and row['comparisons']==count*92
    assert math.isfinite(row['max_abs']) and row['max_abs']>=0
    assert math.isfinite(row['max_tolerance_fraction']) and 0<=row['max_tolerance_fraction']<=1
assert sum(x['comparisons'] for x in rows)==1460224
manifest=json.loads((p.parent.parent/'fixtures/r430/layer3.json').read_text())
spans=[json.loads(x) for x in (p/'spans.jsonl').read_text().splitlines()]
assert [x['span'] for x in spans]==['q0','q1','k','v','out']
for x in spans:
    ref=manifest[x['span']]; assert x['sha256']==ref['sha256'] and x['bytes']==ref['bytes']
assert 'mismatch position=91 projection=0 row=0' in (p/'negative-verified.err').read_text()
print('PASS_SUMMARY: 1460224 scalar row checks; raw trace is required to rerun')
