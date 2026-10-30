"""Verify the preserved summary; it cannot replace rerunning on the raw trace."""
import hashlib,json,math
from pathlib import Path
p=Path(__file__).parent
expected={'qsa_projection.cpp':'133b2535c913db27bc7c123e48e2f00da28684afbcd6bc81676dc6a242f81f0f',
          'check_spans.py':'12dc6dc01116b1fc700a3d8cc45c9406e885fff9200c424a938c341acf50834b',
          'result.jsonl':'dd6cbe12b57c1fff43e000fdaa5c02d92b6bff1dcaef8946fa82d1da5ae84118',
          'spans.jsonl':'e7cd5ca9e08705075a55afe2caf01475c53ea21ad9127095d8326a2d9d468bde',
          'negative.err':'a15e1c234e8ab3bc4aa2ca39a0cc02ffee45d2a3f75730d5d1a38abe5b415010'}
for name,digest in expected.items(): assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
rows=[json.loads(x) for x in (p/'result.jsonl').read_text().splitlines()]
assert len(rows)==4
for i,(row,count) in enumerate(zip(rows,(1012,460,460,644),strict=True)):
    assert row['projection']==i and row['sampled']==count
    assert math.isfinite(row['max_abs']) and row['max_abs']>=0
    assert math.isfinite(row['max_tolerance_fraction']) and 0<=row['max_tolerance_fraction']<=1
manifest=json.loads((p.parent.parent/'fixtures/r430/layer3.json').read_text())
spans=[json.loads(x) for x in (p/'spans.jsonl').read_text().splitlines()]
assert [x['span'] for x in spans]==['q0','q1','k','v','out']
for x in spans:
    ref=manifest[x['span']]; assert x['sha256']==ref['sha256'] and x['bytes']==ref['bytes']
assert 'mismatch position=91 projection=0 row=0' in (p/'negative.err').read_text()
print('PASS_SUMMARY: 2576 scalar row checks; raw trace is required to rerun')
