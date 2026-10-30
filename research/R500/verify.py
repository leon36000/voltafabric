#!/usr/bin/env python3
"""Verify complete R498 source observations, without running workers."""
import base64,gzip,hashlib,json,runpy
from pathlib import Path
p=Path(__file__).parent
s=runpy.run_path(str(p.parent/'R498/verify.py'))
def unpack(name,expected,count):
    files=sorted((p/(name+'-parts')).glob('part-*.txt')); assert len(files)==count
    raw=gzip.decompress(base64.b64decode(''.join(f.read_text().strip() for f in files),validate=True))
    assert hashlib.sha256(raw).hexdigest()==expected
    return json.loads(raw)
selected=unpack('selection','85a888a7b862e173d0570871f272635c1c8fbc9d150f540216d3fe4434385c3f',3)
result=unpack('observation','8236ebe15cd3469cd4d2f0e6187c5a8e0ba3ca9ef9c0d95f2547d53fa6900596',3)
assert len(selected)==len(result['experts'])==991
expected={(x['layer'],x['expert']):x['sha256'] for x in selected}; assert len(expected)==991
assert expected=={k:v for k,v in s['admitted'].items() if k not in s['observed']}
assert result['selection_sha256']=='85a888a7b862e173d0570871f272635c1c8fbc9d150f540216d3fe4434385c3f'
assert result['index_sha256']=='adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24'
assert result['pack']['bytes']==50292326400 and result['source']['bytes']==55063448064
rows=[line.split() for line in (p.parent.parent/'fixtures/strata_sc117_native_experts_v3.txt').read_text().splitlines() if line and not line.startswith('#')]
assert len(rows)==48
found={}
for x in result['experts']:
    k=(x['layer'],x['expert']); assert k not in found and x['matches_source_shard_slices'] is True
    row=rows[k[0]]; assert x['offset']==int(row[3])+k[1]*int(row[4]) and x['bytes']==int(row[4])
    found[k]=x['sha256']
assert found==expected
assert len(s['observed'].keys() & s['admitted'].keys())+len(found)==len(s['admitted'])==3302
print(json.dumps({'status':'PASS','verified_distinct_experts':3302,'new_independent_observations':len(found)}))
