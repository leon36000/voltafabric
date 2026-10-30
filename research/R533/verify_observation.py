"""Verify new source-slice observations against the independently reconstructed selection."""
import base64, gzip, hashlib, json, runpy
from pathlib import Path
p=Path(__file__).parent
v=runpy.run_path(str(p/'verify.py'))
meta=json.loads((p/'METADATA.json').read_text())
def unpack(name):
    files=sorted((p/(name+'-parts')).glob('part-*.txt'))
    assert len(files)==meta[name+'_parts']
    raw=gzip.decompress(base64.b64decode(''.join(f.read_text().strip() for f in files),validate=True))
    assert hashlib.sha256(raw).hexdigest()==meta[name+'_sha256']
    return json.loads(raw)
selection=unpack('selection'); observation=unpack('observation')
expected={(x['layer'],x['expert']):x['sha256'] for x in selection}
assert len(expected)==len(selection)==len(v['selection'])
assert selection==v['selection']
assert observation['selection_sha256']==meta['selection_sha256']
assert observation['index_sha256']=='adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24'
assert observation['pack']['bytes']==50292326400 and observation['source']['bytes']==55063448064
rows=[s.split() for s in (p.parent.parent/'fixtures/strata_sc117_native_experts_v3.txt').read_text().splitlines() if s and not s.startswith('#')]
found={}
for x in observation['experts']:
    k=(x['layer'],x['expert']); assert k not in found and x['matches_source_shard_slices'] is True
    row=rows[k[0]]; assert x['offset']==int(row[3])+k[1]*int(row[4]) and x['bytes']==int(row[4])
    found[k]=x['sha256']
assert found==expected
print(json.dumps({'status':'PASS','distinct_experts':len(v['admitted']),'new_observations':len(found),'matched_prefix':73,'positions':len(v['r']['input_tokens'])}))
