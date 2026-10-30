"""Check retained decoder-comparison record; raw weights are required to rerun."""
import hashlib,json
from pathlib import Path
p=Path(__file__).parent
hashes={'ggml_rows.cpp':'ac04d7154591dbf6f8f1bc016ff74b32d911d2231f1b789c83bd79b11b58d308',
'compare_decoders.py':'aaee1ba0f4777954656a8ae37a74a1113c7f3c1d2e297799adafb9d7fd4e68c2',
'result.jsonl':'6cc18df61a15b18e8181c586c22a92946c3dc0129ab63ea117e6f61770144ffe',
'spans.jsonl':'e7cd5ca9e08705075a55afe2caf01475c53ea21ad9127095d8326a2d9d468bde',
'negative.txt':'65af71a010bc7a90d0344bb6c52c255721b1e8a6acc5c3a6efd21495d3e9335b'}
for name,digest in hashes.items(): assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
rows=[json.loads(x) for x in (p/'result.jsonl').read_text().splitlines()]
assert [(r['projection'],r['rows'],r['values'],r['bit_equal']) for r in rows]==[('q',12288,31457280,True),('k',512,1310720,True),('v',512,1310720,True),('out',2560,15728640,True)]
assert sum(r['values'] for r in rows)==49807360
manifest=json.loads((p.parent.parent/'fixtures/r430/layer3.json').read_text())
spans=[json.loads(x) for x in (p/'spans.jsonl').read_text().splitlines()]
assert [x['span'] for x in spans]==['q0','q1','k','v','out']
for x in spans:
    ref=manifest[x['span']]; assert x['sha256']==ref['sha256'] and x['bytes']==ref['bytes']
assert 'decoder mismatch projection=q row=0' in (p/'negative.txt').read_text()
print('PASS_SUMMARY: 15872 rows, 49807360 bit-equal decoded values; weights required to rerun')
