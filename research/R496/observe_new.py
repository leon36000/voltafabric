#!/usr/bin/env python3
"""Observe the R495 expert selection without executing or replaying a worker."""
import hashlib,json,sys
from pathlib import Path
from observe_sc117_experts import run

selection=Path(sys.argv[1]).read_bytes()
if hashlib.sha256(selection).hexdigest()!='de7a6b376da8eac0378364c886b3916abb167e1e14b18bc36e5c262e39894466':
    raise ValueError('selection digest mismatch')
items=json.loads(selection)
if len(items)!=1191 or len({(x['layer'],x['expert']) for x in items})!=len(items):
    raise ValueError('selection count or uniqueness')
expected={(x['layer'],x['expert']):x['sha256'] for x in items}
records=[]; identity=None
for start in range(0,len(items),16):
    batch=items[start:start+16]
    o=run(sys.argv[2],sys.argv[3],[(x['layer'],x['expert']) for x in batch],sys.argv[4])
    ident=(o['index_sha256'],o['pack'],o['source'])
    if identity is None: identity=ident
    if ident!=identity: raise ValueError('pack or source changed between batches')
    for v in o['experts']:
        if not v['matches_source_shard_slices'] or v['sha256']!=expected[(v['layer'],v['expert'])]:
            raise ValueError('selected expert mismatch')
        records.append(v)
if len(records)!=len(expected): raise ValueError('incomplete observations')
print(json.dumps({'scope':'local consistency, not publisher authentication','selection_sha256':hashlib.sha256(selection).hexdigest(),'index_sha256':identity[0],'pack':identity[1],'source':identity[2],'experts':records},sort_keys=True,separators=(',',':')))
