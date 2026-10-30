#!/usr/bin/env python3
"""Check fresh test claim bytes and independent operation-name derivation."""
import hashlib,json,struct
from collections import defaultdict
from pathlib import Path
raw=(Path(__file__).parent/'CLAIMS.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='97378e5e313c195ada9493f5a05f01d613a7e50050cfe1397f3ff41031dc24ca'
records=json.loads(raw);assert len(records)==9
seen=set(); runs=defaultdict(list)
for x in records:
    b=x['raw'].encode();assert hashlib.sha256(b).hexdigest()==x['sha256']
    c=json.loads(b);s=c['stamp'];assert c['status']=='RESERVED_UNKNOWN_NO_REPLAY' and s['run_id']==x['directory'] and s['model_revision']=='model-test' and s['plan_id']=='r510-plan' and s['epoch']==3 and s['layer']==0
    base=int(x['directory'].rsplit('-',1)[1]);assert c['attempt']==base+s['batch']
    ev='binding' in c
    key=b'VF_NATIVE_EVICTION_V1\0' if ev else b'VF_NATIVE_BATCH_V1\0'
    for v in (s['run_id'],s['model_revision'],s['plan_id']):
        bts=v.encode(); key+=struct.pack('<Q',len(bts))+bts
    key+=struct.pack('<I',s['layer'])
    if ev:
        z=c['binding']; assert z['rank']==2 and z['expert']==1 and z['local']==1 and s['batch']==0
        key+=struct.pack('<I',z['expert'])
    key+=struct.pack('<QQ',s['batch'],s['epoch'])
    digest=hashlib.sha256(key).hexdigest(); filename=('native-evict-' if ev else 'native-')+digest+'.claim.json'
    assert x['file']==filename and (not ev or c['operation_id']==digest) and (x['directory'],filename) not in seen
    seen.add((x['directory'],filename));runs[x['directory']].append(('evict' if ev else 'batch',s['batch']))
assert len(runs)==3 and all(sorted(v)==[('batch',0),('batch',1),('evict',0)] for v in runs.values())
print('PASS: 3 distinct runs; 9 non-replayable claims; operation names verified')
