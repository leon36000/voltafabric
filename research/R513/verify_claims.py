#!/usr/bin/env python3
"""Check fresh test claim bytes and independent operation-name derivation."""
import hashlib,json,struct
from collections import defaultdict
from pathlib import Path
raw=(Path(__file__).parent/'CLAIMS.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='f74217e96072af78aa1374099615bf58cecb30323e99e131bfb17650a69c96d7'
records=json.loads(raw);assert len(records)==9
seen=set(); runs=defaultdict(list)
for x in records:
    b=x['raw'].encode();assert hashlib.sha256(b).hexdigest()==x['sha256']
    c=json.loads(b);s=c['stamp'];assert c['status']=='RESERVED_UNKNOWN_NO_REPLAY' and s['run_id']==x['directory'] and s['model_revision']=='model-test' and s['plan_id'] in ('r510-plan','r513-nonzero') and s['epoch']==3 and s['layer']==0
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
assert sum(json.loads(x['raw'])['stamp']['plan_id']=='r513-nonzero' for x in records)==6
assert len(runs)==3 and all(sorted(v)==[('batch',0),('batch',1),('evict',0)] for v in runs.values())
print('PASS: 3 distinct runs; 9 non-replayable claims; operation names verified')
