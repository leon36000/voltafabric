"""Independently reconstruct the retained synthetic native batch and eviction identities."""
import hashlib,json,struct
from pathlib import Path
p=Path(__file__).parent/'claims'; files=sorted(p.glob('*.claim.json'))
assert len(files)==3
for f in files:
 r=json.loads(f.read_text()); s=r['stamp']; ev='binding' in r
 assert r['status']=='RESERVED_UNKNOWN_NO_REPLAY' and r['attempt']>0
 h=bytearray(b'VF_NATIVE_EVICTION_V1\0' if ev else b'VF_NATIVE_BATCH_V1\0')
 for field in ('run_id','model_revision','plan_id'):
  v=s[field].encode();h+=struct.pack('<Q',len(v))+v
 h+=struct.pack('<I',s['layer'])
 if ev: h+=struct.pack('<I',r['binding']['expert'])
 h+=struct.pack('<QQ',s['batch'],s['epoch'])
 digest=hashlib.sha256(h).hexdigest()
 assert f.name==('native-evict-' if ev else 'native-')+digest+'.claim.json'
 if ev: assert r['operation_id']==digest
print('PASS: independently reconstructed 1 batch and 2 eviction synthetic claims')
