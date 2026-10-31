"""Independently check one fresh claim and FP16/f64 reduction."""
import hashlib,json,struct
from pathlib import Path
p=Path(__file__).parent;plan_bytes=(p/'PLAN.json').read_bytes();plan=json.loads(plan_bytes);plan_hash=hashlib.sha256(plan_bytes).hexdigest();claims=json.loads((p/'CLAIMS.json').read_text());assert len(claims)==1
record=claims[0];raw=record['raw'].encode();assert hashlib.sha256(raw).hexdigest()==record['sha256'];c=json.loads(raw);s=c['stamp'];assert c['status']=='RESERVED_UNKNOWN_NO_REPLAY';assert s=={'run_id':plan['run_id'],'model_revision':'synthetic-r602','plan_id':'r602-'+plan_hash,'layer':0,'batch':0,'epoch':1};assert c['attempt']==1
key=b'VF_NATIVE_BATCH_V1\0'
for text in (s['run_id'],s['model_revision'],s['plan_id']):
 v=text.encode();key+=struct.pack('<Q',len(v))+v
key+=struct.pack('<IQQ',s['layer'],s['batch'],s['epoch']);assert record['file']=='native-'+hashlib.sha256(key).hexdigest()+'.claim.json'
ref=(p/'ORACLE.f64').read_bytes();out=(p/'COLLECTOR.f64').read_bytes();assert len(ref)==len(out)==20480
rows=struct.unpack('<2560d',ref);expect=[]
for v in rows:
 f=struct.unpack('<f',struct.pack('<f',v))[0];h=struct.unpack('<e',struct.pack('<e',f))[0];expect.append(h*0.5+h*0.5)
assert any(v!=0 for v in expect);assert out==struct.pack('<2560d',*expect);assert '1 passed; 0 failed' in (p/'RESULT.txt').read_text()
print('PASS: one non-replayable claim and 2560 exact independently reduced rows; sha256='+hashlib.sha256(out).hexdigest())
