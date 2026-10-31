"""Check fresh claim, historical PC1 captures and independent FP16/f64 reduction."""
import hashlib,json,struct
from pathlib import Path
p=Path(__file__).parent;root=p.parents[1];plan_bytes=(p/'WORKER-PLAN.json').read_bytes();plan=json.loads(plan_bytes);plan_hash=hashlib.sha256(plan_bytes).hexdigest();claims=json.loads((p/'CLAIMS.json').read_text());assert len(claims)==1
w=root/'.runs/vf-r626-worker';assert (w/'SUBMITTED').read_text().strip()==plan_hash and (w/'exit-status').read_text().strip()=='0' and not (w/'stderr').read_bytes()
record=claims[0];raw=record['raw'].encode();assert hashlib.sha256(raw).hexdigest()==record['sha256'];c=json.loads(raw);s=c['stamp'];assert c['status']=='RESERVED_UNKNOWN_NO_REPLAY';assert s=={'run_id':plan['run_id'],'model_revision':'sc117-r603-observed','plan_id':'r603-'+plan_hash,'layer':2,'batch':0,'epoch':1};assert c['attempt']==1
key=b'VF_NATIVE_BATCH_V1\0'
for text in (s['run_id'],s['model_revision'],s['plan_id']):
 v=text.encode();key+=struct.pack('<Q',len(v))+v
key+=struct.pack('<IQQ',s['layer'],s['batch'],s['epoch']);assert record['file']=='native-'+hashlib.sha256(key).hexdigest()+'.claim.json'
captures=[('.runs/vf-r590-work/pc1-75.bin','43c6043458558d33a359170aa3a6deffbda0ae388008231622f84f9ac5ddac3d'),('.runs/vf-r590-work/pc1-12.bin','927784846bbddb3407f551cf33a82a998d9b74d4cefe343ec79f074fdb390fe5'),('.runs/vf-r590-work/pc1.bin','8bbbf4a9bd630873700f39bb6eb69e6fc67e43fa707e5491e6ca337d1ab2cd52')]
rows=[]
for path,expected_hash in captures:
 data=(root/path).read_bytes();assert len(data)==31760 and hashlib.sha256(data).hexdigest()==expected_hash
 rows.append(struct.unpack('<2560d',data[11280:]))
weights=[0.3420320749282837,0.1739722639322281,0.04014204069972038];expected=[]
for i in range(2560):
 total=0.0
 for row,weight in zip(rows,weights):
  f=struct.unpack('<f',struct.pack('<f',row[i]))[0];h=struct.unpack('<e',struct.pack('<e',f))[0];total+=h*weight
 expected.append(total)
out=(root/'.runs/vf-r626-worker/collector.f64').read_bytes();assert len(out)==20480 and any(v!=0 for v in expected);assert out==struct.pack('<2560d',*expected);assert '1 passed; 0 failed' in (p/'RESULT.txt').read_text()
result={'collector_sha256':hashlib.sha256(out).hexdigest(),'exact_rows':2560,'real_experts':[75,12,255],'capture_origin':'R590 prior PC1 scalar captures','routes_omitted':7,'model_authentication':False,'full_layer_or_generation':False}
assert result == json.loads((p/'WORKER-RESULT.json').read_text())
print('PASS: one non-replayable claim and 2560 exact rows; sha256='+result['collector_sha256'])
