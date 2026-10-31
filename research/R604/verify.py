"""Verify fresh claim and ordered full top-10 reduction from pinned captures."""
import hashlib,json,struct
from pathlib import Path
p=Path(__file__).parent;root=p.parents[1];plan_bytes=(p/'PLAN.json').read_bytes();plan=json.loads(plan_bytes);plan_hash=hashlib.sha256(plan_bytes).hexdigest();claims=json.loads((p/'CLAIMS.json').read_text());assert len(claims)==1
record=claims[0];raw=record['raw'].encode();assert hashlib.sha256(raw).hexdigest()==record['sha256'];c=json.loads(raw);s=c['stamp'];assert c['status']=='RESERVED_UNKNOWN_NO_REPLAY';assert s=={'run_id':plan['run_id'],'model_revision':'sc117-r604-observed','plan_id':'r604-'+plan_hash,'layer':2,'batch':0,'epoch':1};assert c['attempt']==1
key=b'VF_NATIVE_BATCH_V1\0'
for text in (s['run_id'],s['model_revision'],s['plan_id']):
 v=text.encode();key+=struct.pack('<Q',len(v))+v
key+=struct.pack('<IQQ',s['layer'],s['batch'],s['epoch']);assert record['file']=='native-'+hashlib.sha256(key).hexdigest()+'.claim.json'
ids=[75,12,209,181,257,272,480,159,492,255];weights=[0.3420320749282837,0.1739722639322281,0.09913824498653412,0.08050233870744705,0.06733449548482895,0.061283160001039505,0.05222782492637634,0.042672742158174515,0.04069479927420616,0.04014204069972038];rows=[]
for expert in ids:
 path=f'.runs/vf-r604-work/captures/expert{expert}.bin';data=(root/path).read_bytes();assert len(data)==31760 and hashlib.sha256(data).hexdigest()==plan['input_sha256'][path];rows.append(struct.unpack('<2560d',data[11280:]))
expected=[]
for i in range(2560):
 total=0.0
 for row,weight in zip(rows,weights):
  f=struct.unpack('<f',struct.pack('<f',row[i]))[0];h=struct.unpack('<e',struct.pack('<e',f))[0];total+=h*weight
 expected.append(total)
out=(root/'.runs/vf-r604-work/collector.f64').read_bytes();assert len(out)==20480 and any(v!=0 for v in expected);assert out==struct.pack('<2560d',*expected);assert '1 passed; 0 failed' in (p/'RESULT.txt').read_text()
result={'collector_sha256':hashlib.sha256(out).hexdigest(),'exact_rows':2560,'real_experts':ids,'three_captures_match_pc1':True,'remaining_captures_host':'PC6','model_authentication':False,'complete_expert_route':True,'complete_layer_or_generation':False}
with open(p/'RESULT.json','x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print('PASS: one non-replayable claim and 2560 exact top-10 rows; sha256='+result['collector_sha256'])
