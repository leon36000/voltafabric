"""Check fresh claims, per-expert captures and Rust Collector ordered semantics."""
import hashlib,json,math,re,struct
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
h=lambda b:hashlib.sha256(b).hexdigest()
records={};claims={}
for mode in ('scalar','vector'):
 w=root/'.runs'/('vf-r612-'+mode); raw=(p/('PLAN-'+mode+'.json')).read_bytes();plan=json.loads(raw);ph=h(raw)
 assert (w/'SUBMITTED').read_text().strip()==ph and (w/'EXIT_STATUS').read_text().strip()=='0' and not (w/'error.log').read_bytes()
 x=json.loads((w/'RESULT.json').read_bytes());assert x['run_id']==plan['run_id'] and x['plan_id']=='r612-'+mode+'-'+ph and x['backend']==mode and x['workers_closed']==2
 assert x['residual_sha256']==json.loads((root/'.runs'/('vf-r611-'+mode)/'RESULT.json').read_bytes())['residual_sha256']
 activation=struct.pack('<2560f',*x['activation']);assert h(activation)==x['activation_sha256'];records[mode]=x
 files=list((w/'ledger').glob('*.claim.json'));assert len(files)==1
 cb=files[0].read_bytes();c=json.loads(cb);s=c['stamp'];assert c['status']=='RESERVED_UNKNOWN_NO_REPLAY' and c['attempt']==x['attempt']
 assert s=={'run_id':x['run_id'],'model_revision':'sc117-local-observation','plan_id':x['plan_id'],'layer':0,'batch':0,'epoch':1}
 key=b'VF_NATIVE_BATCH_V1\0'
 for field in ('run_id','model_revision','plan_id'):
  b=s[field].encode();key+=struct.pack('<Q',len(b))+b
 key+=struct.pack('<IQQ',0,0,1);assert files[0].name=='native-'+h(key)+'.claim.json'
 claims[mode]={'file':files[0].name,'sha256':h(cb),'raw':cb.decode()}
a,b=records['scalar'],records['vector'];assert a['ids']==b['ids'] and a['weights']==b['weights'] and a['activation']==b['activation']
wp=(p/'PLAN-witness.json').read_bytes();w=root/'.runs/vf-r612-witness';assert (w/'SUBMITTED').read_text().strip()==h(wp)
witness=json.loads((w/'RESULT.json').read_bytes());assert len(witness)==10 and all(x['exit']==0 and 'quant_bytes_differ=0' in x['output'] and 'violations=0' in x['output'] for x in witness)
pp=(p/'PLAN-pairs.json').read_bytes();plan=json.loads(pp);w=root/'.runs/vf-r612-pairs';assert (w/'SUBMITTED').read_text().strip()==h(pp)
obs=json.loads((w/'RESULT.json').read_bytes());assert [x['expert'] for x in obs]==a['ids'];assert [x['id'] for x in plan['experts']]==a['ids']
scalar_sum=[0.0]*2560;vector_sum=[0.0]*2560;details=[];different_positions=[]
for entry,weight in zip(obs,a['weights']):
 data=(w/('expert-'+str(entry['expert'])+'.bin')).read_bytes();assert len(data)==30720 and h(data)==entry['sha256']
 vector=struct.unpack('<2560f',data[:10240]);scalar=struct.unpack('<2560d',data[10240:]);different=[];max_pre=0.0
 for i,(v,s) in enumerate(zip(vector,scalar)):
  assert math.isfinite(v) and math.isfinite(s)
  sf=struct.unpack('<f',struct.pack('<f',s))[0]
  sh=struct.unpack('<e',struct.pack('<e',sf))[0];vh=struct.unpack('<e',struct.pack('<e',v))[0]
  scalar_sum[i]+=sh*weight;vector_sum[i]+=vh*weight
  max_pre=max(max_pre,abs(float(v)-sf))
  if sh!=vh:different.append(i);different_positions.append(i)
 details.append({'expert':entry['expert'],'fp16_different':len(different),'f32_max_abs':max_pre,'capture_sha256':entry['sha256']})
assert scalar_sum==a['collector'] and vector_sum==b['collector']
diffs=[i for i,(x,y) in enumerate(zip(a['collector'],b['collector'])) if x!=y]
assert set(diffs)==set(different_positions)
summary={'activation_sha256':a['activation_sha256'],'ids':a['ids'],'routes_and_coefficients_equal':True,
 'collector_values':2560,'collector_different_values':len(diffs),'collector_max_absolute_difference':max(abs(x-y) for x,y in zip(a['collector'],b['collector'])),
 'fp16_different_values':sum(x['fp16_different'] for x in details),'experts_with_fp16_difference':sum(x['fp16_different']>0 for x in details),
 'ordered_python_reconstruction_exact':True,'witness_quantized_intermediate_bytes_identical':True,
 'witness_tolerance_violations':0,'experts':details,'claims':claims,
 'shared_ggml_decoders_and_quantizers':True,'independent_complete_expert_oracle':False,'gpu':False,'network_worker':False}
print(json.dumps(summary,indent=2,sort_keys=True))
