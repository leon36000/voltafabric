"""Verify two fresh layer-0 claims and compare exact f32 residuals."""
import hashlib,json,math,struct
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent;root=p.parents[1];results={};values={};claims={}
for mode in ('scalar','vector'):
 w=root/'.runs'/('vf-r611-'+mode);raw=(p/('PLAN-'+mode+'.json')).read_bytes();plan=json.loads(raw);ph=hashlib.sha256(raw).hexdigest()
 assert (w/'SUBMITTED').read_text().strip()==ph and (w/'EXIT_STATUS').read_text().strip()=='0' and not (w/'error.log').read_bytes()
 x=json.loads((w/'RESULT.json').read_bytes()); assert x['run_id']==plan['run_id'] and x['plan_id']=='r611-'+mode+'-'+ph and x['backend']==mode
 assert x['embedding_sha256']==plan['embedding_sha256'] and x['token']==248045 and x['workers_closed']==2
 a=np.asarray(x['residual'],dtype='<f4');assert a.shape==(10240,) and np.isfinite(a).all() and hashlib.sha256(a.tobytes()).hexdigest()==x['residual_sha256'];values[mode]=a;results[mode]=x
 paths=list((w/'ledger').glob('*.claim.json'));assert len(paths)==1
 cb=paths[0].read_bytes();c=json.loads(cb);s=c['stamp'];assert c['status']=='RESERVED_UNKNOWN_NO_REPLAY' and c['attempt']==x['attempt']
 assert s=={'run_id':x['run_id'],'model_revision':'sc117-local-observation','plan_id':x['plan_id'],'layer':0,'batch':0,'epoch':1}
 key=b'VF_NATIVE_BATCH_V1\0'
 for field in ('run_id','model_revision','plan_id'):
  b=s[field].encode();key+=struct.pack('<Q',len(b))+b
 key+=struct.pack('<IQQ',0,0,1);assert paths[0].name=='native-'+hashlib.sha256(key).hexdigest()+'.claim.json'
 claims[mode]={'file':paths[0].name,'sha256':hashlib.sha256(cb).hexdigest(),'raw':cb.decode()}
a,b=values['scalar'],values['vector'];d=np.abs(a.astype(np.float64)-b.astype(np.float64));assert results['scalar']['admissions']==results['vector']['admissions']
summary={'token':248045,'embedding_sha256':results['scalar']['embedding_sha256'],'residual_count':10240,
 'different_values':int(np.count_nonzero(a!=b)),'max_absolute_difference':float(d.max()),
 'admissions_equal':True,'dynamic_admissions':len(results['scalar']['admissions']),
 'residual_sha256':{m:results[m]['residual_sha256'] for m in results},'claims':claims,
 'gpu':False,'network_worker':False,'complete_model_or_independent_expert_oracle':False}
print(json.dumps(summary,indent=2,sort_keys=True))
