import hashlib,json,math,os,struct,sys,urllib.request
from pathlib import Path
p=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/r132-qwen-expert-sample');raw=(p/'plan.json').read_bytes();expected='4ac81d87fbc14853e5dea524ae77bcd0d1e05f97bd03c38edc13741b21b05c78'
if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('plan hash mismatch')
plan=json.loads(raw);result={'status':'STARTED_NO_RETRY','plan_sha256':expected,'samples':{},'full_shard_sha_verified':False,'gpu':False};(p/'intent.json').write_text(json.dumps(result)+'\n')
try:
 total=0
 for name in ('scalars','block_scales','codes'):
  start,end=plan['ranges'][name];length=end-start+1;url=plan['source']+'?download=true&vf_r132='+name;req=urllib.request.Request(url,headers={'User-Agent':'VoltaFabric-bounded-expert-sample','Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
  with urllib.request.urlopen(req,timeout=25) as f:
   cr=f.headers.get('Content-Range','')
   if f.status!=206 or cr!=f'bytes {start}-{end}/{plan["shard_bytes"]}':raise ValueError('range not honored')
   data=f.read(length+1)
  if len(data)!=length:raise ValueError('range length mismatch')
  total+=len(data)
  if total>plan['max_response_bytes']:raise ValueError('sample cap exceeded')
  with (p/(name+'.bin')).open('xb') as o:o.write(data);o.flush();os.fsync(o.fileno())
  result['samples'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'range':[start,end]}
 if total!=plan['total_bytes']:raise ValueError('total mismatch')
 vals=struct.unpack('<6f',(p/'scalars.bin').read_bytes());result['scalar_values']=[float(x) for x in vals];result['scalars_finite_positive']=all(math.isfinite(x) and x>0 for x in vals)
 result['status']='RANGES_FETCHED_NO_NUMERICAL_ORACLE' if result['scalars_finite_positive'] else 'RANGES_FETCHED_SCALAR_CHECK_FAILED'
except Exception as e:result['status']='FAILED_OR_INCOMPLETE_NO_RETRY';result['error_class']=type(e).__name__
(p/'terminal.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result));sys.exit(0 if result['status']=='RANGES_FETCHED_NO_NUMERICAL_ORACLE' else 1)
