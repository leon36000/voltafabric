"""No-worker protocol checks; no session identity is admitted."""
import json,subprocess
from pathlib import Path
b=Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');p=json.loads((b/'r281-layer1-session-r1/input.json').read_text());p['routing']['stamp']['run_id']='r325-no-worker-test';p['workers']=[];p['artifacts']=[];binary=b/'r325-correlated-errors-r1/target/debug/examples/r261_multipart_cpu_session';rid='a'*32
cases=[('correlated',{'request_id':rid,'init':p},rid),('legacy',p,None),('invalid_id',{'request_id':'bad','init':p},None)];results=[]
for name,body,expected in cases:
 c=subprocess.run([str(binary)],input=json.dumps(body)+'\n',text=True,capture_output=True,timeout=10);reply=json.loads(c.stdout);assert c.returncode==1 and reply['ok'] is False and reply.get('request_id')==expected,(name,reply);results.append({'case':name,'request_id_matches':reply.get('request_id')==expected,'error':reply['error']})
print(json.dumps({'status':'PASS','cases':results}))
