import json,pathlib,struct,subprocess,uuid,hashlib,sys
import torch
from safetensors.torch import load_file
from scripts.resident_artifact import load_stack
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');data=B/'.runs/r272-layer3-owners-20261028';root=B/'.runs/r272-layer3-session-20261028-r2';root.mkdir();(root/'receiver').mkdir();old=B/'.runs/r267-exact-provider-20261028/input.json';source=B/'.runs/r271-layer3-selection-20261028/receipt.json';p=json.loads(old.read_text());receipt=json.loads(source.read_text());manifest=json.loads((data/'owners/manifest.json').read_text());c=load_file(str(data/'capture.safetensors'));assert hashlib.sha256((data/'capture.safetensors').read_bytes()).hexdigest()==receipt['capture_sha256']
ids=receipt['experts'];copies=[];ranks=manifest['ranks'];p['artifacts']=[ranks[str(i)]['path'] for i in range(2)]
for rank in range(2):
 r=ranks[str(rank)];artifact=pathlib.Path(r['path']);assert hashlib.sha256(artifact.read_bytes()).hexdigest()==r['sha256'];data=load_stack(artifact,expected_sha256=r['sha256']);info=data.metadata;assert info['layer']==3 and info['rank']==rank and info['epoch']==1 and [e['id'] for e in info['physical_experts']]==r['experts'];copies.extend({'expert':e,'rank':rank,'local':i} for i,e in enumerate(r['experts']))
p['routing']['experts']=ids;p['routing']['registry']={'epoch':1,'copies':copies};p['routing']['stamp']={'run_id':'r272-'+uuid.uuid4().hex,'model_revision':receipt['source_sha256'] if False else '1355db6a052410cfd62085d94b58866fd0f2c3c5','plan_id':'r272-layer3-cpu','layer':3,'batch':0,'epoch':1};p['routing']['resident_order']={str(i):list(range(len(ranks[str(i)]['experts']))) for i in range(2)}
p['routing']['routes']=[[{'expert':e,'weight':w} for e,w in zip(receipt['routes'][0]['ids'],receipt['routes'][0]['weights'])]]
for i,worker in enumerate(p['workers']):worker['rank']=i;worker['argv'][-1]=str(root/'receiver')
p['inputs']=[c['batch0.input'][0].tolist()];(root/'input.json').write_text(json.dumps(p,indent=2)+'\n')
binary=B/'.runs/r181-target/debug/examples/r261_multipart_cpu_session'
with (root/'coordinator-stderr.txt').open('wb') as err:
 proc=subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,bufsize=0)
 def send(x):proc.stdin.write(json.dumps(x,separators=(',',':'),allow_nan=False).encode()+b'\n');proc.stdin.flush()
 def recv(label):
  raw=proc.stdout.readline();(root/(label+'.json')).write_bytes(raw)
  if not raw:raise RuntimeError('coordinator ended; outcome unknown, never replay')
  value=json.loads(raw)
  if not value.get('ok'):raise RuntimeError('coordinator refused: '+label)
  return value
 send(p);assert recv('ready')['state']=='ready'
 for b in range(2):
  x=c[f'batch{b}.input'].contiguous().numpy().tobytes();bits=list(struct.unpack('<2048H',x));route=receipt['routes'][b];send({'batch':b,'routes':[[{'expert':e,'weight':w} for e,w in zip(route['ids'],route['weights'])]],'inputs_fp16_bits':[bits]});assert recv('batch'+str(b))['batch']==b
 send({'op':'shutdown'});closed=recv('close');assert closed['state']=='closed' and len(closed['close'])==2 and all(v['exit_code']==0 for v in closed['close']);proc.stdin.close();assert proc.wait(timeout=30)==0
(root/'run-receipt.json').write_text(json.dumps({'status':'PASS','run_id':p['routing']['stamp']['run_id'],'batches':2,'gpu':False},indent=2)+'\n');print('PASS',p['routing']['stamp']['run_id'])
