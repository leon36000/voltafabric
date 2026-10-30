"""CPU research bridge: live attention/collector acknowledgement; not durable KV."""
import json,pathlib,subprocess,uuid,sys,torch
from safetensors.torch import load_file
from adapters.session_bridge import DurableSession
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');root=B/'.runs/r275-live-decoder-20261028-r1';root.mkdir();(root/'receiver').mkdir();p=json.loads((B/'.runs/r272-layer3-session-20261028-r2/input.json').read_text());stamp=p['routing']['stamp'];stamp['run_id']='r275-'+uuid.uuid4().hex;stamp['plan_id']='r275-live-decoder';identity={k:stamp[k] for k in ('run_id','model_revision','plan_id','layer','epoch')}
from scripts.resident_artifact import load_stack
import copy,hashlib
manifest=json.loads((B/'.runs/r275-layer3-full-20261028/owners/manifest.json').read_text());assert manifest['source_sha256']=='2802aa69c3254c8b7c8944448ec3e9f6116e9633c504681ea5b8b4726bed864f' and manifest['layer']==3
copies=[];p['artifacts']=[];template=p['workers'][0];p['workers']=[]
for rank in range(16):
 entry=manifest['ranks'][str(rank)];path=pathlib.Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'];m=load_stack(path,expected_sha256=entry['sha256']).metadata;ids=[a['id'] for a in m['physical_experts']];assert ids==list(range(rank*16,(rank+1)*16)) and m['rank']==rank and m['layer']==3 and m['epoch']==1
 copies.extend({'expert':e,'rank':rank,'local':j} for j,e in enumerate(ids));p['artifacts'].append(str(path));worker=copy.deepcopy(template);worker['rank']=rank;worker['argv'][-1]=str(root/'receiver');p['workers'].append(worker)
p['routing']['experts']=list(range(256));p['routing']['registry']={'epoch':1,'copies':copies};p['routing']['resident_order']={str(i):list(range(16)) for i in range(16)}
(root/'input.json').write_text(json.dumps(p,indent=2)+'\n')
peer='/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs'
cmd='cd '+peer+'/r271-selection-code && CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 MODEL_SHARD='+peer+'/r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors ../r178-isolated-1cat-cuda128-20261028-r1/venv/bin/python -B attention_agent_r274.py'
err=(root/'attention-stderr.txt').open('wb');proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pc1-lan',cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,bufsize=0)
def send(x):proc.stdin.write(json.dumps(x,separators=(',',':')).encode()+b'\n');proc.stdin.flush()
def receive():
 line=proc.stdout.readline()
 if not line or len(line)>1024*1024:raise RuntimeError('attention agent ended; state UNKNOWN')
 return json.loads(line)
results=[];session=None
try:
 send(identity);assert receive()['state']=='ready';session=DurableSession(B/'.runs/r181-target/debug/examples/r261_multipart_cpu_session',p,root/'ledger',deadline_seconds=90)
 old=json.loads((B/'.runs/r271-layer3-selection-20261028/receipt.json').read_text());cap=load_file(str(B/'.runs/r272-layer3-owners-20261028/capture.safetensors'))
 for b in range(2):
  send({'op':'stage','batch':b});row=receive();assert row['state']=='staged' and row['batch']==b
  x=torch.tensor(row['x'],dtype=torch.float16);ids=torch.tensor(row['ids'],dtype=torch.int64);weights=torch.tensor(row['weights'],dtype=torch.float32)
  assert torch.equal(x,cap[f'batch{b}.input']) and ids[0].tolist()==old['routes'][b]['ids']
  output=session(None,x,weights,ids,None,None)
  send({'op':'commit','batch':b,'routed':output.tolist()});ack=receive();assert ack['state']=='committed' and ack['batch']==b and ack['committed']==b+1 and len(ack['layer_output'][0])==2048
  results.append({'batch':b,'output':output[0].tolist(),'ids':ids[0].tolist(),'weights':weights[0].tolist(),'layer_output':ack['layer_output'][0],'shared_output':ack['shared_output'][0]})
 assert receive()=={'state':'closed','committed':2};proc.stdin.close();assert proc.wait(timeout=30)==0
 closed=session.close();assert len(closed['close'])==16;out={'status':'PASS','identity':identity,'batches':results,'ledger':str(session.directory),'attention_exit':0,'gpu':False};(root/'receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'PASS','run_id':identity['run_id'],'batches':2,'ledger':str(session.directory)}))
except BaseException as e:
 # Never send commit after uncertain collection; an EOF poisons the attention state.
 try:proc.stdin.close()
 except BaseException:pass
 (root/'failure.json').write_text(json.dumps({'status':'UNKNOWN','identity':identity,'error':str(e)[:400],'attention_pid':proc.pid,'session_pid':session.proc.pid if session else None})+'\n')
 raise
finally:err.close()
