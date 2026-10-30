import base64,hashlib,json,os,pathlib,selectors,struct,subprocess,sys,time,uuid
import torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R/'research/R182'));from durable_hook import DurableSession
BIN=B/'.runs/r181-target/debug/examples/r181_persistent_cpu_session';PY=B/'source/.work/a92-cpu-venv/bin/python';REMOTE='/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r186-onecat-session-crosshost-20261028-r1-script.py';INTERP='/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r178-isolated-1cat-cuda128-20261028-r1/venv/bin/python'
run='r181-'+uuid.uuid4().hex;state={'operation_id':'r186-onecat-session-crosshost-20261028-r1','run_id':run,'status':'STARTED','no_replay':True}
def save():
 p=D/'progress.tmp';p.write_text(json.dumps(state,indent=2)+'\n');os.replace(p,D/'progress.json')
save();torch.set_num_threads(1);assert torch.version.cuda is None
p=json.loads((R/'research/R159/input.json').read_bytes());p['routing']['stamp']['run_id']=run;p['routing']['stamp']['plan_id']='pc1-onecat-pc6-session-cpu';p['workers']=[{'rank':r,'argv':[str(PY),'-B',str(R/'research/R181/worker.py'),run]} for r in (2,9)]
hook=DurableSession(BIN,p,D/'ledger',deadline_seconds=20);state['ledger']=str(hook.directory);state['coordinator_pid']=hook.proc.pid;save()
proc=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','pc1-lan',INTERP,'-B',REMOTE],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0);state['ssh_pid']=proc.pid;state['status']='RUNNING';save()
selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ);buf=b'';count=0;received=[];deadline=time.monotonic()+50
capture=json.loads((B/'.runs/r158-real-resident-owner-cpu/remote.stdout').read_bytes());row=struct.unpack('<2048e',base64.b64decode(capture['results'][0]['row'],validate=True))
while time.monotonic()<deadline:
 events=selector.select(deadline-time.monotonic())
 if not events:break
 part=os.read(proc.stdout.fileno(),65536)
 if not part:break
 buf+=part
 if len(buf)>1000000:raise RuntimeError('oversized remote output')
 while b'\n' in buf:
  line,buf=buf.split(b'\n',1)
  if line.startswith(b'VF_REQUEST '):
   request=json.loads(line[len(b'VF_REQUEST '):]);x=torch.tensor(request['x'],dtype=torch.float16);weights=torch.tensor(request['weights']);ids=torch.tensor(request['ids']);out=hook(None,x,weights,ids,None,None);payload=out.numpy().tobytes();scale=[.75,.375][count];assert torch.equal(out,torch.tensor([[scale*v for v in row]],dtype=torch.float16));received.append(hashlib.sha256(payload).hexdigest());proc.stdin.write(json.dumps({'ok':True,'fp16':base64.b64encode(payload).decode()}).encode()+b'\n');proc.stdin.flush();count+=1;state['completed_batches']=count;save()
  elif line.startswith(b'VF_RECEIPT '):state['remote_receipt']=json.loads(line[len(b'VF_RECEIPT '):]);save()
 if state.get('remote_receipt'):break
if count!=2 or state.get('remote_receipt',{}).get('status')!='SUCCESS':raise RuntimeError('remote outcome incomplete; reconcile without replay')
ret=proc.wait(timeout=10);assert ret==0 and state['remote_receipt']['output_hashes']==received;closed=hook.close();assert all(c['exit_code']==0 and c['bye_execs']==2 for c in closed['close']);state.update(status='SUCCEEDED',ssh_exit=ret,output_hashes=received,cuda_initialized=False,gpu_execution=False);save();(D/'receipt.json').write_text(json.dumps(state,indent=2)+'\n');proc.stdin.close();proc.stdout.close();proc.stderr.close();print(json.dumps(state,sort_keys=True))
