import base64,hashlib,json,os,pathlib,selectors,struct,subprocess,sys,time,uuid
import torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(B/'.runs/r207-gate-runner-20261028'));from durable8 import DurableSession
BIN=B/'.runs/r181-target/debug/examples/r181_persistent_cpu_session';PY=B/'source/.work/a92-cpu-venv/bin/python';REMOTE='/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r212-constructed-moe-block-20261028-r1-script.py';INTERP='/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r178-isolated-1cat-cuda128-20261028-r1/venv/bin/python'
run='r202-'+uuid.uuid4().hex;state={'operation_id':'r212-constructed-moe-block-20261028-r1','run_id':run,'status':'STARTED','no_replay':True}
def save():
 p=D/'progress.tmp';p.write_text(json.dumps(state,indent=2)+'\n');os.replace(p,D/'progress.json')
save();torch.set_num_threads(1);assert torch.version.cuda is None
p=json.loads((R/'research/R159/input.json').read_bytes());p['routing']['stamp']['run_id']=run;p['routing']['stamp']['plan_id']='pc1-eight-runner-pc6-session-cpu';p['workers']=[{'rank':rank,'argv':[str(PY),'-B',str(B/'.runs/r207-gate-runner-20261028/worker.py'),run,str(D/'hashes.json'),str(D)]} for rank in (2,9)]
p['routing']['stamp']['model_revision']='1355db6a052410cfd62085d94b58866fd0f2c3c5';p['routing']['experts']=[198, 204, 109, 121, 171, 76, 166, 173];p['routing']['routes']=[[{'expert':e,'weight':w} for e,w in zip([198, 204, 109, 121, 171, 76, 166, 173],[0.14534828066825867, 0.14492307603359222, 0.13614262640476227, 0.12487051635980606, 0.12032157182693481, 0.11314209550619125, 0.10886082798242569, 0.10639100521802902])]];p['routing']['registry']['copies']=[{'expert':e,'rank':2 if i<4 else 9,'local':i%4} for i,e in enumerate([198, 204, 109, 121, 171, 76, 166, 173])];p['routing']['resident_order']={'2':list(range(4)),'9':list(range(4))};p['artifacts']=[str(B/f'.runs/r206-gate-route-20261028/owners/rank{rank}.safetensors') for rank in (2,9)];p['fp16_completions']=True
hook=DurableSession(BIN,p,D/'ledger',deadline_seconds=30);state['ledger']=str(hook.directory);state['coordinator_pid']=hook.proc.pid;save()
proc=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','pc1-lan',INTERP,'-B',REMOTE],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0);state['ssh_pid']=proc.pid;state['status']='RUNNING';save()
selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ);buf=b'';count=0;received=[];deadline=time.monotonic()+50
sys.path.insert(0,str(R/'research/R201'));from verify import accumulate;from nvfp4_oracle import expert_forward
source=B/'.runs/r206-gate-route-20261028/selected.safetensors';assert hashlib.sha256(source.read_bytes()).hexdigest()=='0e597ecf47a568d6066ac2490bb09b37f615a1f725f9bdac9364cab4c195928b';errors=[]
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
   request=json.loads(line[len(b'VF_REQUEST '):]);x=torch.tensor(request['x'],dtype=torch.float16);weights=torch.tensor(request['weights']);ids=torch.tensor(request['ids']);out=hook(None,x,weights,ids,None,None);payload=out.numpy().tobytes();routes=[[{'expert':int(e),'weight':float(w)} for e,w in zip(ids[0].tolist(),weights[0].tolist())]];expected=accumulate(routes,[x[0]],lambda expert,row:expert_forward(source,0,expert,row));error=float((out.to(torch.float64)-expected).abs().max());errors.append(error);assert error<=0.0001;received.append(hashlib.sha256(payload).hexdigest());proc.stdin.write(json.dumps({'ok':True,'fp16':base64.b64encode(payload).decode()}).encode()+b'\n');proc.stdin.flush();count+=1;state['completed_batches']=count;save()
  elif line.startswith(b'VF_RECEIPT '):state['remote_receipt']=json.loads(line[len(b'VF_RECEIPT '):]);save()
 if state.get('remote_receipt'):break
if count!=2 or state.get('remote_receipt',{}).get('status')!='SUCCESS':raise RuntimeError('remote outcome incomplete; reconcile without replay')
ret=proc.wait(timeout=10);assert ret==0 and state['remote_receipt']['output_hashes']==received;closed=hook.close();assert all(c['exit_code']==0 and c['bye_execs']==2 for c in closed['close']);state.update(status='SUCCEEDED',ssh_exit=ret,output_hashes=received,oracle_max_abs_errors=errors,cuda_initialized=False,gpu_execution=False);save();(D/'receipt.json').write_text(json.dumps(state,indent=2)+'\n');proc.stdin.close();proc.stdout.close();proc.stderr.close();print(json.dumps(state,sort_keys=True))
