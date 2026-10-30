import copy,json,os,pathlib,subprocess,sys,uuid
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs')
code=base/'r309-old-code-r1'; root=base/'r309-old-run-r1';root.mkdir(mode=0o700);receiver=root/'receiver';receiver.mkdir()
sys.path.insert(0,str(code));sys.path.insert(1,str(base/'r276-package-r1'));sys.path.insert(2,str(base));sys.path.insert(3,str(base/'r299-code-r1'))
from adapters.guarded_fleet import GuardedFleet
from adapters.resource_admission import AdmissionError,ResourcePolicy
p=json.loads((base/'r281-layer1-session-r1/input.json').read_text()); manifest=json.loads((base/'r283-four-layer-owners-r1/layer0/manifest.json').read_text());run='r309-old-'+uuid.uuid4().hex
p['artifacts']=[manifest['ranks']['0']['path']];p['workers']=p['workers'][:1];p['routing']['experts']=list(range(64));p['routing']['registry']={'epoch':1,'copies':[{'expert':i,'rank':0,'local':i} for i in range(64)]};p['routing']['resident_order']={'0':list(range(64))};p['routing']['stamp'].update(run_id=run,plan_id='r309-old-controlled',layer=0,batch=0);p['routing']['routes']=[[{'expert':5,'weight':.625},{'expert':7,'weight':.375}]]
# The real local-reuse owner requires an explicit pinned authorization map.
import hashlib
artifact=pathlib.Path(p['artifacts'][0]);hasher=hashlib.sha256()
with artifact.open('rb') as stream:
 for chunk in iter(lambda:stream.read(4*1024*1024),b''):hasher.update(chunk)
assert hasher.hexdigest()==manifest['ranks']['0']['sha256']
(root/'reuse-map.json').write_text(json.dumps({'0:0':{'path':str(artifact),'sha256':hasher.hexdigest(),'bytes':artifact.stat().st_size}},sort_keys=True)+'\n')
sock='/run/media/pc1/Infrastructure/Naos/.r309-old-r1.sock';p['workers'][0]['argv']=['/usr/bin/python3','-B',str(code/'proxy_diag.py'),str(base/'r299-code-r1/proxy.py'),sock,str(root/'proxy.stderr')]
policy=ResourcePolicy(8*1024**3,6*1024**3,5*1024**3,1*1024**3);admission=policy.check(base);calls=[]
def check():
 calls.append(policy.check(base))
 if len(calls)==2:raise AdmissionError('intentional refusal before second session')
fleet=GuardedFleet(check);state={'status':'STARTED','run_id':run,'admission':admission,'gpu':False}
err=(root/'daemon.stderr').open('wb');proc=subprocess.Popen([sys.executable,'-B',str(code/'multiplex.py'),sock,str(receiver),'2'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,bufsize=0)
(root/'daemon-pid.json').write_text(json.dumps({'pid':proc.pid})+'\n')
try:
 if proc.stdout.readline()!=b'READY\n':raise RuntimeError('daemon did not start')
 session=fleet.open(base/'r307-rust-r1/target/debug/examples/r261_multipart_cpu_session',p,root/'ledger0',deadline_seconds=180)
 status=pathlib.Path(f'/proc/{session.proc.pid}/status').read_text().splitlines()
 state['coordinator_vmhwm_kib']=int(next(line.split()[1] for line in status if line.startswith('VmHWM:')))
 state['owner_vmhwm_kib']=int(next(line.split()[1] for line in pathlib.Path(f'/proc/{proc.pid}/status').read_text().splitlines() if line.startswith('VmHWM:')))
 import torch
 from nvfp4_oracle import expert_forward
 from adapters.session_bridge import DurableHookError
 torch.set_num_threads(1);torch.manual_seed(304)
 x=(torch.randn(1,2048)*.1).half();ids=torch.tensor([[5,7]],dtype=torch.int64);weights=torch.tensor([[.625,.375]],dtype=torch.float32)
 output=session(None,x,weights,ids,None,None)
 source=base/'r262-full-layer-r1/experts.safetensors'
 expected=(expert_forward(source,0,5,x[0]).double()*.625+expert_forward(source,0,7,x[0]).double()*.375).half()
 difference=(output[0].float()-expected.float()).abs().max().item()
 assert difference<1e-4 and not torch.cuda.is_initialized()
 try:session(None,x,weights,torch.tensor([[5,5]]),None,None)
 except DurableHookError:pass
 else:raise AssertionError('duplicate route admitted')
 assert not (session.directory/'batch-00000000000000000001.claim.json').exists()
 done=json.loads((session.directory/'batch-00000000000000000000.done.json').read_text())
 payload=(session.directory/'batch-00000000000000000000.fp16').read_bytes()
 assert hashlib.sha256(payload).hexdigest()==done['sha256']
 state.update(expert_batch='PASS',expert_max_abs=difference,batch_sha256=done['sha256'],duplicate_route_refused=True,duplicate_batch_claimed=False)
 q=copy.deepcopy(p);q['routing']['stamp']['layer']=1
 try:fleet.open(base/'r307-rust-r1/target/debug/examples/r261_multipart_cpu_session',q,root/'ledger1',deadline_seconds=180)
 except AdmissionError:pass
 else:raise AssertionError('second session was admitted')
 assert not fleet.cleanup_errors and session.proc.poll()==0 and (session.directory/'closed.json').is_file() and not (root/'ledger1').exists()
 proc.stdin.write(b'S');proc.stdin.flush();line=proc.stdout.readline();rc=proc.wait(timeout=30)
 assert line==b'CLOSED 1\n' and rc==0
 state.update(status='PASS',owner_exit_code=session.proc.returncode,multiplexer_exit_code=rc,admission_checks=len(calls),second_identity_claimed=False,closed_file=True)
except BaseException as exc:
 state.update(status='FAILED',error_type=type(exc).__name__,error=str(exc)[:500],daemon_pid=proc.pid,daemon_poll=proc.poll())
 # Preserve uncertainty: only stop acceptance after the failed coordinator has
 # demonstrably exited; do not kill or replay an uncertain owner.
 process_files=list((root/'ledger0').glob('*/process.json')) if (root/'ledger0').exists() else []
 coordinator_pid=json.loads(process_files[0].read_text())['pid'] if process_files else None
 import time
 def exited(pid):
  if pid is None:return True
  try:return pathlib.Path('/proc/'+str(pid)+'/stat').read_text().split(') ',1)[1].startswith('Z ')
  except FileNotFoundError:return True
 for _ in range(50):
  if exited(coordinator_pid):break
  time.sleep(.1)
 stopped=exited(coordinator_pid)
 state.update(coordinator_pid=coordinator_pid,coordinator_absent=stopped)
 if stopped and proc.poll() is None:
  try:
   proc.stdin.write(b'S');proc.stdin.flush();state['daemon_after_stop']=proc.wait(timeout=30)
  except Exception as cleanup:state['cleanup_error']=type(cleanup).__name__
finally:
 err.close();(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print(json.dumps(state),flush=True)
if state['status']!='PASS':sys.exit(1)
