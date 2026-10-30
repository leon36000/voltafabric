import copy,json,os,pathlib,subprocess,sys,uuid
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs')
code=base/'r303-controlled-fleet-code-r1'; root=base/'r303-controlled-fleet-run-r1';root.mkdir(mode=0o700);receiver=root/'receiver';receiver.mkdir()
sys.path.insert(0,str(code));sys.path.insert(1,str(base/'r276-package-r1'));sys.path.insert(2,str(base));sys.path.insert(3,str(base/'r299-code-r1'))
from adapters.guarded_fleet import GuardedFleet
from adapters.resource_admission import AdmissionError,ResourcePolicy
p=json.loads((base/'r281-layer1-session-r1/input.json').read_text()); manifest=json.loads((base/'r283-four-layer-owners-r1/layer0/manifest.json').read_text());run='r303-'+uuid.uuid4().hex
p['artifacts']=[manifest['ranks']['0']['path']];p['workers']=p['workers'][:1];p['routing']['experts']=list(range(64));p['routing']['registry']={'epoch':1,'copies':[{'expert':i,'rank':0,'local':i} for i in range(64)]};p['routing']['resident_order']={'0':list(range(64))};p['routing']['stamp'].update(run_id=run,plan_id='r303-controlled-fleet',layer=0,batch=0);p['routing']['routes']=[[{'expert':5,'weight':.625},{'expert':7,'weight':.375}]]
sock='/run/media/pc1/Infrastructure/Naos/.r303-controlled-fleet-r1.sock';p['workers'][0]['argv']=['/usr/bin/python3','-B',str(base/'r299-code-r1/proxy.py'),sock]
policy=ResourcePolicy(8*1024**3,6*1024**3,5*1024**3,1*1024**3);admission=policy.check(base);calls=[]
def check():
 calls.append(policy.check(base))
 if len(calls)==2:raise AdmissionError('intentional refusal before second session')
fleet=GuardedFleet(check);state={'status':'STARTED','run_id':run,'admission':admission,'gpu':False}
err=(root/'daemon.stderr').open('wb');proc=subprocess.Popen([sys.executable,'-B',str(code/'multiplex.py'),sock,str(receiver),'2'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,bufsize=0)
(root/'daemon-pid.json').write_text(json.dumps({'pid':proc.pid})+'\n')
try:
 if proc.stdout.readline()!=b'READY\n':raise RuntimeError('daemon did not start')
 session=fleet.open(base/'r296-rust-r1/target/debug/examples/r261_multipart_cpu_session',p,root/'ledger0',deadline_seconds=180)
 q=copy.deepcopy(p);q['routing']['stamp']['layer']=1
 try:fleet.open(base/'r296-rust-r1/target/debug/examples/r261_multipart_cpu_session',q,root/'ledger1',deadline_seconds=180)
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
 for _ in range(50):
  if coordinator_pid is None or not pathlib.Path('/proc/'+str(coordinator_pid)).exists(): break
  time.sleep(.1)
 stopped=coordinator_pid is None or not pathlib.Path('/proc/'+str(coordinator_pid)).exists()
 state.update(coordinator_pid=coordinator_pid,coordinator_absent=stopped)
 if stopped and proc.poll() is None:
  try:
   proc.stdin.write(b'S');proc.stdin.flush();state['daemon_after_stop']=proc.wait(timeout=30)
  except Exception as cleanup:state['cleanup_error']=type(cleanup).__name__
finally:
 err.close();(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print(json.dumps(state),flush=True)
if state['status']!='PASS':sys.exit(1)
