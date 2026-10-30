"""Two real CPU layers, four owner ranks, fresh identities, bounded resources."""
import copy,hashlib,json,os,pathlib,subprocess,sys,traceback,uuid
os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');code=base/'r312-multilayer-code-r1';root=base/'r312-multilayer-run-r1';root.mkdir(mode=0o700);receiver=root/'receiver';receiver.mkdir();sys.path.insert(0,str(code))
import torch
from nvfp4_oracle import expert_forward
from adapters.resource_admission import ResourcePolicy,AdmissionError
from adapters.guarded_fleet import GuardedFleet
from adapters.session_bridge import DurableHookError
torch.set_num_threads(1);torch.manual_seed(312);x=(torch.randn(1,2048)*.1).half();ids=torch.tensor([[5,70,130,200]],dtype=torch.int64)
policy=ResourcePolicy(8*1024**3,6*1024**3,5*1024**3,1*1024**3);samples=[]
def admit():
 samples.append(policy.check(base))
 if len(samples)==3:raise AdmissionError('intentional third-session refusal')
fleet=GuardedFleet(admit);sessions=fleet.sessions;daemons={};logs=[];run='r312-'+uuid.uuid4().hex;state={'status':'STARTED','run_id':run,'gpu':False};sockets={};approvals={}
try:
 state['preflight']=policy.check(base)
 for layer in (0,1):
  manifest=json.loads((base/'r283-four-layer-owners-r1'/f'layer{layer}'/'manifest.json').read_text())
  for rank in range(4):
   a=manifest['ranks'][str(rank)];path=pathlib.Path(a['path']);h=hashlib.sha256()
   with path.open('rb') as f:
    for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
   if h.hexdigest()!=a['sha256']:raise ValueError('source digest mismatch')
   approvals[f'{layer}:{rank}']={'path':str(path),'sha256':h.hexdigest(),'bytes':path.stat().st_size}
 (root/'reuse-map.json').write_text(json.dumps(approvals,sort_keys=True)+'\n')
 for rank in range(4):
  sock=f'/run/media/pc1/Infrastructure/Naos/.r312-rank{rank}-r1.sock';sockets[rank]=sock;err=(root/f'daemon{rank}.stderr').open('wb');logs.append(err)
  proc=subprocess.Popen([sys.executable,'-B',str(code/'multiplex.py'),sock,str(receiver),'3'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,bufsize=0);daemons[rank]=proc
  if proc.stdout.readline()!=b'READY\n':raise RuntimeError('owner multiplexer unavailable')
 (root/'daemon-pids.json').write_text(json.dumps({r:p.pid for r,p in daemons.items()})+'\n')
 init=json.loads((base/'r281-layer1-session-r1/input.json').read_text());init['inputs']=[x[0].tolist()]
 for layer in (0,1):
  q=copy.deepcopy(init);q['artifacts']=[approvals[f'{layer}:{rank}']['path'] for rank in range(4)];q['workers']=q['workers'][:4]
  for worker in q['workers']:worker['argv']=['/usr/bin/python3','-B',str(base/'r299-code-r1/proxy.py'),sockets[worker['rank']]]
  q['routing']['registry']={'epoch':1,'copies':[{'expert':e,'rank':e//64,'local':e%64} for e in range(256)]};q['routing']['resident_order']={str(i):list(range(64)) for i in range(4)};q['routing']['stamp'].update(run_id=run,plan_id='r312-multilayer',layer=layer,batch=0);q['routing']['routes']=[[{'expert':int(e),'weight':.25} for e in ids[0]]]
  fleet.open(base/'r307-rust-r1/target/debug/examples/r261_multipart_cpu_session',q,root/f'ledger{layer}',deadline_seconds=180)
 results=[]
 for batch,raw in enumerate(([.125,.25,.375,.25],[.5,.125,.125,.25])):
  weights=torch.tensor([raw],dtype=torch.float32)
  for layer in (0,1):
   out=sessions[layer](None,x,weights,ids,None,None)
   source=base/('r262-full-layer-r1' if layer==0 else 'r280-layer1-r1')/'experts.safetensors'
   rows=[expert_forward(source,layer,int(e),x[0]).double()*float(w) for e,w in zip(ids[0],weights[0])]
   expected=sum(rows).half();error=(out[0].float()-expected.float()).abs().max().item()
   if error>=1e-4:raise AssertionError(('oracle mismatch',batch,layer,error))
   done=json.loads((sessions[layer].directory/f'batch-{batch:020d}.done.json').read_text());payload=(sessions[layer].directory/f'batch-{batch:020d}.fp16').read_bytes();assert hashlib.sha256(payload).hexdigest()==done['sha256']
   results.append({'batch':batch,'layer':layer,'max_abs':error,'sha256':done['sha256']})
 state['owner_vmhwm_kib']={rank:int(next(v.split()[1] for v in pathlib.Path(f'/proc/{proc.pid}/status').read_text().splitlines() if v.startswith('VmHWM:'))) for rank,proc in daemons.items()}
 q=copy.deepcopy(init);q['routing']['stamp'].update(run_id=run,plan_id='r312-multilayer',layer=2,batch=0)
 try:fleet.open(base/'r307-rust-r1/target/debug/examples/r261_multipart_cpu_session',q,root/'ledger2',deadline_seconds=180)
 except AdmissionError:pass
 else:raise AssertionError('third session unexpectedly admitted')
 assert not fleet.cleanup_errors and not (root/'ledger2').exists() and all(s.proc.poll()==0 and (s.directory/'closed.json').exists() for s in sessions.values())
 for proc in daemons.values():proc.stdin.write(b'S');proc.stdin.flush()
 for proc in daemons.values():assert proc.wait(timeout=30)==0 and proc.stdout.readline()==b'CLOSED 2\n'
 assert not torch.cuda.is_initialized();state.update(status='PASS',results=results,admission_samples=samples,third_identity_claimed=False,daemon_exit_codes={r:p.returncode for r,p in daemons.items()})
except BaseException as exc:
 state.update(status='FAILED',error_type=type(exc).__name__,error=str(exc)[:500],daemon_pids={r:p.pid for r,p in daemons.items()},daemon_polls={r:p.poll() for r,p in daemons.items()});(root/'traceback.txt').write_text(traceback.format_exc())
finally:
 for f in logs:f.close()
 (root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print(json.dumps(state),flush=True)
if state['status']!='PASS':sys.exit(1)
