import os,pathlib,sys,json,uuid,hashlib
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');d=b/'r325-correlated-errors-r1';root=d/'live-r1';root.mkdir(mode=0o700);(root/'receiver').mkdir()
sys.path.insert(0,str(d/'code'));sys.path.insert(1,str(b/'r315-cross-host-code-r1'))
import torch
from adapters.session_bridge import DurableSession
from adapters.resource_admission import ResourcePolicy
assert torch.version.cuda is None;torch.set_num_threads(1)
policy=ResourcePolicy(2*1024**3,1024**3,2*1024**3,256*1024**2);snapshot=policy.check(root)
p=json.loads((b/'r281-layer1-session-r1/input.json').read_bytes());m=json.loads((b/'r315-cross-host-remote-r1/reuse-map.json').read_text());artifact=pathlib.Path(m['0:9']['path']);assert hashlib.sha256(artifact.read_bytes()).hexdigest()==m['0:9']['sha256'];(root/'reuse-map.json').write_text(json.dumps(m))
run='r325-'+uuid.uuid4().hex;p['routing']['stamp'].update(run_id=run,plan_id='r325-correlated',model_revision='qwen-r132-eight-copies',layer=0,batch=0);p['routing']['experts']=[4,5,6,7];p['routing']['routes']=[[{'expert':e,'weight':.25} for e in [4,5,6,7]]];p['routing']['registry']={'epoch':p['routing']['stamp']['epoch'],'copies':[{'expert':e,'rank':9,'local':e-4} for e in [4,5,6,7]]};p['routing']['resident_order']={'9':[0,1,2,3]};p['artifacts']=[str(artifact)];p['workers']=[{'rank':9,'argv':[sys.executable,'-B',str(b/'r315-cross-host-code-r1/worker.py'),str(root/'receiver')]}];p['fp16_completions']=True
state={'status':'STARTED','run_id':run,'admission':snapshot,'gpu':False};(root/'receipt.json').write_text(json.dumps(state)+'\n')
try:
 s=DurableSession(d/'target/debug/examples/r261_multipart_cpu_session',p,root/'ledger',deadline_seconds=120,admission=lambda:policy.check(root),correlated=True)
 x=torch.tensor(p['inputs'][:1],dtype=torch.float16);ids=torch.tensor([[4,5,6,7]]);weights=torch.tensor([[.25]*4]);out=s(None,x,weights,ids,None,None);assert out.shape==x.shape and torch.isfinite(out).all();done=json.loads((s.directory/'batch-00000000000000000000.done.json').read_text());closed=s.close();assert len(closed['close'])==1 and closed['close'][0]['exit_code']==0 and closed['close'][0]['bye_execs']==1
 state.update(status='PASS',output_sha256=done['sha256'],close=closed['close'],session_directory=str(s.directory))
except BaseException as e:
 state.update(status='FAILED_OR_UNKNOWN',error_type=type(e).__name__,error=str(e)[:300])
(root/'receipt.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state));sys.exit(0 if state['status']=='PASS' else 1)
