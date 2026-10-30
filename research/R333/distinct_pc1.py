"""Correlated CPU session over two local owners, eight distinct checkpoint experts."""
import os,pathlib,sys,json,uuid,hashlib,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');d=base/'r333-distinct-correlated-r1';root=d/'run';root.mkdir(mode=0o700);(root/'receiver').mkdir();sys.path[:0]=[str(base/'r325-correlated-errors-r1/code'),str(base/'r315-cross-host-code-r1'),str(base/'r276-package-r1'),str(base),str(base/'r299-code-r1')]
import torch
from adapters.session_bridge import DurableSession
from adapters.resource_admission import ResourcePolicy
from nvfp4_oracle import expert_forward
assert torch.version.cuda is None;torch.set_num_threads(1)
policy=ResourcePolicy(2*1024**3,1024**3,2*1024**3,256*1024**2);pre=policy.check(root)
source=base/'r204-real-experts-20261028/selected-r207.safetensors';assert hashlib.sha256(source.read_bytes()).hexdigest()=='0e597ecf47a568d6066ac2490bb09b37f615a1f725f9bdac9364cab4c195928b'
expected_hash={'2':'cd947531488c90034c3b9ba8213673cbbf628d59354437dd49d1ceba1aac87c1','9':'4d8eaf68cda68c998c7c6f7bfc98e8c22ba1aa7bbea6b271cdea33a22db67482'};approval={}
for rank in (2,9):
 f=d/f'rank{rank}.safetensors';assert hashlib.sha256(f.read_bytes()).hexdigest()==expected_hash[str(rank)];approval[f'0:{rank}']={'path':str(f),'sha256':expected_hash[str(rank)],'bytes':f.stat().st_size}
(root/'reuse-map.json').write_text(json.dumps(approval,sort_keys=True)+'\n');ids=[198,204,109,121,171,76,166,173];p=json.loads((base/'r281-layer1-session-r1/input.json').read_text());run='r333-'+uuid.uuid4().hex;p['routing']['stamp'].update(run_id=run,model_revision='1355db6a052410cfd62085d94b58866fd0f2c3c5',plan_id='r333-distinct-correlated',layer=0,batch=0);p['routing']['experts']=ids;p['routing']['routes']=[[{'expert':e,'weight':.125} for e in ids]];p['routing']['registry']={'epoch':1,'copies':[{'expert':e,'rank':2 if i<4 else 9,'local':i%4} for i,e in enumerate(ids)]};p['routing']['resident_order']={'2':list(range(4)),'9':list(range(4))};p['artifacts']=[str(d/f'rank{rank}.safetensors') for rank in (2,9)];worker=base/'r315-cross-host-code-r1/worker.py';p['workers']=[{'rank':rank,'argv':[sys.executable,'-B',str(worker),str(root/'receiver')]} for rank in (2,9)];p['fp16_completions']=True
state={'status':'STARTED','run_id':run,'gpu':False,'admission':pre};(root/'receipt.json').write_text(json.dumps(state)+'\n')
try:
 session=DurableSession(base/'r325-correlated-errors-r1/target/debug/examples/r261_multipart_cpu_session',p,root/'ledger',deadline_seconds=180,admission=lambda:policy.check(root),correlated=True)
 x=torch.tensor(p['inputs'][:1],dtype=torch.float16);cases=[(ids,[.125]*8),(list(reversed(ids)),[.2,.17,.14,.12,.11,.1,.09,.07])];results=[]
 for batch,(order,coefficients) in enumerate(cases):
  weights=torch.tensor([coefficients],dtype=torch.float32);routes=torch.tensor([order],dtype=torch.int64);output=session(None,x,weights,routes,None,None)
  expected=torch.zeros(x.shape[1],dtype=torch.float64)
  for expert,weight in zip(order,weights[0].tolist()):expected+=expert_forward(source,0,expert,x[0]).double()*weight
  err=float((output[0].double()-expected.half().double()).abs().max());assert err<0.0001,err
  done=json.loads((session.directory/f'batch-{batch:020d}.done.json').read_text());raw=(session.directory/f'batch-{batch:020d}.fp16').read_bytes();assert hashlib.sha256(raw).hexdigest()==done['sha256'];results.append({'batch':batch,'max_abs':err,'sha256':done['sha256'],'order':order})
 closed=session.close();assert len(closed['close'])==2 and all(c['exit_code']==0 and c['bye_execs']==2 for c in closed['close']);state.update(status='PASS',results=results,close=closed['close'],session_directory=str(session.directory))
except BaseException as e:
 state.update(status='FAILED_OR_UNKNOWN',error_type=type(e).__name__,error=str(e)[:300]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state));sys.exit(0 if state['status']=='PASS' else 1)
