"""CPU-only two-host expert Session using SSH stdio; fresh identity and no replay."""
import base64,hashlib,json,os,pathlib,struct,sys,traceback,uuid
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012'); R=B/'.runs/gitlab-import-r113-20261028/repository';code=B/'.runs/r315-cross-host-code-r1';root=B/'.runs/r315-cross-host-local-r1';sys.path.insert(0,str(R))
import torch
from adapters.session_bridge import DurableSession
from adapters.resource_admission import ResourcePolicy
assert torch.version.cuda is None;torch.set_num_threads(1)
run='r315-'+uuid.uuid4().hex;state={'status':'STARTED','run_id':run,'gpu':False,'transport':'SSH stdio, pc6 + pc1-lan'}
p=json.loads((R/'research/R159/input.json').read_bytes());p['routing']['stamp'].update(run_id=run,plan_id='r315-cross-host',model_revision='qwen-r132-eight-copies');p['routing']['experts']=list(range(8));p['routing']['routes']=[[{'expert':e,'weight':.125} for e in range(8)]];p['routing']['registry']['copies']=[{'expert':e,'rank':2 if e<4 else 9,'local':e%4} for e in range(8)];p['routing']['resident_order']={'2':list(range(4)),'9':list(range(4))};p['artifacts']=[str(R/f'research/R190/artifacts/rank{r}.safetensors') for r in (2,9)]
remote='/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs';remote_code=remote+'/r315-cross-host-code-r1';remote_root=remote+'/r315-cross-host-remote-r1';remote_py=remote+'/r178-isolated-1cat-cuda128-20261028-r1/venv/bin/python';remote_path=':'.join([remote_code,remote+'/r276-package-r1',remote,remote+'/r299-code-r1'])
p['workers']=[{'rank':2,'argv':[str(B/'source/.work/a92-cpu-venv/bin/python'),'-B',str(code/'worker.py'),str(root/'receiver')]},{'rank':9,'argv':['/usr/bin/ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','pc1-lan','env','CUDA_VISIBLE_DEVICES=-1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+remote_path,remote_py,'-B',remote_code+'/worker.py',remote_root+'/receiver']}]
x=torch.tensor(p['inputs'],dtype=torch.float16);ids=torch.tensor([list(range(8))],dtype=torch.int64);weights=torch.full((1,8),.125,dtype=torch.float32);policy=ResourcePolicy(2*1024**3,1*1024**3,2*1024**3,256*1024**2)
try:
 session=DurableSession(B/'.runs/r306-target/debug/examples/r261_multipart_cpu_session',p,root/'ledger',deadline_seconds=180,admission=lambda:policy.check(root));state['session_directory']=str(session.directory)
 expected=torch.tensor([struct.unpack('<2048e',base64.b64decode(json.loads((B/'.runs/r158-real-resident-owner-cpu/remote.stdout').read_bytes())['results'][0]['row'],validate=True))],dtype=torch.float16)
 batches=[]
 for index in range(2):
  output=session(None,x,weights,ids,None,None)
  if not torch.equal(output,expected):raise AssertionError('independent captured expert reference mismatch')
  done=json.loads((session.directory/f'batch-{index:020d}.done.json').read_text());raw=(session.directory/f'batch-{index:020d}.fp16').read_bytes();assert hashlib.sha256(raw).hexdigest()==done['sha256'];batches.append({'batch':index,'sha256':done['sha256']})
 closed=session.close();assert len(closed['close'])==2 and all(c['exit_code']==0 and c['bye_execs']==2 for c in closed['close'])
 state.update(status='PASS',batches=batches,close=closed['close'],reference='R158 FP16 row',identities='8 routes with 4 experts per rank')
except BaseException as exc:
 state.update(status='FAILED_OR_UNKNOWN',error_type=type(exc).__name__,error=str(exc)[:500]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');print(json.dumps(state),flush=True)
if state['status']!='PASS':sys.exit(1)
