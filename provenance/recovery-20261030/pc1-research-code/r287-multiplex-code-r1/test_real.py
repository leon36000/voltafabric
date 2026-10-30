import copy,json,pathlib,subprocess,sys,torch,uuid,os,hashlib
from adapters.session_bridge import DurableSession, DurableHookError
from nvfp4_oracle import expert_forward
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r287-failure-isolation-r1';root.mkdir();(root/'receiver').mkdir();code=base/'r287-multiplex-code-r1';python=base/'r178-isolated-1cat-cuda128-20261028-r1/venv/bin/python';sock='/run/media/pc1/Infrastructure/Naos/.r287-owner-r1.sock';stderr=(root/'daemon-stderr.txt').open('wb');daemon=subprocess.Popen([str(python),'-B',str(code/'multiplex.py'),sock,str(root/'receiver'),'2'],stdout=subprocess.PIPE,stderr=stderr,bufsize=0);assert daemon.stdout.readline()==b'READY\n';(root/'daemon-pid.json').write_text(json.dumps({'pid':daemon.pid})+'\n')
p=json.loads((base/'r281-layer1-session-r1/input.json').read_text());sessions={};run='r287-'+uuid.uuid4().hex;torch.set_num_threads(1);torch.manual_seed(285);x=(torch.randn(1,2048)*.1).half();ids=torch.tensor([[5,7]],dtype=torch.int64);weights=torch.tensor([[.625,.375]],dtype=torch.float32);p['inputs']=[x[0].tolist()]
try:
 for layer in (0,1):
  q=copy.deepcopy(p);manifest=json.loads((base/'r283-four-layer-owners-r1'/f'layer{layer}'/'manifest.json').read_text());q['artifacts']=[manifest['ranks']['0']['path']];q['workers']=[{'rank':0,'argv':['/usr/bin/python3','-B',str(code/'proxy.py'),sock]}];q['routing']['experts']=list(range(64));q['routing']['registry']={'epoch':1,'copies':[{'expert':i,'rank':0,'local':i} for i in range(64)]};q['routing']['resident_order']={'0':list(range(64))};q['routing']['stamp'].update(run_id=run,plan_id='r287-failure-isolation',layer=layer,batch=0);q['routing']['routes']=[[{'expert':5,'weight':.625},{'expert':7,'weight':.375}]];(root/f'input{layer}.json').write_text(json.dumps(q,indent=2)+'\n');sessions[layer]=DurableSession(base/'r280-rust-x86-r1/target/debug/examples/r261_multipart_cpu_session',q,root/f'ledger{layer}',deadline_seconds=180)
 try:
  sessions[0](None,x,weights,ids,None,None)
  raise AssertionError('failed layer produced output')
 except DurableHookError as e:
  failure=str(e)
 assert (root/'fault-marker.txt').read_text()=='layer 0 failure after admission\n'
 assert sessions[0].terminal
 claim=list(sessions[0].directory.glob('*.claim.json')); assert len(claim)==1
 assert not list(sessions[0].directory.glob('*.done.json')) and not list(sessions[0].directory.glob('*.fp16'))
 try:
  sessions[0](None,x,weights,ids,None,None)
  raise AssertionError('terminal session allowed replay')
 except DurableHookError as e:
  assert 'terminal session; no replay' in str(e)
 results=[]
 source=base/'r280-layer1-r1'/'experts.safetensors'
 for batch in (0,1):
  out=sessions[1](None,x,weights,ids,None,None)
  expect=(expert_forward(source,1,5,x[0]).double()*.625+expert_forward(source,1,7,x[0]).double()*.375).half()
  err=(out[0].float()-expect.float()).abs().max().item();assert err<1e-4,(batch,err)
  results.append({'batch':batch,'layer':1,'max_abs':err})
 assert sessions[1].close()['state']=='closed'
 failed_exit=sessions[0].proc.wait(timeout=30)
 daemon_output=daemon.stdout.read();daemon_exit=daemon.wait(timeout=30);stderr.flush()
 assert failed_exit!=0 and daemon_exit!=0 and b'CLOSED' not in daemon_output
 assert 'synthetic CPU failure after admission' in (root/'daemon-stderr.txt').read_text()
 assert not torch.cuda.is_initialized()
 result={'status':'PASS','run_id':run,'daemon_pid':daemon.pid,'failure':failure,'failed_session_exit':failed_exit,'daemon_exit':daemon_exit,'no_replay':True,'healthy_session_cases':results,'gpu':False}
 (root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
except BaseException as e:
 (root/'failure.json').write_text(json.dumps({'status':'UNKNOWN','run_id':run,'error':str(e),'daemon_pid':daemon.pid})+'\n');raise
finally:stderr.close()
