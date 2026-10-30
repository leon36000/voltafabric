import ast,base64,hashlib,json,pathlib,struct,sys,time,types,unittest,uuid
import torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';P=pathlib.Path(__file__).parent;sys.path.insert(0,str(P))
from durable_hook import DurableSession,DurableHookError
BIN=B/'.runs/r181-target/debug/examples/r181_persistent_cpu_session';PY=B/'source/.work/a92-cpu-venv/bin/python'
def init():
 p=json.loads((R/'research/R159/input.json').read_bytes());run='r181-'+uuid.uuid4().hex;p['routing']['stamp']['run_id']=run;p['routing']['stamp']['plan_id']='r182-durable-cpu';p['workers']=[{'rank':r,'argv':[str(PY),'-B',str(R/'research/R181/worker.py'),run]} for r in (2,9)];return p
class TestDurable(unittest.TestCase):
 def test_real_session_and_reopen_refusal(self):
  torch.set_num_threads(1);self.assertIsNone(torch.version.cuda);p=init();hook=DurableSession(BIN,p,P/'ledger',deadline_seconds=15)
  source=(B/'.runs/r124-complete-overlay/sources/modelopt.py').read_bytes();self.assertEqual(hashlib.sha256(source).hexdigest(),'7d8288560318c37c340c6aff44c7d2c32ff80d7c22bd9b2eb82cb82e6bd1fb8f');cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='ModelOptNvFp4FusedMoE');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply');mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[]);ast.fix_missing_locations(mod);env={'_skinny_moe':types.SimpleNamespace(SKINNY_MOE_ENABLED=True,moe_apply=hook)};exec(compile(mod,'<donor>','exec'),env)
  x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);ids=torch.tensor([[0,1]]);obj=types.SimpleNamespace(is_monolithic=False);outs=[env['apply'](obj,object(),x,torch.tensor([w]),ids,None,None) for w in ([.25,.5],[.125,.25])];closed=hook.close();self.assertEqual([c['bye_execs'] for c in closed['close']],[2,2]);row=struct.unpack('<2048e',base64.b64decode(json.loads((B/'.runs/r158-real-resident-owner-cpu/remote.stdout').read_bytes())['results'][0]['row'],validate=True))
  for i,(out,scale) in enumerate(zip(outs,[.75,.375])):
   self.assertTrue(torch.equal(out,torch.tensor([[scale*v for v in row]],dtype=torch.float16)));prefix=hook.directory/('batch-%020d'%i);claim=json.loads(prefix.with_suffix('.claim.json').read_bytes());done=json.loads(prefix.with_suffix('.done.json').read_bytes());payload=prefix.with_suffix('.fp16').read_bytes();self.assertEqual(done['sha256'],hashlib.sha256(payload).hexdigest());self.assertEqual(claim['attempt'],'assigned-by-rust');self.assertIsInstance(done['attempt'],int)
  with self.assertRaises(FileExistsError):DurableSession(BIN,p,P/'ledger')
  print(json.dumps({'real_run':p['routing']['stamp']['run_id'],'ledger':str(hook.directory)}))
 def test_timeout_leaves_claim_unknown_and_refuses_replay(self):
  p=init();hook=DurableSession(P/'fake_timeout.py',p,P/'ledger',deadline_seconds=.2);x=torch.zeros((1,2048),dtype=torch.float16);w=torch.tensor([[.25,.5]]);ids=torch.tensor([[0,1]])
  with self.assertRaises(DurableHookError):hook(None,x,w,ids,None,None)
  self.assertTrue((hook.directory/'batch-00000000000000000000.claim.json').exists());self.assertFalse((hook.directory/'batch-00000000000000000000.done.json').exists())
  with self.assertRaises(DurableHookError):hook(None,x,w,ids,None,None)
  with self.assertRaises(FileExistsError):DurableSession(P/'fake_timeout.py',p,P/'ledger')
  self.assertEqual(hook.proc.wait(timeout=3),0);hook.proc.stdin.close();hook.proc.stdout.close();print(json.dumps({'synthetic_timeout':p['routing']['stamp']['run_id'],'ledger':str(hook.directory)}))
if __name__=='__main__':unittest.main()
