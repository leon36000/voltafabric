import ast,base64,hashlib,json,pathlib,struct,sys,types,unittest,subprocess
from unittest.mock import patch
import torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from session_hook import SessionHook,SessionHookError
class TestSessionDonor(unittest.TestCase):
 def test_real_method_session_and_owners(self):
  torch.set_num_threads(1);self.assertIsNone(torch.version.cuda)
  cli=B/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';self.assertEqual(hashlib.sha256(cli.read_bytes()).hexdigest(),'11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4')
  template=json.loads((R/'research/R159/input.json').read_bytes());hook=SessionHook(template=template,cli=cli,worker=P/'worker.py',python=B/'source/.work/a92-cpu-venv/bin/python',operation_dir=P/'operations')
  src=(B/'.runs/r124-complete-overlay/sources/modelopt.py').read_bytes();self.assertEqual(hashlib.sha256(src).hexdigest(),'7d8288560318c37c340c6aff44c7d2c32ff80d7c22bd9b2eb82cb82e6bd1fb8f')
  cls=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='ModelOptNvFp4FusedMoE');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply');mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[]);ast.fix_missing_locations(mod);env={'_skinny_moe':types.SimpleNamespace(SKINNY_MOE_ENABLED=True,moe_apply=hook)};exec(compile(mod,'<donor>','exec'),env)
  x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);w=torch.tensor([[0.25,0.5]]);ids=torch.tensor([[0,1]])
  out=env['apply'](types.SimpleNamespace(is_monolithic=False),object(),x,w,ids,None,None)
  capture=(B/'.runs/r158-real-resident-owner-cpu/remote.stdout').read_bytes();self.assertEqual(hashlib.sha256(capture).hexdigest(),'2667891667f5d05d2b7a9eb28c3f0a8f88c201a6c464a3583038d438687958f6');row=struct.unpack('<2048e',base64.b64decode(json.loads(capture)['results'][0]['row'],validate=True));self.assertTrue(torch.equal(out,torch.tensor([[0.75*v for v in row]],dtype=torch.float16)))
  with self.assertRaises(SessionHookError):hook(None,x,w,ids,None,None)
 def test_refuses_shared_before_session(self):
  hook=SessionHook(template={},cli='',worker='',python='',operation_dir=P/'operations')
  with self.assertRaises(SessionHookError):hook(None,None,None,None,object(),None)
  self.assertFalse(hook.terminal)
 def test_timeout_is_unknown_and_no_replay(self):
  template=json.loads((R/'research/R159/input.json').read_bytes());hook=SessionHook(template=template,cli='unused',worker='',python='',operation_dir=P/'operations');x=torch.zeros((1,2048),dtype=torch.float16);w=torch.tensor([[0.25,0.5]]);ids=torch.tensor([[0,1]])
  before={p for p in (P/'operations').glob('r180-*.json') if not p.name.endswith('.output.json')}
  with patch('session_hook.subprocess.run',side_effect=subprocess.TimeoutExpired('synthetic',60)) as run:
   with self.assertRaises(SessionHookError):hook(None,x,w,ids,None,None)
   with self.assertRaises(SessionHookError):hook(None,x,w,ids,None,None)
   self.assertEqual(run.call_count,1)
  new={p for p in (P/'operations').glob('r180-*.json') if not p.name.endswith('.output.json')}-before;self.assertEqual(len(new),1);receipt=json.loads(new.pop().read_bytes());self.assertEqual(receipt['status'],'RUNNING');self.assertFalse(receipt['replay_allowed'])
if __name__=='__main__':unittest.main()
