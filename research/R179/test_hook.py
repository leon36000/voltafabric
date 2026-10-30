import ast,base64,hashlib,json,pathlib,struct,sys,types,unittest,uuid
import torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');REPO=B/'.runs/gitlab-import-r113-20261028/repository';sys.path[:0]=[str(REPO),str(pathlib.Path(__file__).resolve().parent)]
from hook import CpuHook,HookError
from adapters.persistent_gpu_owner import PersistentQPN2Owner
from scripts.resident_artifact import _decode_json

def setup_owners():
 provider_src=(REPO/'research/R156/remote.py').read_bytes();func=next(n for n in ast.parse(provider_src).body if isinstance(n,ast.FunctionDef) and n.name=='unpack_qpn');g={'torch':torch};exec(compile(ast.Module(body=[func],type_ignores=[]),'<cpu-oracle>','exec'),g)
 class Provider:
  def __init__(self):self.device=torch.device('cpu');self.admissions=self.launches=self.synchronized_launches=0
  def gemm_qpn2_moe(self,x,codes,scales,gs,blocks,slots,ntpp,s,n,k,in_div,splitk,nacc):
   self.admissions+=1;assert s==1 and blocks.tolist()==[0] and slots.tolist()==[0,1,1,1,1,1,1,1];out=torch.mv(g['unpack_qpn'](codes,scales,gs,n,k),x[0].to(torch.float64)).to(torch.float16).reshape(1,n);self.launches+=1;return out
  def synchronize(self):self.synchronized_launches=self.launches
  def close(self):pass
 def align(ids,block,e):
  assert ids.tolist()==[[0]] and block==8 and e==1
  return torch.tensor([0,1,1,1,1,1,1,1]),torch.tensor([0]),torch.tensor(8)
 owners={};probs=json.loads((B/'.runs/r158-real-resident-owner-cpu/problem.json').read_bytes()); hashes={2:'8f1563c9a1c9308b2b63ebc58ceef94689393402a878386e2f32bdf3166b2ff7',9:'1d3eafe7c7e37f96a03b5bfa1cbce04619adcb50f061c91648a2643555611c6d'}
 for rank in (2,9):
  path=B/f'.runs/r158-real-resident-owner-cpu/resident-rank{rank}.safetensors'; raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==hashes[rank];n=struct.unpack('<Q',raw[:8])[0];meta=_decode_json(raw[8:8+n]);info=json.loads(meta['__metadata__']['vf_info']);mapping=torch.full((2,),-1,dtype=torch.int64);mapping[info['physical_experts'][0]['id']]=0;owners[rank]=PersistentQPN2Owner.load(artifact_path=str(path),expected_info=info,expected_sha256=hashes[rank],provider=Provider(),expert_map=mapping,device='cpu')
 return owners,probs,align

class EngineHookTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);assert torch.version.cuda is None;cls.owners,cls.problem,cls.align=setup_owners()
 @classmethod
 def tearDownClass(cls):
  for x in cls.owners.values():x.close()
 def test_donor_actual_apply_to_real_owners_and_collector(self):
  cli=B/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';self.assertEqual(hashlib.sha256(cli.read_bytes()).hexdigest(),'11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4');p=self.problem.copy();p['stamp']={**p['stamp'],'run_id':'r179-'+uuid.uuid4().hex,'plan_id':'donor-hook'};hook=CpuHook(problem=p,owners=self.owners,cli=str(cli),align=type(self).align);source=(B/'.runs/r124-complete-overlay/sources/modelopt.py').read_bytes();self.assertEqual(hashlib.sha256(source).hexdigest(),'7d8288560318c37c340c6aff44c7d2c32ff80d7c22bd9b2eb82cb82e6bd1fb8f');tree=ast.parse(source);cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelOptNvFp4FusedMoE');method=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='apply');module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[]);ast.fix_missing_locations(module);env={'_skinny_moe':types.SimpleNamespace(SKINNY_MOE_ENABLED=True,moe_apply=hook)};exec(compile(module,'<actual-pinned-donor-apply>','exec'),env)
  x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);weights=torch.tensor([[0.25,0.5]],dtype=torch.float32);ids=torch.tensor([[0,1]],dtype=torch.int64);result=env['apply'](types.SimpleNamespace(is_monolithic=False),object(),x,weights,ids,None,None);capture=(B/'.runs/r158-real-resident-owner-cpu/remote.stdout').read_bytes();self.assertEqual(hashlib.sha256(capture).hexdigest(),'2667891667f5d05d2b7a9eb28c3f0a8f88c201a6c464a3583038d438687958f6');captured=json.loads(capture);row=struct.unpack('<2048e',base64.b64decode(captured['results'][0]['row'],validate=True));self.assertEqual(result.dtype,torch.float16);self.assertTrue(torch.equal(result,torch.tensor([[0.75*v for v in row]],dtype=torch.float16)))
  with self.assertRaises(HookError):hook(object(),x,weights,ids,None,None)
class FailureTest(unittest.TestCase):
 def test_failure_is_terminal(self):
  class BadOwner:
   def __init__(self):self.calls=0
   def execute(self,*a,**kw):self.calls+=1;raise RuntimeError('synthetic unknown after admission')
  owner=BadOwner();p=json.loads((B/'.runs/r158-real-resident-owner-cpu/problem.json').read_bytes());p['stamp']={**p['stamp'],'run_id':'r179-negative-'+uuid.uuid4().hex,'plan_id':'failure-test'};cli=B/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';hook=CpuHook(problem=p,owners={2:owner},cli=str(cli),align=None);x=torch.zeros((1,2048),dtype=torch.float16);w=torch.tensor([[0.25,0.5]]);ids=torch.tensor([[0,1]])
  with self.assertRaises(RuntimeError):hook(object(),x,w,ids,None,None)
  with self.assertRaises(HookError):hook(object(),x,w,ids,None,None)
  self.assertEqual(owner.calls,1)
 def test_stale_owner_output_fails_atomically(self):
  class Output:
   def __init__(self,stamp,rank,attempt):self.stamp=stamp;self.rank=rank;self.attempt=attempt
   def as_raw_output_bytes(self,identities):return {'stamp':{**self.stamp,'epoch':0},'rank':self.rank,'attempt':self.attempt,'rows':1,'columns':2048,'fp16_le_bytes':bytes(4096)}
  class Owner:
   def __init__(self):self.calls=0
   def execute(self,dispatch,**kwargs):self.calls+=1;return Output(dispatch['stamp'],dispatch['rank'],kwargs['attempt'])
  owners={2:Owner(),9:Owner()};p=json.loads((B/'.runs/r158-real-resident-owner-cpu/problem.json').read_bytes());p['stamp']={**p['stamp'],'run_id':'r179-stale-'+uuid.uuid4().hex,'plan_id':'stale-test'};cli=B/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';hook=CpuHook(problem=p,owners=owners,cli=str(cli),align=None);x=torch.zeros((1,2048),dtype=torch.float16);w=torch.tensor([[0.25,0.5]]);ids=torch.tensor([[0,1]])
  with self.assertRaises(HookError):hook(None,x,w,ids,None,None)
  with self.assertRaises(HookError):hook(None,x,w,ids,None,None)
  self.assertEqual([owners[r].calls for r in (2,9)],[1,1])
 def test_shared_experts_refused_before_dispatch(self):
  hook=CpuHook(problem={},owners={},cli='',align=None)
  with self.assertRaises(HookError):hook(None,None,None,None,object(),None)
  self.assertEqual(hook.attempt,0)
if __name__=='__main__':unittest.main()
