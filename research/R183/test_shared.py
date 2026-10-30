import ast,base64,enum,hashlib,json,pathlib,struct,sys,types,unittest,uuid
import torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';P=pathlib.Path(__file__).parent;sys.path.insert(0,str(P));from shared_hook import SharedAwareSession,DurableHookError
BIN=B/'.runs/r181-target/debug/examples/r181_persistent_cpu_session';PY=B/'source/.work/a92-cpu-venv/bin/python'
def donor_shared():
 src=(P/'shared_experts_pinned.py').read_bytes();assert hashlib.sha256(src).hexdigest()=='e7504952e81b45ee69e4be5edce5f73b06186b6b309e3acd9c27bb6fe8109ee4';tree=ast.parse(src);order=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SharedExpertsOrder');cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SharedExperts');apply=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply');mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),order,apply],type_ignores=[]);ast.fix_missing_locations(mod);env={'IntEnum':enum.IntEnum};exec(compile(mod,'<pinned-1cat-shared>','exec'),env);return env['SharedExpertsOrder'],env['apply']
class TestShared(unittest.TestCase):
 def test_actual_shared_apply_and_real_persistent_owners(self):
  torch.set_num_threads(1);order,apply=donor_shared();p=json.loads((R/'research/R159/input.json').read_bytes());run='r181-'+uuid.uuid4().hex;p['routing']['stamp']['run_id']=run;p['routing']['stamp']['plan_id']='shared-order-cpu';p['workers']=[{'rank':r,'argv':[str(PY),'-B',str(R/'research/R181/worker.py'),run]} for r in (2,9)];hook=SharedAwareSession(BIN,p,P/'ledger',deadline_seconds=15,internal_order=order.MK_INTERNAL_OVERLAPPED)
  calls=[]
  class Shared:
   _output_idx=0
   def __init__(self):self._output=[None,None]
   def _determine_shared_experts_order(self,x):return order.MK_INTERNAL_OVERLAPPED
   def _layer(self,x):calls.append(1);return x*2
   def apply(self,x,o):return apply(self,x,o)
  shared=Shared();x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);ids=torch.tensor([[0,1]]);w=torch.tensor([[.25,.5]])
  outs=[]
  for _ in range(2):
   outs.append(hook(None,x,w,ids,shared,x));self.assertTrue(torch.equal(shared._output[0],x*2));shared._output[0]=None
  self.assertEqual(len(calls),2);closed=hook.close();self.assertEqual([c['bye_execs'] for c in closed['close']],[2,2]);row=struct.unpack('<2048e',base64.b64decode(json.loads((B/'.runs/r158-real-resident-owner-cpu/remote.stdout').read_bytes())['results'][0]['row'],validate=True));self.assertTrue(all(torch.equal(out,torch.tensor([[.75*v for v in row]],dtype=torch.float16)) for out in outs));print(json.dumps({'run':run,'ledger':str(hook.directory)}))
 def test_other_orders_do_not_double_run(self):
  order,apply=donor_shared()
  class S:
   _output_idx=0
   def __init__(self,current):self.current=current;self._output=[None,None];self.calls=0
   def _determine_shared_experts_order(self,x):return self.current
   def _layer(self,x):self.calls+=1;return x
  for current in (order.NO_OVERLAP,order.MULTI_STREAM_OVERLAPPED):
   shared=S(current);apply(shared,torch.ones(1),order.MK_INTERNAL_OVERLAPPED);self.assertEqual(shared.calls,0);self.assertIsNone(shared._output[0])
if __name__=='__main__':unittest.main()
