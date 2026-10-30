import pathlib,sys,unittest,torch
R=pathlib.Path(__file__).resolve().parent.parent.parent;sys.path.insert(0,str(R));from adapters.onecat_shared_guard import call_after_shared,SharedOrderError
class Shared:
 def __init__(self,x):self.enable_dbo=False;self._output_idx=0;self._output=[x.clone(),None];self.order=1
 def _determine_shared_experts_order(self,x):return self.order
class TestGuard(unittest.TestCase):
 def setUp(self):self.x=torch.ones((2,4),dtype=torch.float16);self.s=Shared(self.x);self.calls=[]
 def call(self,shared=None,x=None):
  return call_after_shared(lambda *args:self.calls.append(args) or args[1],None,self.x,None,None,shared or self.s,self.x if x is None else x,shared_type=Shared,no_overlap=1)
 def test_pass_does_not_consume_or_add_shared(self):
  before=self.s._output[0].clone();self.assertIs(self.call(),self.x);self.assertTrue(torch.equal(self.s._output[0],before));self.assertEqual(self.calls[0][-2:],(None,None))
 def test_wrong_order_refused(self):
  self.s.order=2
  with self.assertRaises(SharedOrderError):self.call()
  self.assertEqual(self.calls,[])
 def test_missing_or_nonfinite_refused(self):
  for out in [None,torch.full((2,4),float('nan'),dtype=torch.float16)]:
   self.s._output[0]=out
   with self.assertRaises(SharedOrderError):self.call()
  self.assertEqual(self.calls,[])
 def test_wrong_input_and_dbo_refused(self):
  with self.assertRaises(SharedOrderError):self.call(x=self.x.clone())
  self.s.enable_dbo=True
  with self.assertRaises(SharedOrderError):self.call()
  self.assertEqual(self.calls,[])
if __name__=='__main__':unittest.main()
