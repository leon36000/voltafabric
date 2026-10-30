import unittest
from adapters.qwen_decode import TransactionalDecoder, DecodeError
from research.R340.test_decode import Attention

class DecodeLimits(unittest.TestCase):
 def build(self,fn=lambda x:x+1,**kwargs):
  a=Attention(); ident={'run_id':'r','model_revision':'m','plan_id':'p','layer':0,'epoch':1}
  calls=[]
  def forward(token,pos):calls.append((token,pos));return token
  return a,calls,TransactionalDecoder([a],[ident],forward,fn,**kwargs)
 def test_eos_stops_before_next_model_call(self):
  a,calls,d=self.build(lambda x:7,eos_token_ids={7},context_limit=10)
  self.assertEqual(d.generate([1,2],5),[7]);self.assertEqual(calls,[(1,0),(2,1)]);self.assertEqual(a.batches,[0,1])
 def test_context_boundary_before_any_call(self):
  a,calls,d=self.build(context_limit=3)
  with self.assertRaises(DecodeError):d.generate([1,2],3)
  self.assertEqual(calls,[]);self.assertEqual(a.batches,[])
  self.assertEqual(d.generate([1,2],2),[3,4])
 def test_invalid_configuration(self):
  for kw in ({'context_limit':True},{'context_limit':0},{'eos_token_ids':{True}},{'eos_token_ids':{-1}}):
   with self.assertRaises(DecodeError):self.build(**kw)
