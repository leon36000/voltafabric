import unittest
from adapters.qwen_decode import TransactionalDecoder, DecodeError

class Attention:
 def __init__(self, fail_commit=False): self.batches=[]; self.pending=False; self.unknown=False; self.fail_commit=fail_commit
 def begin(self,stamp):
  if self.pending or self.unknown: raise RuntimeError('not reusable')
  self.pending=True;self.batches.append(stamp['batch'])
 def commit(self):
  if self.fail_commit: raise RuntimeError('commit failure')
  self.pending=False
 def invalidate(self): self.unknown=True;self.pending=False

class Tests(unittest.TestCase):
 def make(self,attentions,forward=None,predict=None):
  ids=[{'run_id':'run','model_revision':'model','plan_id':'plan','layer':i,'epoch':1} for i in range(len(attentions))]
  return TransactionalDecoder(attentions,ids,forward or (lambda token,pos:token+pos),predict or (lambda value: value+10))
 def test_prefill_then_generation_and_no_replay(self):
  a,b=Attention(),Attention();calls=[]
  decoder=self.make([a,b],lambda token,pos:(calls.append((token,pos)) or token+pos))
  self.assertEqual(decoder.generate([1,2,3],2),[15,28]);self.assertEqual(calls,[(1,0),(2,1),(3,2),(15,3)])
  self.assertEqual(a.batches,[0,1,2,3]);self.assertEqual(b.batches,a.batches)
  with self.assertRaises(DecodeError):decoder.generate([1],1)
 def test_partial_commit_fences_all_and_returns_no_tokens(self):
  a,b=Attention(),Attention(fail_commit=True);decoder=self.make([a,b])
  with self.assertRaisesRegex(RuntimeError,'commit failure'):decoder.generate([1],1)
  self.assertTrue(a.unknown and b.unknown)
  with self.assertRaises(DecodeError):decoder.generate([1],1)
 def test_failure_before_commit_fences_all(self):
  a,b=Attention(),Attention()
  def broken(token,pos):raise RuntimeError('forward failure')
  decoder=self.make([a,b],broken)
  with self.assertRaisesRegex(RuntimeError,'forward failure'):decoder.generate([1],1)
  self.assertTrue(a.unknown and b.unknown)
 def test_invalid_request_rejected_before_attention(self):
  a=Attention();decoder=self.make([a])
  with self.assertRaises(DecodeError):decoder.generate([],1)
  self.assertEqual(a.batches,[])
 def test_mixed_run_rejected_before_attention(self):
  a,b=Attention(),Attention()
  ids=[dict(run_id=r,model_revision='m',plan_id='p',epoch=1,layer=i) for i,r in enumerate(('a','b'))]
  with self.assertRaisesRegex(DecodeError,'mixed execution'):TransactionalDecoder([a,b],ids,lambda t,p:t,lambda x:x)
  self.assertEqual(a.batches,[])
 def test_bad_prediction_fences_attention(self):
  a=Attention();decoder=self.make([a],predict=lambda x:float('nan'))
  with self.assertRaisesRegex(DecodeError,'invalid predicted'):decoder.generate([1],1)
  self.assertTrue(a.unknown)
