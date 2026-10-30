import torch,unittest
from adapters.qwen_gdn_transaction import TransactionalGdnAttention,GdnTransactionError
class Reference:
 def __init__(self):self.fail=False
 def __call__(self,x,h,s):
  if self.fail:raise ValueError('injected')
  h=h.clone();h[:,:,0]+=1
  state=torch.ones(1,32,128,128) if s is None else s+1
  return x*2+(h[0,0,0]-1),h,state
IDENT={'run_id':'r','model_revision':'m','plan_id':'p','layer':0,'epoch':1}
class Test(unittest.TestCase):
 def setUp(self):self.ref=Reference();self.a=TransactionalGdnAttention(self.ref,IDENT);self.x=torch.ones(1,2048,dtype=torch.float16);self.y=torch.zeros_like(self.x)
 def test_commit_abort_and_no_replay(self):
  self.a.begin({**IDENT,'batch':0,'attempt':1});self.a(self.x,self.y);self.assertTrue((self.y==2).all());self.assertEqual(self.a.committed_batch,0);self.a.abort();self.assertEqual(self.a.committed_batch,0)
  with self.assertRaises(GdnTransactionError):self.a.begin({**IDENT,'batch':0,'attempt':1})
  self.a.begin({**IDENT,'batch':0,'attempt':2});self.a(self.x,self.y);self.assertTrue((self.y==2).all());self.a.commit();self.assertEqual(self.a.committed_batch,1)
  self.a.begin({**IDENT,'batch':1,'attempt':1});self.a(self.x,self.y);self.assertTrue((self.y==3).all());self.a.commit()
 def test_wrong_identity_and_missing_output_rejected(self):
  with self.assertRaises(GdnTransactionError):self.a.begin({**IDENT,'plan_id':'other','batch':0,'attempt':1})
  with self.assertRaises(GdnTransactionError):self.a.begin({**IDENT,'epoch':True,'batch':0,'attempt':1})
  self.a.begin({**IDENT,'batch':0,'attempt':1})
  with self.assertRaises(GdnTransactionError):self.a.commit()
  with self.assertRaises(GdnTransactionError):self.a(self.x,self.x)
  self.a.abort()
 def test_error_does_not_change_committed_state(self):
  self.a.begin({**IDENT,'batch':0,'attempt':1});self.ref.fail=True
  with self.assertRaises(ValueError):self.a(self.x,self.y)
  self.assertEqual(self.a.committed_batch,0);self.a.abort();self.assertTrue((self.y==0).all())
 def test_unknown_never_replayed(self):
  self.a.begin({**IDENT,'batch':0,'attempt':1});self.a(self.x,self.y);self.a.mark_unknown()
  with self.assertRaises(GdnTransactionError):self.a.begin({**IDENT,'batch':0,'attempt':2})
  with self.assertRaises(GdnTransactionError):self.a.commit()
 def test_bad_sequence_and_second_forward(self):
  with self.assertRaises(GdnTransactionError):self.a.begin({**IDENT,'batch':1,'attempt':1})
  self.a.begin({**IDENT,'batch':0,'attempt':1});self.a(self.x,self.y)
  with self.assertRaises(GdnTransactionError):self.a(self.x,self.y)
  self.a.abort()
if __name__=='__main__':unittest.main()
