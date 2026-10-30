import unittest
import torch
from adapters.qwen_decode import TransactionalDecoder
from adapters.qwen_gdn_transaction import TransactionalGdnAttention, GdnTransactionError
from adapters.qwen_kv_transaction import TransactionalKvAttention, KvTransactionError

class FailGdn(TransactionalGdnAttention):
 def commit(self): raise RuntimeError('forced commit failure')

class Tests(unittest.TestCase):
 def test_real_states_multitoken_prefill(self):
  ids=[dict(run_id='valid',model_revision='m',plan_id='p',epoch=1,layer=i) for i in range(2)]
  def kv_ref(x,old):
   size=1 if old is None else old[0].shape[2]+1
   return x,(torch.zeros((1,2,size,256)),torch.zeros((1,2,size,256)))
  def gdn_ref(x,history,state):return x,history,torch.zeros((1,32,128,128))
  a=TransactionalKvAttention(kv_ref,ids[0]);b=TransactionalGdnAttention(gdn_ref,ids[1]);x=torch.ones((1,2048),dtype=torch.float16)
  def forward(token,position):
   a(x,torch.empty_like(x));b(x,torch.empty_like(x));return token+position
  d=TransactionalDecoder([a,b],ids,forward,lambda value:value+10)
  self.assertEqual(d.generate([1,2,3],2),[15,28])
  self.assertEqual(a.committed_batch,4);self.assertEqual(b.committed_batch,4)
  self.assertFalse(torch.cuda.is_initialized())
 def test_partial_real_commit_fences_both_states(self):
  ids=[dict(run_id='real',model_revision='m',plan_id='p',epoch=1,layer=i) for i in range(2)]
  def kv_ref(x,old):
   size=1 if old is None else old[0].shape[2]+1
   return x, (torch.zeros((1,2,size,256)),torch.zeros((1,2,size,256)))
  def gdn_ref(x,history,state):return x,history,torch.zeros((1,32,128,128))
  a=TransactionalKvAttention(kv_ref,ids[0]);b=FailGdn(gdn_ref,ids[1]);x=torch.ones((1,2048),dtype=torch.float16)
  def forward(token,position):
   a(x,torch.empty_like(x));b(x,torch.empty_like(x));return token
  d=TransactionalDecoder([a,b],ids,forward,lambda value:1)
  with self.assertRaisesRegex(RuntimeError,'forced commit failure'):d.generate([1],1)
  self.assertEqual(a.committed_batch,1)
  with self.assertRaises(KvTransactionError):a.begin({**ids[0],'batch':1,'attempt':1})
  with self.assertRaises(GdnTransactionError):b.begin({**ids[1],'batch':0,'attempt':2})
  self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
