import unittest
import torch
from adapters.qwen_decode import TransactionalDecoder,DecodeError
from adapters.qwen_kv_transaction import TransactionalKvAttention
from adapters.qwen_gdn_transaction import TransactionalGdnAttention

class RealLimits(unittest.TestCase):
 def test_eos_and_context_with_real_states(self):
  ids=[dict(run_id='limits',model_revision='m',plan_id='p',epoch=1,layer=i) for i in range(2)]
  def kv(x,old):
   n=1 if old is None else old[0].shape[2]+1
   return x,(torch.zeros((1,2,n,256)),torch.zeros((1,2,n,256)))
  def gdn(x,h,s):return x,h,torch.zeros((1,32,128,128))
  a=TransactionalKvAttention(kv,ids[0]);b=TransactionalGdnAttention(gdn,ids[1]);x=torch.ones((1,2048),dtype=torch.float16);calls=[]
  def forward(token,pos):
   calls.append(pos);a(x,torch.empty_like(x));b(x,torch.empty_like(x));return token
  decoder=TransactionalDecoder([a,b],ids,forward,lambda _:9,context_limit=3,eos_token_ids=(9,))
  with self.assertRaises(DecodeError):decoder.generate([1,2],3)
  self.assertEqual(decoder.generate([1,2],2),[9]);self.assertEqual(calls,[0,1]);self.assertEqual(a.committed_batch,2);self.assertEqual(b.committed_batch,2);self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
