import unittest,torch
from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule
from adapters.qwen_gdn_cpu import recurrent_delta,GdnInputError
class Test(unittest.TestCase):
 def test_hand_calculated_two_steps_and_state(self):
  q=torch.ones(1,2,1,1);k=q.clone();v=torch.full_like(q,2);g=torch.zeros(1,2,1);beta=torch.full_like(g,.5);old=torch.zeros(1,1,1,1);out,state=recurrent_delta(q,k,v,g,beta,old);torch.testing.assert_close(out.flatten(),torch.tensor([1.,1.5]));torch.testing.assert_close(state.flatten(),torch.tensor([1.5]));self.assertEqual(old.item(),0.)
 def test_against_independent_chunk_and_split(self):
  torch.manual_seed(14);q=torch.randn(1,7,2,4)*.4;k=torch.randn_like(q)*.4;v=torch.randn(1,7,2,3);g=-torch.rand(1,7,2)*.4;beta=torch.rand(1,7,2);initial=torch.randn(1,2,4,3)*.1
  out,state=recurrent_delta(q,k,v,g,beta,initial,normalize=True);ref,ref_state=torch_chunk_gated_delta_rule(q,k,v,g,beta,chunk_size=4,initial_state=initial.clone(),output_final_state=True,use_qk_l2norm_in_kernel=True);torch.testing.assert_close(out,ref,rtol=2e-5,atol=2e-6);torch.testing.assert_close(state,ref_state,rtol=2e-5,atol=2e-6)
  a,sa=recurrent_delta(q[:,:3],k[:,:3],v[:,:3],g[:,:3],beta[:,:3],initial,normalize=True);b,sb=recurrent_delta(q[:,3:],k[:,3:],v[:,3:],g[:,3:],beta[:,3:],sa,normalize=True);torch.testing.assert_close(torch.cat((a,b),1),out);torch.testing.assert_close(sb,state)
 def test_limit(self):
  q=torch.zeros(1,4097,1,1);g=torch.zeros(1,4097,1)
  with self.assertRaises(GdnInputError):recurrent_delta(q,q,q,g,g)
 def test_rejections(self):
  q=torch.ones(1,1,1,1);g=torch.zeros(1,1,1);beta=torch.ones_like(g)
  for args in [(q,q,q,g,beta*2,None),(q,q,q,g+1,beta,None),(q,q,q,g,beta,torch.zeros(1,1,2,1)),(q,q,q,g+float('nan'),beta,None)]:
   with self.subTest(args=args),self.assertRaises(GdnInputError):recurrent_delta(*args)
if __name__=='__main__':unittest.main()
