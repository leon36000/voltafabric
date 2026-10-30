import unittest,torch
from transformers.models.qwen3_next.modeling_qwen3_next import torch_recurrent_gated_delta_rule
from explicit_delta import explicit_delta
class Tests(unittest.TestCase):
 def test_three_step_recurrence(self):
  torch.manual_seed(712); state1=state2=None
  for _ in range(3):
   q=torch.randn(1,1,2,8);k=torch.randn(1,1,2,8);v=torch.randn(1,1,2,6);g=-torch.rand(1,1,2);beta=torch.rand(1,1,2)
   a,state1=explicit_delta(q,k,v,g,beta,state1)
   b,state2=torch_recurrent_gated_delta_rule(q,k,v,g,beta,state2,output_final_state=True,use_qk_l2norm_in_kernel=True)
   self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(state1,state2))
  self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()
