import hashlib,json,pathlib,torch,torch.nn.functional as F,unittest
from safetensors import safe_open
from transformers.models.qwen3_next.modeling_qwen3_next import torch_causal_conv1d_update,torch_chunk_gated_delta_rule
from adapters.qwen_gdn_projected_cpu import projected_core,ProjectedInputError
ROOT=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r188-qwen-checkpoint-pc1-20261028-r1/model');PREFIX='model.language_model.layers.0.linear_attn.'
class Test(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
  with safe_open(str(ROOT/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:
   cls.w=f.get_tensor(PREFIX+'conv1d.weight').squeeze(1).float();cls.a=f.get_tensor(PREFIX+'A_log').float();cls.dt=f.get_tensor(PREFIX+'dt_bias').float()
 def test_real_conv_gate_parameters_against_donor(self):
  torch.manual_seed(245);qkv=torch.randn(1,3,8192)*.02;a=torch.randn(1,3,32)*.1;b=torch.randn(1,3,32)*.1;history=torch.randn(1,8192,3)*.01;history_before=history.clone()
  out,h,state=projected_core(qkv,a,b,self.w,self.a,self.dt,history,None,key_heads=16,value_heads=32,key_dim=128,value_dim=128)
  ch=history.clone();parts=[]
  for i in range(3):parts.append(torch_causal_conv1d_update(qkv[:,i:i+1].transpose(1,2),ch,self.w,None,'silu').transpose(1,2))
  conv=torch.cat(parts,dim=1);q,k,v=conv.split([2048,2048,4096],dim=-1);q=q.reshape(1,3,16,128).repeat_interleave(2,2);k=k.reshape(1,3,16,128).repeat_interleave(2,2);v=v.reshape(1,3,32,128);g=-self.a.exp()*F.softplus(a+self.dt);beta=b.sigmoid();ref,rs=torch_chunk_gated_delta_rule(q,k,v,g,beta,chunk_size=4,initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=True)
  torch.testing.assert_close(out,ref,rtol=2e-5,atol=2e-6);torch.testing.assert_close(state,rs,rtol=2e-5,atol=2e-6);torch.testing.assert_close(h,ch);torch.testing.assert_close(history,history_before)
  first,h1,s1=projected_core(qkv[:,:1],a[:,:1],b[:,:1],self.w,self.a,self.dt,history,None,key_heads=16,value_heads=32,key_dim=128,value_dim=128);rest,h2,s2=projected_core(qkv[:,1:],a[:,1:],b[:,1:],self.w,self.a,self.dt,h1,s1,key_heads=16,value_heads=32,key_dim=128,value_dim=128);torch.testing.assert_close(torch.cat([first,rest],1),out);torch.testing.assert_close(h2,h);torch.testing.assert_close(s2,state)
 def test_bad_shape_and_nan_refused(self):
  q=torch.zeros(1,1,8192);a=torch.zeros(1,1,32);history=torch.zeros(1,8192,3)
  with self.assertRaises(ProjectedInputError):projected_core(q[:,:,:-1],a,a,self.w,self.a,self.dt,history,None,key_heads=16,value_heads=32,key_dim=128,value_dim=128)
  a[0,0,0]=float('nan')
  with self.assertRaises(ProjectedInputError):projected_core(q,a,a,self.w,self.a,self.dt,history,None,key_heads=16,value_heads=32,key_dim=128,value_dim=128)
if __name__=='__main__':unittest.main()
