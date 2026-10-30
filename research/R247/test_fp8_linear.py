import pathlib,torch,unittest
from safetensors import safe_open
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform=CpuPlatform()
from vllm.config import VllmConfig,set_current_vllm_config
from adapters.onecat_cpu_fp8 import CpuStaticFp8Linear,Fp8ReferenceError
ROOT=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors');P='model.language_model.layers.0.linear_attn.in_proj_qkv.'
class Test(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.ctx=set_current_vllm_config(VllmConfig());cls.ctx.__enter__()
  with safe_open(str(ROOT),framework='pt',device='cpu') as f:
   cls.w=f.get_tensor(P+'weight')[:4].contiguous();cls.ws=f.get_tensor(P+'weight_scale');cls.xs=f.get_tensor(P+'input_scale')
 @classmethod
 def tearDownClass(cls):cls.ctx.__exit__(None,None,None)
 def test_real_weights_and_independent_scalar_dot(self):
  m=CpuStaticFp8Linear(self.w,self.xs,self.ws);x=torch.linspace(-1,1,2048).reshape(1,-1)*float(self.xs)*3;out=m(x)
  quant=(x.float()/self.xs).clamp(-448,448).to(torch.float8_e4m3fn).float();
  for row in range(4):
   ref=sum(float(quant[0,i])*float(self.w[row,i].float()) for i in range(2048))*float(self.xs)*float(self.ws)
   self.assertAlmostEqual(float(out[0,row]),ref,delta=max(1e-6,abs(ref)*2e-5))
 def test_full_qkv_projection_matches_selected_rows(self):
  with safe_open(str(ROOT),framework='pt',device='cpu') as f:whole=f.get_tensor(P+'weight')
  m=CpuStaticFp8Linear(whole,self.xs,self.ws);small=CpuStaticFp8Linear(self.w,self.xs,self.ws);x=torch.linspace(-1,1,2048).reshape(1,-1)*float(self.xs)*3
  out=m(x);self.assertEqual(tuple(out.shape),(1,8192));torch.testing.assert_close(out[:,:4],small(x),rtol=2e-5,atol=1e-6)
 def test_saturation(self):
  m=CpuStaticFp8Linear(torch.tensor([[1.,-1.]],dtype=torch.float8_e4m3fn),torch.tensor(.5),torch.tensor(.25));out=m(torch.tensor([[1000.,-1000.]]));self.assertEqual(out.item(),112.)
 def test_invalid_scales_shapes_and_nonfinite(self):
  for bad in (torch.tensor(0.),torch.tensor(-1.),torch.tensor(float('nan'))):
   with self.assertRaises(Fp8ReferenceError):CpuStaticFp8Linear(self.w,bad,self.ws)
  m=CpuStaticFp8Linear(self.w,self.xs,self.ws)
  with self.assertRaises(Fp8ReferenceError):m(torch.tensor([[float('nan')]*2048]))
  with self.assertRaises(Fp8ReferenceError):m(torch.zeros(1,1))
if __name__=='__main__':unittest.main()
