import pathlib,torch,unittest,torch.nn.functional as F,hashlib,json
from safetensors import safe_open
import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
platforms._current_platform=CpuPlatform()
from vllm.config import VllmConfig,set_current_vllm_config
from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule
from adapters.qwen_gdn_attention_cpu import QwenGdnCpuReference
ROOT=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors');P='model.language_model.layers.0.linear_attn.'
class Test(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.ctx=set_current_vllm_config(VllmConfig());cls.ctx.__enter__()
  names=['in_proj_qkv.weight','in_proj_qkv.weight_scale','in_proj_qkv.input_scale','in_proj_z.weight','in_proj_z.weight_scale','in_proj_z.input_scale','out_proj.weight','out_proj.weight_scale','out_proj.input_scale','in_proj_a.weight','in_proj_b.weight','conv1d.weight','A_log','dt_bias','norm.weight']
  with safe_open(str(ROOT),framework='pt',device='cpu') as f:cls.weights={n:f.get_tensor(P+n).contiguous() for n in names}
  cls.model=QwenGdnCpuReference(cls.weights)
 @classmethod
 def tearDownClass(cls):cls.ctx.__exit__(None,None,None)
 def test_real_weights_chunk_oracle_and_segmented_state(self):
  torch.manual_seed(248);x=torch.randn(1,2,2048)*.05;history=torch.zeros(1,8192,3);y,h,state=self.model(x,history,None)
  def fp8(name,inp):
   scale=self.weights[name+'.input_scale'];weight=self.weights[name+'.weight'];ws=self.weights[name+'.weight_scale'];q=(inp.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).float();return (q@weight.float().T)*scale*ws
  qkv=fp8('in_proj_qkv',x.reshape(2,2048)).reshape(1,2,8192);z=fp8('in_proj_z',x.reshape(2,2048)).reshape(1,2,32,128);a=F.linear(x,self.weights['in_proj_a.weight'].float());b=F.linear(x,self.weights['in_proj_b.weight'].float());conv=F.silu(F.conv1d(torch.cat([history,qkv.transpose(1,2)],-1),self.weights['conv1d.weight'].float(),groups=8192)).transpose(1,2);q,k,v=conv.split([2048,2048,4096],-1);q=q.reshape(1,2,16,128).repeat_interleave(2,2);k=k.reshape(1,2,16,128).repeat_interleave(2,2);v=v.reshape(1,2,32,128);g=-self.weights['A_log'].float().exp()*F.softplus(a+self.weights['dt_bias'].float());core,refstate=torch_chunk_gated_delta_rule(q,k,v,g,b.sigmoid(),chunk_size=4,initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=True);norm=core*torch.rsqrt(core.square().mean(-1,keepdim=True)+1e-6)*self.weights['norm.weight'].float()*F.silu(z);ref=fp8('out_proj',norm.reshape(2,4096)).reshape(1,2,2048)
  print('VF_METRICS '+json.dumps({'max_output_error':float((y-ref).abs().max()),'max_state_error':float((state-refstate).abs().max()),'output_sha256':hashlib.sha256(y.contiguous().numpy().tobytes()).hexdigest(),'cuda_initialized':torch.cuda.is_initialized()}),flush=True)
  torch.testing.assert_close(y,ref,rtol=2e-4,atol=2e-5);torch.testing.assert_close(state,refstate,rtol=2e-5,atol=2e-6)
  y1,h1,s1=self.model(x[:,:1],history,None);y2,h2,s2=self.model(x[:,1:],h1,s1);torch.testing.assert_close(torch.cat([y1,y2],1),y,rtol=2e-4,atol=2e-5);torch.testing.assert_close(h2,h);torch.testing.assert_close(s2,state,rtol=2e-5,atol=2e-6)
 def test_bad_input_refused(self):
  with self.assertRaises(ValueError):self.model(torch.zeros(1,1,3),torch.zeros(1,8192,3),None)
if __name__=='__main__':unittest.main()
