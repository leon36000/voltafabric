import ast,hashlib,pathlib,sys,types,unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
root=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs'); src=(root/'r145-wheel-imports/gdn.py').read_bytes(); assert hashlib.sha256(src).hexdigest()=='9bcf5ca97363d8e08a669105fe27f163c51e443fc686580ac4842e6c84d3e7f3'
t=ast.parse(src); wanted={'_get_gdn_head_k_dim','_resolve_gdn_prefill_backend','_sm70_flashqla_original_prefill_enabled'}; nodes=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name in wanted]; assert {x.name for x in nodes}==wanted
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);ast.fix_missing_locations(module)
class Platform:
 def __init__(self,cuda=True,major=7,minor=0):self.cuda=cuda;self.major=major;self.minor=minor
 def is_cuda(self):return self.cuda
 def is_device_capability(self,n):return self.major*10+self.minor==n
 def is_device_capability_family(self,n):return self.major*10==n
 def get_device_capability(self):return NS(major=self.major,minor=self.minor)
torch=NS(float16=object(),bfloat16=object());state={'platform':Platform()}
g={'torch':torch,'current_platform':state['platform'],'logger':NS(warning_once=lambda *a,**k:None),'os':__import__('os'),'_is_libs_cu13_install_intact':lambda:False};exec(compile(module,'<wheel-source-ast>','exec'),g)
def conf(dtype,head=128,backend='auto'):
 return NS(additional_config={'gdn_prefill_backend':backend},model_config=NS(hf_text_config=NS(linear_key_head_dim=head),hf_config=None,dtype=dtype))
def modules():
 names=['flash_qla','flash_qla.ops','flash_qla.ops.gated_delta_rule','flash_qla.ops.gated_delta_rule.chunk','flash_qla.ops.gated_delta_rule.chunk.sm70'];d={n:types.ModuleType(n) for n in names};d[names[-1]].chunk_gated_delta_rule_fwd_sm70_vlk_varlen=object();return d
class TestWheelSelector(unittest.TestCase):
 def setUp(self):g['current_platform']=Platform()
 def test_fp16_sm70_with_available_module(self):
  with patch.dict(sys.modules,modules()):self.assertEqual(g['_resolve_gdn_prefill_backend'](conf(torch.float16)),('auto','flashqla_sm70'))
 def test_bf16_sm70_falls_back(self):
  with patch.dict(sys.modules,modules()):self.assertEqual(g['_resolve_gdn_prefill_backend'](conf(torch.bfloat16)),('auto','triton'))
 def test_missing_module_falls_back(self):
  with patch.dict(sys.modules,{'flash_qla':None}):self.assertEqual(g['_resolve_gdn_prefill_backend'](conf(torch.float16)),('auto','triton'))
 def test_incompatible_head_falls_back(self):
  with patch.dict(sys.modules,modules()):self.assertEqual(g['_resolve_gdn_prefill_backend'](conf(torch.float16,64)),('auto','triton'))
 def test_cpu_falls_back(self):
  g['current_platform']=Platform(cuda=False)
  with patch.dict(sys.modules,modules()):self.assertEqual(g['_resolve_gdn_prefill_backend'](conf(torch.float16)),('auto','triton'))
 def test_disabled_original_env(self):
  with patch.dict('os.environ',{'VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL':'0'},clear=True):self.assertFalse(g['_sm70_flashqla_original_prefill_enabled']())
 def test_original_default(self):
  with patch.dict('os.environ',{},clear=True):self.assertTrue(g['_sm70_flashqla_original_prefill_enabled']())
if __name__=='__main__':unittest.main(verbosity=2)
