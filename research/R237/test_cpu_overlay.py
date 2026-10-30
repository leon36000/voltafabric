import pathlib,sys,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]))
from adapters.onecat_cpu_shared_overlay import CpuSharedOverlay,CpuSharedOverlayError
class Linear:pass
class Method:
 def __init__(self,c):self.config=c
class Delegate:
 w4a16_nvfp4_config='quant'
 def _resolve_quant_algo(self,p):return 'W4A16_NVFP4'
 def get_quant_method(self,l,p):return 'original'
class Tests(unittest.TestCase):
 def setUp(self):
  modules={}
  for name in ('vllm','vllm.model_executor','vllm.model_executor.layers','vllm.model_executor.layers.linear','vllm.platforms','adapters.onecat_cpu_nvfp4'):modules[name]=types.ModuleType(name)
  modules['vllm.model_executor.layers.linear'].LinearBase=Linear;modules['vllm.platforms'].current_platform=types.SimpleNamespace(device_type='cpu');modules['adapters.onecat_cpu_nvfp4'].CpuNvfp4Reference=Method
  self.patch=patch.dict(sys.modules,modules);self.patch.start();self.addCleanup(self.patch.stop)
 def test_targets_and_delegate(self):
  q=CpuSharedOverlay(Delegate(),[0]);p='model.language_model.layers.0.mlp.shared_expert.';self.assertIsInstance(q.get_quant_method(Linear(),p+'gate_up_proj'),Method);self.assertIsInstance(q.get_quant_method(Linear(),p+'down_proj'),Method);self.assertEqual(q.get_quant_method(Linear(),'other'),'original');self.assertEqual(len(q.require_constructed()),2)
 def test_duplicate_wrong_type_and_missing(self):
  q=CpuSharedOverlay(Delegate(),[0]);p='model.language_model.layers.0.mlp.shared_expert.gate_up_proj'
  with self.assertRaises(CpuSharedOverlayError):q.require_constructed()
  with self.assertRaises(CpuSharedOverlayError):q.get_quant_method(object(),p)
  q.get_quant_method(Linear(),p)
  with self.assertRaises(CpuSharedOverlayError):q.get_quant_method(Linear(),p)
 def test_non_cpu_refused(self):
  sys.modules['vllm.platforms'].current_platform.device_type='cuda'
  with self.assertRaises(CpuSharedOverlayError):CpuSharedOverlay(Delegate(),[0])
 def test_wrong_algo_refused(self):
  d=Delegate();d._resolve_quant_algo=lambda _: 'FP8';q=CpuSharedOverlay(d,[0])
  with self.assertRaises(CpuSharedOverlayError):q.get_quant_method(Linear(),'model.language_model.layers.0.mlp.shared_expert.down_proj')
if __name__=='__main__':unittest.main()
