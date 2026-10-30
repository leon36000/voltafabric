import pathlib,sys,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]))
from adapters.onecat_quant_overlay import ResidentQuantOverlay,ResidentOverlayError
class Fused:
 def __init__(self):self.moe_config='moe'
class Method:
 def __init__(self,moe,session,**kw):self.moe=moe;self.session=session;self.binding=kw;self._admitted=True
 def _check_binding(self):pass
class Delegate:
 def get_name(self):return 'original'
 def get_quant_method(self,layer,prefix):return ('upstream',layer,prefix)
class Tests(unittest.TestCase):
 def setUp(self):
  self.layers=types.ModuleType('vllm.model_executor.layers.fused_moe.layer');self.layers.FusedMoE=Fused
  self.resident=types.ModuleType('adapters.onecat_resident_method');self.resident.ResidentMoEMethod=Method
  self.modules=patch.dict(sys.modules,{'vllm':types.ModuleType('vllm'),'vllm.model_executor':types.ModuleType('vllm.model_executor'),'vllm.model_executor.layers':types.ModuleType('vllm.model_executor.layers'),'vllm.model_executor.layers.fused_moe':types.ModuleType('vllm.model_executor.layers.fused_moe'),'vllm.model_executor.layers.fused_moe.layer':self.layers,'adapters.onecat_resident_method':self.resident});self.modules.start();self.addCleanup(self.modules.stop)
 def test_only_selected_layer_is_intercepted(self):
  session=object();q=ResidentQuantOverlay(Delegate(),{0:session},model_revision='rev',plan_id='plan',epoch=1);f=Fused();name='model.language_model.layers.0.mlp.experts';m=q.get_quant_method(f,name);self.assertIs(m.session,session);self.assertEqual(m.binding['layer_id'],0);self.assertEqual(q.get_name(),'original');self.assertEqual(q.get_quant_method(f,'model.language_model.layers.1.mlp.experts')[0],'upstream');self.assertIs(q.require_complete()[0],m)
 def test_missing_repeated_and_wrong_type_refused(self):
  name='model.language_model.layers.0.mlp.experts';q=ResidentQuantOverlay(Delegate(),{0:object()},model_revision='r',plan_id='p',epoch=1)
  with self.assertRaises(ResidentOverlayError):q.require_complete()
  with self.assertRaises(ResidentOverlayError):q.get_quant_method(object(),name)
  q.get_quant_method(Fused(),name)
  with self.assertRaises(ResidentOverlayError):q.get_quant_method(Fused(),name)
 def test_packed_mapping_updates_delegate(self):
  delegate=Delegate();q=ResidentQuantOverlay(delegate,{0:object()},model_revision='r',plan_id='p',epoch=1)
  q.packed_modules_mapping={'gate_up_proj':['gate_proj','up_proj']}
  self.assertEqual(delegate.packed_modules_mapping,{'gate_up_proj':['gate_proj','up_proj']})
 def test_missing_delegate_refused(self):
  with self.assertRaises(ResidentOverlayError):ResidentQuantOverlay(None,{0:object()},model_revision='r',plan_id='p',epoch=1)
 def test_invalid_layer_binding(self):
  for mapping in ({},{-1:object()},{True:object()}):
   with self.assertRaises(ResidentOverlayError):ResidentQuantOverlay(Delegate(),mapping,model_revision='r',plan_id='p',epoch=1)
if __name__=='__main__':unittest.main()
