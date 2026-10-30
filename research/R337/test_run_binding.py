import pathlib, sys, types, unittest
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from adapters.onecat_quant_overlay import ResidentQuantOverlay, ResidentOverlayError

class Layer:
    moe_config = None
class Delegate:
    def get_quant_method(self, layer, prefix): return None
class Session:
    def __init__(self, run): self.identity = {'run_id': run}
class Method:
    def __init__(self, moe, session, **kwargs):
        self.session, self.binding, self._admitted = session, kwargs, True
    def _check_binding(self):
        if self.session.identity['run_id'] != self.binding['run_id']:
            raise RuntimeError('run identity mismatch')
class Tests(unittest.TestCase):
    def test_mixed_run_rejected(self):
        layer_module = types.ModuleType('vllm.model_executor.layers.fused_moe.layer'); layer_module.FusedMoE = Layer
        resident = types.ModuleType('adapters.onecat_resident_method'); resident.ResidentMoEMethod = Method
        mods={name:types.ModuleType(name) for name in ('vllm','vllm.model_executor','vllm.model_executor.layers','vllm.model_executor.layers.fused_moe')}
        mods.update({'vllm.model_executor.layers.fused_moe.layer':layer_module,'adapters.onecat_resident_method':resident})
        with patch.dict(sys.modules, mods):
            overlay=ResidentQuantOverlay(Delegate(), {0:Session('run-a'), 1:Session('run-b')}, model_revision='model',plan_id='plan',epoch=1,run_id='run-a')
            for i in range(2): overlay.get_quant_method(Layer(),f'model.language_model.layers.{i}.mlp.experts')
            with self.assertRaisesRegex(RuntimeError,'run identity mismatch'): overlay.require_complete()
    def test_empty_explicit_run_rejected(self):
        with self.assertRaises(ResidentOverlayError):
            ResidentQuantOverlay(Delegate(),{0:Session('run')},model_revision='m',plan_id='p',epoch=1,run_id='')
class RealBindingTests(unittest.TestCase):
    def test_real_method_rejects_wrong_or_changed_run(self):
        import importlib
        names=('vllm','vllm.model_executor','vllm.model_executor.layers','vllm.model_executor.layers.fused_moe')
        mods={n:types.ModuleType(n) for n in names}
        base=types.ModuleType('vllm.model_executor.layers.fused_moe.fused_moe_method_base')
        class Base:
            def __init__(self, moe): pass
        base.FusedMoEMethodBase=Base
        bridge=types.ModuleType('adapters.session_bridge')
        class DurableSession:
            terminal=False
            proc=types.SimpleNamespace(poll=lambda:None)
            identity={'run_id':'actual','model_revision':'m','plan_id':'p','layer':0,'epoch':1}
        bridge.DurableSession=DurableSession
        fused=types.ModuleType('vllm.model_executor.layers.fused_moe.layer')
        fused.FusedMoE=Layer
        shared=types.ModuleType('adapters.onecat_shared_guard')
        shared.call_after_shared=lambda *args, **kwargs: None
        mods[base.__name__]=base;mods[bridge.__name__]=bridge;mods[shared.__name__]=shared;mods[fused.__name__]=fused
        with patch.dict(sys.modules,mods):
            sys.modules.pop('adapters.onecat_resident_method',None)
            module=importlib.import_module('adapters.onecat_resident_method')
            session=DurableSession()
            method=module.ResidentMoEMethod(None,session,model_revision='m',plan_id='p',layer_id=0,epoch=1,layer_name='layer',run_id='other')
            with self.assertRaises(module.ResidentAdmissionError): method._check_binding()
            method=module.ResidentMoEMethod(None,session,model_revision='m',plan_id='p',layer_id=0,epoch=1,layer_name='layer')
            method._check_binding()
            session.identity=dict(session.identity,run_id='changed')
            with self.assertRaises(module.ResidentAdmissionError): method._check_binding()
            first=DurableSession(); second=DurableSession()
            second.identity=dict(second.identity,run_id='different',layer=1)
            overlay=ResidentQuantOverlay(Delegate(),{0:first,1:second},model_revision='m',plan_id='p',epoch=1)
            first_method=overlay.get_quant_method(Layer(),'model.language_model.layers.0.mlp.experts')
            second_method=overlay.get_quant_method(Layer(),'model.language_model.layers.1.mlp.experts')
            first_method._check_binding()
            with self.assertRaises(module.ResidentAdmissionError): second_method._check_binding()
            sys.modules.pop('adapters.onecat_resident_method',None)

if __name__=='__main__': unittest.main()
