import pathlib, sys, types, unittest
from unittest.mock import patch
R=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from adapters.onecat_weight_filter import ResidentFilterError
from adapters.onecat_pre_read import ResidentPreReadFilter
FIELDS=[f'{p}.{f}' for p in ('gate_proj','up_proj','down_proj') for f in ('weight','weight_scale','weight_scale_2','input_scale')]
class Method:
 def __init__(self):
  self._admitted=True;self.binding={'layer':0};self.layer_name='model.language_model.layers.0.mlp.experts';self.session=types.SimpleNamespace(experts=frozenset(range(2)));self.moe=types.SimpleNamespace(num_experts=2);self.alive=True
 def _check_binding(self):
  if not self.alive:raise RuntimeError('session lost')
class File:
 def __init__(self,names,read):self.names=names;self.read=read
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def keys(self):return self.names
 def get_tensor(self,name):self.read.append(name);return name
class Loader:
 def __init__(self):self.local_expert_ids=None;self.counter_before_loading_weights=0.0;self.load_config=types.SimpleNamespace(load_format='safetensors',model_loader_extra_config={},safetensors_load_strategy=None)
 def _prepare_weights(self,*a):return '/fixture',['fixture.safetensors'],True
 def _get_weights_iterator(self,source):raise AssertionError('old iterator')
 def get_all_weights(self,config,target):return self._get_weights_iterator(types.SimpleNamespace(model_or_path='/fixture',subfolder=None,revision=None,fall_back_to_pt=False,allow_patterns_overrides=None,prefix=''))
 def load_weights(self,model,config):model.extend(self.get_all_weights(config,model))
def names():return [f'model.language_model.layers.0.mlp.experts.{e}.{f}' for e in range(2) for f in FIELDS]+['model.language_model.layers.0.mlp.gate.weight']
class Tests(unittest.TestCase):
 def run_loader(self,keys,method=None,loader=None):
  reads=[];loader=loader or Loader();result=[];f=ResidentPreReadFilter({0:method or Method()},Method,opener=lambda *_a,**_k:File(keys,reads),is_local=lambda _:True)
  f.load(loader,None,result);return reads,result,loader
 def test_skips_before_get_tensor(self):
  reads,output,loader=self.run_loader(names());self.assertEqual(reads,['model.language_model.layers.0.mlp.gate.weight']);self.assertEqual(len(output),1);self.assertNotIn('_get_weights_iterator',vars(loader));self.assertNotIn('get_all_weights',vars(loader))
 def test_missing_duplicate_unknown_refused(self):
  for keys in (names()[1:],names()+[names()[0]],names()+['model.language_model.layers.0.mlp.experts.0.unknown']):
   with self.subTest(keys=len(keys)),self.assertRaises(ResidentFilterError):self.run_loader(keys)
 def test_does_not_accept_eager_prefetch_or_existing_filter(self):
  for kind in ('eager','prefetch','torchao'):
   loader=Loader();loader.load_config.safetensors_load_strategy=kind
   with self.subTest(kind=kind),self.assertRaises(ResidentFilterError):self.run_loader(names(),loader=loader)
  loader=Loader();loader.local_expert_ids={0}
  with self.assertRaises(ResidentFilterError):self.run_loader(names(),loader=loader)
 def test_early_stop_refused_and_restores(self):
  loader=Loader();loader.load_weights=lambda m,c:next(loader.get_all_weights(c,m))
  with self.assertRaises(ResidentFilterError):self.run_loader(names(),loader=loader)
  self.assertNotIn('_get_weights_iterator',vars(loader));self.assertNotIn('get_all_weights',vars(loader))
 def test_shards_follow_natural_filename_order(self):
  loader=Loader();loader._prepare_weights=lambda *a:('/fixture',['/fixture/model-10.safetensors','/fixture/model-2.safetensors'],True)
  opened=[]
  def opener(path,**kwargs):
   opened.append(path)
   return File(names() if path.endswith('2.safetensors') else [],[])
  f=ResidentPreReadFilter({0:Method()},Method,opener=opener,is_local=lambda _:True)
  f.load(loader,None,[])
  self.assertEqual(opened,['/fixture/model-2.safetensors','/fixture/model-10.safetensors'])
 def test_session_loss_prevents_read(self):
  m=Method();reads=[]
  class F(File):
   def keys(self):
    yield names()[0];m.alive=False;yield names()[-1]
  f=ResidentPreReadFilter({0:m},Method,opener=lambda *_a,**_k:F([],reads),is_local=lambda _:True)
  with self.assertRaises(RuntimeError):f.load(Loader(),None,[])
  self.assertEqual(reads,[])
if __name__=='__main__':unittest.main()
