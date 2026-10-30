import pathlib,sys,types,unittest
R=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));from adapters.onecat_weight_filter import ResidentWeightFilter,ResidentFilterError
FIELDS=[f'{p}.{f}' for p in ('gate_proj','up_proj','down_proj') for f in ('weight','weight_scale','weight_scale_2','input_scale')]
class Session:
 experts=frozenset(range(2))
class Method:
 def __init__(self):self._admitted=True;self.binding={'layer':0};self.layer_name='model.language_model.layers.0.mlp.experts';self.session=Session();self.moe=types.SimpleNamespace(num_experts=2);self.alive=True
 def _check_binding(self):
  if not self.alive:raise RuntimeError('session stopped')
def items():return [(f'model.language_model.layers.0.mlp.experts.{e}.{field}',e) for e in range(2) for field in FIELDS]+[('model.language_model.layers.1.mlp.experts.0.gate_proj.weight',7),('model.language_model.layers.0.mlp.shared_expert.gate_proj.weight',8)]
class Test(unittest.TestCase):
 def test_preserves_other_layers_and_shared(self):
  f=ResidentWeightFilter({0:Method()},Method);out=list(f.filter(iter(items())));self.assertEqual([x[1] for x in out],[7,8]);self.assertTrue(f.finished)
 def test_missing_duplicate_unknown_are_refused(self):
  for values in (items()[1:],items()+[items()[0]],items()+[('model.language_model.layers.0.mlp.experts.0.other',1)]):
   f=ResidentWeightFilter({0:Method()},Method)
   with self.assertRaises(ResidentFilterError):list(f.filter(iter(values)))
   self.assertFalse(f.finished)
 def test_session_loss_is_refused(self):
  m=Method();f=ResidentWeightFilter({0:m},Method)
  def stream():
   yield items()[0];m.alive=False;yield items()[-1]
  with self.assertRaises(RuntimeError):list(f.filter(stream()))
  self.assertFalse(f.finished)
 def test_incomplete_coverage_is_refused(self):
  m=Method();m.session=types.SimpleNamespace(experts=frozenset({0}))
  with self.assertRaises(ResidentFilterError):ResidentWeightFilter({0:m},Method)
 def test_loader_requires_exhaustion_and_restores(self):
  class Loader:
   local_expert_ids=None
   def get_all_weights(self,c,m):return iter(items())
   def load_weights(self,m,c):next(self.get_all_weights(c,m))
  l=Loader();f=ResidentWeightFilter({0:Method()},Method)
  with self.assertRaises(ResidentFilterError):f.load(l,None,None)
  self.assertNotIn('get_all_weights',vars(l))
if __name__=='__main__':unittest.main()
