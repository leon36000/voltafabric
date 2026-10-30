import ast,hashlib,json,pathlib,types,unittest
p=pathlib.Path(__file__).parent;r=json.loads((p/'receipt.json').read_text());c=json.loads((p/'config.json').read_bytes());assert r['status']=='VERIFIED' and r['root_dtype']=='bfloat16' and r['text_dtype']=='bfloat16';assert hashlib.sha256((p/'config.json').read_bytes()).hexdigest()==r['sha256']
s=(p/'source.py').read_bytes();assert hashlib.sha256(s).hexdigest()==json.loads((p/'source-receipt.json').read_text())['member_sha256'];t=ast.parse(s)
cls=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='ModelArchConfigConvertorBase');f=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='get_torch_dtype');f.decorator_list=[];m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[]);ast.fix_missing_locations(m)
def no_io(*a,**k):raise AssertionError('metadata lookup unexpected')
g={'torch':types.SimpleNamespace(float32='f32'),'get_safetensors_params_metadata':no_io};exec(compile(m,'<wheel-source-ast>','exec'),g)
class Config:
 def __init__(self,root,text,vision=None):self.dtype=root;self._text=text;self.vision_config=types.SimpleNamespace(dtype=vision)
 def get_text_config(self):return types.SimpleNamespace(dtype=self._text)
def conv(x):return g['get_torch_dtype'](None,x,'test-model',None,'hf')
class TestConverter(unittest.TestCase):
 def test_qwen_root_bf16(self):self.assertEqual(conv(Config('bf16','bf16')),'bf16')
 def test_root_precedes_text(self):self.assertEqual(conv(Config('f16','bf16')),'f16')
 def test_text_fallback(self):self.assertEqual(conv(Config(None,'bf16')),'bf16')
 def test_vision_fallback(self):self.assertEqual(conv(Config(None,None,'f16')),'f16')
if __name__=='__main__':unittest.main(verbosity=2)
