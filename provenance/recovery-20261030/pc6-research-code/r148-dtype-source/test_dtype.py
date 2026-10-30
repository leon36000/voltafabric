import ast,hashlib,json,pathlib,types,unittest
from unittest.mock import patch
p=pathlib.Path(__file__).parent; receipt=json.loads((p/'receipt.json').read_text()); sources={n:(p/f).read_bytes() for n,f in [('vllm/config/model.py','model.py'),('vllm/platforms/cuda.py','cuda.py')]}
for n,data in sources.items():assert hashlib.sha256(data).hexdigest()==receipt['member_sha256'][n]
torch=types.SimpleNamespace(float16='f16',bfloat16='bf16',float32='f32'); model=ast.parse(sources['vllm/config/model.py']);nodes=[]
for n in model.body:
 if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='_FLOAT16_NOT_SUPPORTED_MODELS' for a in n.targets):nodes.append(n)
 if isinstance(n,ast.FunctionDef) and n.name in ('_is_valid_dtype','_resolve_auto_dtype'):nodes.append(n)
assert len(nodes)==3
cuda=ast.parse(sources['vllm/platforms/cuda.py']); matches=[n for n in ast.walk(cuda) if isinstance(n,ast.FunctionDef) and n.name=='supported_dtypes'];assert len(matches)==1; method=matches[0];method.decorator_list=[];nodes.append(method)
mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);ast.fix_missing_locations(mod)
class Platform:
 def __init__(self,cap):self.cap=cap
 def has_device_capability(self,n):return self.cap>=n
 def get_device_name(self):return 'simulated'
 def get_device_capability(self):return types.SimpleNamespace(as_version_str=lambda:str(self.cap))
g={'torch':torch,'logger':types.SimpleNamespace(warning=lambda *a:None)};exec(compile(mod,'<wheel-source-ast>','exec'),g)
def resolve(cap,dtype):
 platform=Platform(cap);platform.supported_dtypes=g['supported_dtypes'](platform);g['current_platform']=platform
 return g['_resolve_auto_dtype']('qwen3_5_moe',dtype,is_pooling_model=False)
class TestAutoDtype(unittest.TestCase):
 def test_bf16_sm70_fallback_to_fp16(self):self.assertEqual(resolve(70,torch.bfloat16),torch.float16)
 def test_bf16_sm80_stays_bf16(self):self.assertEqual(resolve(80,torch.bfloat16),torch.bfloat16)
 def test_f32_sm70_auto_downcasts(self):self.assertEqual(resolve(70,torch.float32),torch.float16)
 def test_fp16_sm70_unchanged(self):self.assertEqual(resolve(70,torch.float16),torch.float16)
if __name__=='__main__':unittest.main(verbosity=2)
