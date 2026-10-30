import hashlib,json,pathlib,sys,types,zipfile
root=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');wheel=root/'r125-isolated-1cat-20261028/1cat_vllm-1.2.2-cp312-cp312-linux_x86_64.whl';conf=root/'r150-transformers-config-cpu-20261028/config.json';h=hashlib.sha256()
with wheel.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()=='8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6';raw=conf.read_bytes();assert hashlib.sha256(raw).hexdigest()=='58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc'
with zipfile.ZipFile(wheel) as z:source=z.read('vllm/transformers_utils/configs/qwen3_5_moe.py')
import torch,transformers
out={'wheel_sha256':h.hexdigest(),'config_sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':hashlib.sha256(source).hexdigest(),'transformers':transformers.__version__,'torch':torch.__version__,'torch_cuda_build':torch.version.cuda}
try:
 mod=types.ModuleType('vf_r151_qwen_config');sys.modules[mod.__name__]=mod;exec(compile(source,'<1cat-wheel-qwen-config>','exec'),mod.__dict__)
 c=mod.Qwen3_5MoeConfig.from_dict(json.loads(raw));out.update(status='SPECIFIC_1CAT_CONFIG_LOADED',class_name=type(c).__name__,dtype=str(c.dtype),dtype_is_bf16=c.dtype is torch.bfloat16,text_class=type(c.text_config).__name__,text_dtype=str(c.text_config.dtype),head_dim=c.text_config.linear_key_head_dim,layer_count=len(c.text_config.layer_types),linear_count=c.text_config.layer_types.count('linear_attention'))
except Exception as e:out.update(status='SPECIFIC_1CAT_CONFIG_FAILED',exception=type(e).__name__,reason=str(e)[:1000])
print(json.dumps(out,sort_keys=True));sys.exit(0 if out['status']=='SPECIFIC_1CAT_CONFIG_LOADED' else 1)
