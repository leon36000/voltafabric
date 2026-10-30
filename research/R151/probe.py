import hashlib,importlib.util,json,pathlib,sys,traceback
p=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r150-transformers-config-cpu-20261028/config.json');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc'
import torch,transformers
from transformers import AutoConfig
out={'transformers':transformers.__version__,'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'config_sha256':hashlib.sha256(b).hexdigest(),'transformers_qwen35_spec':importlib.util.find_spec('transformers.models.qwen3_5') is not None}
try:
 config=AutoConfig.from_pretrained(str(p.parent),local_files_only=True,trust_remote_code=False);out.update(status='SPECIFIC_CONFIG_LOADED',config_class=type(config).__name__,dtype=str(config.dtype),dtype_is_bf16=config.dtype is torch.bfloat16,text_config_class=type(config.text_config).__name__,text_dtype=str(config.text_config.dtype),head_dim=getattr(config.text_config,'linear_key_head_dim',None),layer_count=len(config.text_config.layer_types))
except Exception as e:out.update(status='SPECIFIC_CONFIG_FAILED',exception=type(e).__name__,reason=str(e)[:1200])
print(json.dumps(out,sort_keys=True));sys.exit(0 if out['status']=='SPECIFIC_CONFIG_LOADED' else 1)
