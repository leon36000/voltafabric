import os,sys,pathlib,json,hashlib,traceback
root=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r193-onecat-model-config-20261028-r1');root.mkdir(mode=0o700);model=root/'model';model.mkdir();(root/'cache').mkdir();(root/'tmp').mkdir();os.environ.update(CUDA_VISIBLE_DEVICES='-1',VLLM_TARGET_DEVICE='cpu',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',VLLM_CACHE_ROOT=str(root/'cache'),XDG_CACHE_HOME=str(root/'cache'),HF_HOME=str(root/'cache'),TMPDIR=str(root/'tmp'))
source=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r188-qwen-checkpoint-pc1-20261028-r1/model/config.json');raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()=='58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc';(model/'config.json').write_bytes(raw);status={'operation_id':'r193-onecat-model-config-20261028-r1','status':'STARTED','gpu_execution':False,'cpu_platform_substituted':True}
try:
 import torch,vllm.platforms as platforms
 from vllm.platforms.cpu import CpuPlatform
 platforms._current_platform=CpuPlatform()
 from vllm.transformers_utils.config import get_config
 config=get_config(str(model),trust_remote_code=False,config_format='hf',local_files_only=True)
 status.update(stage='GET_CONFIG',config_class=type(config).__name__,architectures=config.architectures,text_class=type(config.text_config).__name__,model_type=config.model_type,dtype=str(config.dtype),layers=config.text_config.num_hidden_layers,experts=config.text_config.num_experts,topk=config.text_config.num_experts_per_tok,cuda_initialized=torch.cuda.is_initialized())
 (root/'stage.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
 from vllm.config import ModelConfig
 obj=ModelConfig(model=str(model),tokenizer=str(model),trust_remote_code=False,dtype='half',max_model_len=1024,language_model_only=True)
 status.update(status='SUCCESS',model_architectures=obj.hf_config.architectures,selected_dtype=str(obj.dtype),model_impl=str(obj.model_impl),cuda_initialized=torch.cuda.is_initialized())
except BaseException as e:
 status.update(status='FAILED',error_type=type(e).__name__,error=str(e)[:600]);(root/'traceback.txt').write_text(traceback.format_exc())
(root/'receipt.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n');print(json.dumps(status,sort_keys=True));sys.exit(0 if status['status']=='SUCCESS' else 1)
