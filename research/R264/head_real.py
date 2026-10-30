import hashlib,json,math,os,pathlib,sys,torch
os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer
from qwen_cpu_head import greedy_nvfp4_head
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');model=base/'r188-qwen-checkpoint-pc1-20261028-r1/model';head=model/'model-00003-of-00003.safetensors';capture=base/'r263-qwen-full-gate-20261028-r1/capture.safetensors';root=base/'r264-real-head-r1';torch.set_num_threads(2)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while c:=f.read(4*1024*1024):h.update(c)
 return h.hexdigest()
assert sha(head)=='9758875fc55e49561165f4a44342b654c123c3e25c6811b34abb83553fb1a164';assert sha(capture)=='a24822446c0500c20d66438b9a9008875636a7c780e202661db73f13374fb222';assert sha(model/'tokenizer.json')=='5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False);tokens=tok.encode('Bonjour',add_special_tokens=False);assert tokens and all(0<=t<248320 for t in tokens)
with safe_open(str(model/'model-00001-of-00003.safetensors'),framework='pt',device='cpu') as f:
 embed=f.get_slice('model.language_model.embed_tokens.weight');assert tuple(embed.get_shape())==(248320,2048);row=embed[tokens[0]:tokens[0]+1];assert row.dtype==torch.bfloat16 and torch.isfinite(row.float()).all()
vector=load_file(str(capture))['batch1.output'][0];answer=greedy_nvfp4_head(head,vector,vocab=248320,width=2048,chunk_rows=512)
# Independent scalar check on the winning row, with FP16-rounded decoded weights.
with safe_open(str(head),framework='pt',device='cpu') as f:
 i=answer['token_id'];c=f.get_slice('lm_head.weight')[i:i+1][0].tolist();s=f.get_slice('lm_head.weight_scale')[i:i+1][0].float().tolist();g=float(f.get_tensor('lm_head.weight_scale_2'))
lut=[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.];decoded=[lut[(byte>>(4*(j%2)))&15]*s[j//16]*g for j in range(2048) for byte in [c[j//2]]];d=torch.tensor(decoded,dtype=torch.float16).tolist();scalar=float(torch.tensor(math.fsum(a*b for a,b in zip(d,vector.tolist())),dtype=torch.float16));assert scalar==answer['logit_fp16'],(scalar,answer)
root.mkdir(mode=0o700);result={'status':'PASS','model_revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5','head_shard_sha256':sha(head),'capture_sha256':sha(capture),'tokenizer_input_ids':tokens,'embedding_shape':list(row.shape),'embedding_dtype':str(row.dtype),'head':answer,'scalar_winner_logit':scalar,'gpu':False,'cuda_initialized':torch.cuda.is_initialized(),'scope':'real embedding lookup and real head on synthetic one-layer activation; not full model generation'};(root/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
