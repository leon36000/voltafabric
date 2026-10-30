import hashlib,json,math,os,pathlib,torch
os.environ.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
from safetensors import safe_open
from safetensors.torch import load_file
from qwen_cpu_head import greedy_nvfp4_head
torch.set_num_threads(2);base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r291-twelve-layer-model-r1';model=base/'r188-qwen-checkpoint-pc1-20261028-r1/model';cap=root/'capture.safetensors';r=json.loads((root/'receipt.json').read_text());assert hashlib.sha256(cap.read_bytes()).hexdigest()==r['capture_sha256'];assert len(r['input_tokens'])==2 and r['input_tokens'][1]==r['predicted_tokens'][0]['token_id'];capture=load_file(str(cap));head=model/'model-00003-of-00003.safetensors';results=[];lut=[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]
for b in range(2):
 x=capture[f'batch{b}.output'][0];pred=greedy_nvfp4_head(head,x,vocab=248320,width=2048,chunk_rows=512);assert pred==r['predicted_tokens'][b];i=pred['token_id']
 with safe_open(str(head),framework='pt',device='cpu') as f:
  c=f.get_slice('lm_head.weight')[i:i+1][0].tolist();s=f.get_slice('lm_head.weight_scale')[i:i+1][0].float().tolist();g=float(f.get_tensor('lm_head.weight_scale_2'))
 d=torch.tensor([lut[(c[j//2]>>(4*(j%2)))&15]*s[j//16]*g for j in range(2048)],dtype=torch.float16).tolist();scalar=float(torch.tensor(math.fsum(a*v for a,v in zip(d,x.tolist())),dtype=torch.float16));assert scalar==pred['logit_fp16'];results.append({'batch':b,'token_id':i,'logit_fp16':scalar})
out={'status':'PASS','cases':results,'feedback_token_consistent':True,'gpu':False,'scope':'head rescan and independently checked winning rows only; twelve-layer fixture, not full generation'};(root/'head-verify.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
