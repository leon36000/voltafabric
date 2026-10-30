import pathlib,sys,uuid,torch
from safetensors.torch import save_file
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';sys.path.insert(0,str(R));from adapters.qwen_cpu_head import greedy_nvfp4_head,HeadError
root=B/'.runs'/('r264-head-test-'+uuid.uuid4().hex);root.mkdir(mode=0o700);p=root/'head.safetensors';codes=torch.tensor([[0x11]*16,[0x22]*16,[0x33]*16,[0x33]*16],dtype=torch.uint8);scales=torch.ones((4,2),dtype=torch.float8_e4m3fn);save_file({'lm_head.weight':codes,'lm_head.weight_scale':scales,'lm_head.weight_scale_2':torch.tensor(1.)},str(p));x=torch.arange(32,dtype=torch.float16)/32
# Independent hand calculation: each row uses a constant positive E2M1 code.
expected=[float((x.double().sum()*v).half()) for v in (.5,1.,1.5,1.5)];result=greedy_nvfp4_head(p,x,vocab=4,width=32,chunk_rows=2);assert result['token_id']==2 and result['logit_fp16']==expected[2]
try:greedy_nvfp4_head(p,x,vocab=5,width=32);raise AssertionError('bad geometry accepted')
except HeadError:pass
try:greedy_nvfp4_head(p,torch.full((32,),float('nan'),dtype=torch.float16),vocab=4,width=32);raise AssertionError('NaN accepted')
except HeadError:pass
print('PASS: independent values, tie ordering, geometry and NaN refusal')
