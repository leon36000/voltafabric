import hashlib,pathlib,sys,uuid,torch
from safetensors.torch import save_file,load_file
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent));from extract_layer import extract
base=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs')/('r262-extract-test-'+uuid.uuid4().hex);base.mkdir(mode=0o700)
t={f'model.language_model.layers.0.mlp.experts.{e}.{p}.{field}':torch.tensor([e+1],dtype=torch.float32) for e in range(2) for p in ('gate_proj','up_proj','down_proj') for field in ('weight','weight_scale','weight_scale_2','input_scale')};src=base/'src.safetensors';save_file(t,str(src));digest=hashlib.sha256(src.read_bytes()).hexdigest();dst=base/'out.safetensors';r=extract(src,digest,dst,[0,1]);assert r['tensor_count']==24 and all(torch.equal(load_file(str(dst))[k],v) for k,v in t.items())
try:extract(src,'0'*64,base/'bad.safetensors',[0]);raise AssertionError('wrong hash accepted')
except ValueError:pass
try:extract(src,digest,base/'duplicate.safetensors',[1,1]);raise AssertionError('duplicate accepted')
except ValueError:pass
try:extract(src,digest,dst,[0]);raise AssertionError('overwrite accepted')
except FileExistsError:pass
print('PASS: 24 tensors exact; bad hash, duplicate and overwrite refused')
