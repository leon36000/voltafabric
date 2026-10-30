import hashlib,json,pathlib,struct,sys
from extract_layer import extract
root=pathlib.Path(sys.argv[1]);root.mkdir();header={};offset=0
for plane in ('gate_proj','up_proj','down_proj'):
 for field in ('weight','weight_scale','weight_scale_2','input_scale'):
  header[f'model.language_model.layers.16.mlp.experts.139.{plane}.{field}']={'dtype':'U8','shape':[1],'data_offsets':[offset,offset+1]};offset+=1
raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*(9*1024*1024-len(raw));data=struct.pack('<Q',len(raw))+raw+bytes(range(offset));src=root/'source.safetensors';src.write_bytes(data);result=extract(src,hashlib.sha256(data).hexdigest(),root/'selected.safetensors',[139],16);assert result['tensor_count']==12 and result['tensor_bytes']==12
try:extract(src,'0'*64,root/'wrong.safetensors',[139],16);raise AssertionError('wrong SHA accepted')
except ValueError:pass
huge=root/'oversized.safetensors';huge.write_bytes(struct.pack('<Q',16*1024*1024+1)+b' '*(16*1024*1024+1))
try:extract(huge,hashlib.sha256(huge.read_bytes()).hexdigest(),root/'bad.safetensors',[139],16);raise AssertionError('oversized header accepted')
except ValueError:pass
print('PASS 9 MiB header extracts 12 tensors; wrong hash and >16 MiB header rejected')
