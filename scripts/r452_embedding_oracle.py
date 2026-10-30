"""Compare bounded IQ4_XS rows with local GGUF metadata and gguf-py."""
import hashlib,json,os,sys
import numpy as np
from gguf import GGUFReader,GGMLQuantizationType
from gguf.quants import dequantize
p=sys.argv[1];reader=GGUFReader(p);t=next(t for t in reader.tensors if t.name=='token_embd.weight')
if int(t.tensor_type)!=23 or t.shape.tolist()!=[2560,248320] or int(t.data_offset)!=545644544:raise ValueError('embedding metadata')
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd)
for result in sys.argv[2:]:
 r=json.load(open(result));token=r['token'];raw=os.pread(fd,1360,int(t.data_offset)+token*1360)
 if len(raw)!=1360:raise ValueError('short read')
 ref=dequantize(np.frombuffer(raw,dtype=np.uint8).reshape(1,1360),GGMLQuantizationType.IQ4_XS).ravel().astype('<f4')
 got=np.asarray(r['values'],dtype='<f4')
 if got.shape!=(2560,) or not np.array_equal(ref,got) or hashlib.sha256(got.tobytes()).hexdigest()!=r['values_sha256']:raise ValueError('embedding mismatch')
 print('token',token,'values',len(got),'bit_equal',True,'sha256',r['values_sha256'])
after=os.fstat(fd)
if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('source changed')
