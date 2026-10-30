"""Observe bounded PLE spans from one SC117 shard; not publisher authentication."""
import hashlib, json, os, sys
from gguf import GGUFReader
p = sys.argv[1]
r = GGUFReader(p)
tensors = {x.name: x for x in r.tensors}
spec = {'key':('ple_key.weight',30,[2560,10240]),'value':('ple_value.weight',30,[2560,2560]),
        'norm_key':('ple_norm_key.weight',0,[10240]),'norm_query':('ple_norm_query.weight',0,[10240]),
        'norm_conv':('ple_norm_conv.weight',0,[10240]),'kernel':('ple_conv1d.weight',1,[4,10240])}
def identity(s): return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
with open(p,'rb') as f:
    before=os.fstat(f.fileno())
    if before.st_size != 55063448064: raise ValueError('unexpected source size')
    out={'model_revision':'sc117-local-observation','layer':1,'source_bytes':before.st_size}
    for key,(name,kind,shape) in spec.items():
        t=tensors['blk.1.'+name]
        if int(t.tensor_type)!=kind or t.shape.tolist()!=shape or t.n_bytes>52428800: raise ValueError(name)
        f.seek(int(t.data_offset)); data=f.read(int(t.n_bytes))
        if len(data)!=t.n_bytes: raise ValueError('short read')
        out[key]={'offset':int(t.data_offset),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    if identity(before)!=identity(os.fstat(f.fileno())): raise ValueError('source changed')
print(json.dumps(out,sort_keys=True,indent=2))
