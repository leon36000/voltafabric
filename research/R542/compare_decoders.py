"""Compare local GGML C and gguf-py decoders for every layer-3 projection row."""
import hashlib,json,os,stat,subprocess,sys
import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize
if len(sys.argv) not in (3,4) or (len(sys.argv)==4 and sys.argv[3]!='--negative'):
    raise SystemExit('usage: compare_decoders SHARD GGML_ROWS [--negative]')
path,binary=sys.argv[1:3]
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW); before=os.fstat(fd)
if not stat.S_ISREG(before.st_mode) or before.st_size!=55063448064: raise ValueError('shard identity')
matrices=[('q',4105151616,12288,2560,1360,Q.IQ4_XS),('k',4087363712,512,2560,2100,Q.Q6_K),('v',4121864320,512,2560,2100,Q.Q6_K),('out',4088439936,2560,6144,6528,Q.Q8_0)]
proc=subprocess.Popen([binary,path]+(['--negative'] if len(sys.argv)==4 else []),stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
def read_exact(n):
    data=bytearray()
    while len(data)<n:
        b=proc.stdout.read(n-len(data))
        if not b: raise ValueError('short C decoder stream')
        data.extend(b)
    return bytes(data)
try:
    results=[]
    for name,offset,rows,cols,rowbytes,qtype in matrices:
        digest=hashlib.sha256()
        for row in range(rows):
            packed=os.pread(fd,rowbytes,offset+row*rowbytes)
            if len(packed)!=rowbytes: raise ValueError('short source row')
            decoded=dequantize(np.frombuffer(packed,dtype=np.uint8).reshape(1,rowbytes),qtype).reshape(-1).astype('<f4',copy=False)
            if decoded.size!=cols or not np.isfinite(decoded).all(): raise ValueError('invalid Python decode')
            expected=decoded.tobytes(); actual=read_exact(cols*4)
            if actual!=expected: raise ValueError(f'decoder mismatch projection={name} row={row}')
            digest.update(actual)
        results.append({'projection':name,'rows':rows,'values':rows*cols,'bit_equal':True,'sha256':digest.hexdigest()})
    if proc.stdout.read(1): raise ValueError('extra C decoder data')
    if proc.wait(timeout=10)!=0: raise ValueError('C decoder failed')
    after=os.fstat(fd)
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns): raise ValueError('source changed')
    for result in results: print(json.dumps(result,sort_keys=True))
finally:
    proc.stdout.close()
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    os.close(fd)
