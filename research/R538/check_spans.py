"""Check the observed layer-3 projection spans used by the scalar reference."""
import hashlib,json,os,sys
if len(sys.argv)!=3: raise SystemExit('usage: check_spans MANIFEST SHARD')
m=json.load(open(sys.argv[1])); fd=os.open(sys.argv[2],os.O_RDONLY|os.O_NOFOLLOW); before=os.fstat(fd)
if before.st_size!=m['shard_bytes']: raise ValueError('shard size')
for name in ('q0','q1','k','v','out'):
    x=m[name]; h=hashlib.sha256(); remain=x['bytes']; offset=x['offset']
    while remain:
        b=os.pread(fd,min(remain,1<<20),offset)
        if not b: raise ValueError('short span')
        h.update(b); remain-=len(b); offset+=len(b)
    if h.hexdigest()!=x['sha256']: raise ValueError(name+' mismatch')
    print(json.dumps({'span':name,'bytes':x['bytes'],'sha256':h.hexdigest()},sort_keys=True))
after=os.fstat(fd)
if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns): raise ValueError('shard changed')
os.close(fd)
