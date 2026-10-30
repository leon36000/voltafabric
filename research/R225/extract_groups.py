"""Extract one verified shard's MoE weights into bounded per-owner groups."""
import hashlib,json,os,pathlib,stat,struct
MAX_HEADER=8*1024*1024
MAX_OUTPUT=16*1024*1024-65536

def extract(source, expected_sha, destination, *, layer=0, total=256, per_group=8):
 if not isinstance(expected_sha,str) or len(expected_sha)!=64 or any(c not in '0123456789abcdef' for c in expected_sha):raise ValueError('invalid digest')
 if type(total)!=int or not 0<total<=1024 or type(per_group)!=int or not 0<per_group<=8:raise ValueError('selection bound')
 source=pathlib.Path(source);destination=pathlib.Path(destination);fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  st=os.fstat(fd)
  if not stat.S_ISREG(st.st_mode):raise ValueError('not regular')
  fingerprint=lambda a:(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)
  with os.fdopen(fd,'rb',closefd=False) as f:
   h=hashlib.sha256()
   while chunk:=f.read(4*1024*1024):h.update(chunk)
   if h.hexdigest()!=expected_sha or fingerprint(st)!=fingerprint(os.fstat(fd)):raise ValueError('source SHA or stability')
   f.seek(0);raw=f.read(8)
   if len(raw)!=8:raise ValueError('header')
   n=struct.unpack('<Q',raw)[0]
   if not 0<n<=MAX_HEADER or 8+n>st.st_size:raise ValueError('header size')
   def unique(pairs):
    d={}
    for k,v in pairs:
     if k in d:raise ValueError('duplicate key')
     d[k]=v
    return d
   meta=json.loads(f.read(n),object_pairs_hook=unique);groups=[];all_spans=[]
   for start in range(0,total,per_group):
    ids=list(range(start,min(start+per_group,total)));names=[f'model.language_model.layers.{layer}.mlp.experts.{e}.{plane}.{field}' for e in ids for plane in ('gate_proj','up_proj','down_proj') for field in ('weight','weight_scale','weight_scale_2')];spans=[]
    for name in names:
     entry=meta[name];a,b=entry['data_offsets']
     if type(a)!=int or type(b)!=int or not 0<=a<=b<=st.st_size-8-n:raise ValueError('offset')
     spans.append((a,b));all_spans.append((a,b))
    groups.append((ids,names,spans))
   ordered=sorted(all_spans)
   if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):raise ValueError('overlapping tensors')
   destination.mkdir();result=[]
   for ids,names,spans in groups:
    hdr={'__metadata__':meta.get('__metadata__',{})};offset=0
    for name,(a,b) in zip(names,spans):
     item=meta[name];hdr[name]={'dtype':item['dtype'],'shape':item['shape'],'data_offsets':[offset,offset+b-a]};offset+=b-a
    raw=json.dumps(hdr,separators=(',',':'),ensure_ascii=False).encode();raw+=b' '*((-len(raw))%8)
    if 8+len(raw)+offset>MAX_OUTPUT:raise ValueError('group exceeds wire frame')
    path=destination/('group-%03d.safetensors'%ids[0]);digest=hashlib.sha256()
    with path.open('xb') as out:
     prefix=struct.pack('<Q',len(raw))+raw;out.write(prefix);digest.update(prefix)
     for a,b in spans:
      f.seek(8+n+a);remaining=b-a
      while remaining:
       chunk=f.read(min(1024*1024,remaining))
       if not chunk:raise ValueError('short tensor')
       out.write(chunk);digest.update(chunk);remaining-=len(chunk)
     out.flush();os.fsync(out.fileno())
    result.append({'first':ids[0],'ids':ids,'bytes':path.stat().st_size,'sha256':digest.hexdigest(),'name':path.name})
   if fingerprint(st)!=fingerprint(os.fstat(fd)):raise ValueError('source changed')
  receipt={'source_sha256':expected_sha,'layer':layer,'experts':total,'groups':result};(destination/'manifest.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n');return receipt
 finally:os.close(fd)
if __name__=='__main__':
 import sys
 print(json.dumps(extract(sys.argv[1],sys.argv[2],sys.argv[3]),sort_keys=True))
