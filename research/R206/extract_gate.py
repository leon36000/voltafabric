"""Bounded extraction of the Qwen layer-0 gate from a verified immutable shard."""
import hashlib,json,os,pathlib,stat,struct,sys
source=pathlib.Path(sys.argv[1]);dest=pathlib.Path(sys.argv[2]);expected=sys.argv[3];name='model.language_model.layers.0.mlp.gate.weight'
fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW)
try:
 st=os.fstat(fd);assert stat.S_ISREG(st.st_mode);fingerprint=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 with os.fdopen(fd,'rb',closefd=False) as f:
  h=hashlib.sha256()
  while chunk:=f.read(4194304):h.update(chunk)
  if h.hexdigest()!=expected or fingerprint(st)!=fingerprint(os.fstat(fd)):raise ValueError('source changed or SHA mismatch')
  f.seek(0);n=struct.unpack('<Q',f.read(8))[0]
  if not 0<n<=8*1024*1024:raise ValueError('header limit')
  meta=json.loads(f.read(n));item=meta[name];start,end=item['data_offsets']
  if item['dtype']!='BF16' or item['shape']!=[256,2048] or end-start!=1048576 or not 0<=start<end<=st.st_size-8-n:raise ValueError('unexpected gate layout')
  f.seek(8+n+start);body=f.read(end-start)
  if len(body)!=end-start or fingerprint(st)!=fingerprint(os.fstat(fd)):raise ValueError('short read or changed source')
  header=json.dumps({'__metadata__':{'format':'pt'},name:{'dtype':'BF16','shape':[256,2048],'data_offsets':[0,len(body)]}},separators=(',',':')).encode();header+=b' '*((-len(header))%8)
  with dest.open('xb') as out:out.write(struct.pack('<Q',len(header))+header+body);out.flush();os.fsync(out.fileno())
 print(json.dumps({'source_sha256':expected,'gate_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'gate_bytes':dest.stat().st_size,'name':name},sort_keys=True))
finally:os.close(fd)
