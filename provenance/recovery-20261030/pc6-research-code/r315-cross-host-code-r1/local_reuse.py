"""Local-only CPU receiver: validate streamed bytes and reuse pinned resident file.

This optimization still receives every byte; it avoids an additional disk copy.
A fresh durable identity is consumed before any data is accepted.
"""
import hashlib,json,os,pathlib,stat,threading
from adapters.resident_multipart import MultipartError,MAX_ARTIFACT_BYTES,MAX_CHUNK_BYTES,_fsync_dir

class LocalReuse:
 def __init__(self,root,identity,total_bytes,sha256,approved):
  fields={'run_id','model_revision','plan_id','layer','epoch','rank','attempt'}
  if (type(identity) is not dict or set(identity)!=fields or any(type(identity[k]) is not str or not identity[k] or len(identity[k])>512 for k in ('run_id','model_revision','plan_id')) or any(type(identity[k]) is not int or not 0<=identity[k]<2**64 for k in ('layer','epoch','rank','attempt')) or type(total_bytes) is not int or not 0<total_bytes<=MAX_ARTIFACT_BYTES or type(sha256) is not str or len(sha256)!=64 or any(c not in '0123456789abcdef' for c in sha256)):
   raise MultipartError('invalid transfer identity or bounds')
  root=pathlib.Path(root)
  if root.is_symlink() or not root.is_dir():raise MultipartError('existing non-symlink root required')
  if type(approved) is not dict or set(approved)!={'path','sha256','bytes'} or approved['sha256']!=sha256 or type(approved['bytes']) is not int or approved['bytes']!=total_bytes:raise MultipartError('local artifact not approved')
  self.path=pathlib.Path(approved['path']);raw=json.dumps(identity,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode();self.directory=root/hashlib.sha256(raw).hexdigest();self.directory.mkdir(mode=0o700);_fsync_dir(root)
  self.identity=dict(identity);self.size=total_bytes;self.sha=sha256;self.offset=0;self.hash=hashlib.sha256();self.state='OPEN';self.lock=threading.Lock()
  try:
   with (self.directory/'intent.json').open('xb') as f:
    f.write(json.dumps({'identity':identity,'bytes':total_bytes,'sha256':sha256,'local_path':str(self.path)},sort_keys=True).encode());f.flush();os.fsync(f.fileno())
   _fsync_dir(self.directory)
  except BaseException:self.state='UNKNOWN';raise
 def append(self,offset,data):
  with self.lock:
   if self.state!='OPEN':raise MultipartError('operation is terminal')
   if type(offset) is not int or offset!=self.offset or type(data) is not bytes or not 0<len(data)<=MAX_CHUNK_BYTES or self.offset+len(data)>self.size:
    self.state='FAILED';raise MultipartError('invalid chunk or offset; identity consumed')
   self.hash.update(data);self.offset+=len(data)
 def finish(self,validate):
  with self.lock:
   if self.state!='OPEN':raise MultipartError('operation is terminal')
   if not callable(validate) or self.offset!=self.size or self.hash.hexdigest()!=self.sha:self.state='FAILED';raise MultipartError('incomplete, hash mismatch or validator missing')
   try:
    fd=os.open(self.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
     before=os.fstat(fd)
     if not stat.S_ISREG(before.st_mode) or before.st_size!=self.size:raise MultipartError('invalid local artifact')
     h=hashlib.sha256()
     with os.fdopen(fd,'rb',closefd=False) as f:
      while chunk:=f.read(MAX_CHUNK_BYTES):h.update(chunk)
     if h.hexdigest()!=self.sha:raise MultipartError('local hash mismatch')
     validate(self.path,dict(self.identity));after=os.fstat(fd);pathstat=os.stat(self.path,follow_symlinks=False)
     signature=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
     if signature(before)!=signature(after) or signature(after)!=signature(pathstat):raise MultipartError('local artifact changed during validation')
    finally:os.close(fd)
   except BaseException:self.state='FAILED';raise
   try:
    with (self.directory/'resident.json').open('xb') as f:
     f.write(json.dumps({'sha256':self.sha,'bytes':self.size,'path':str(self.path)},sort_keys=True).encode());f.flush();os.fsync(f.fileno())
    _fsync_dir(self.directory)
   except BaseException:self.state='UNKNOWN';raise
   self.state='COMMITTED';return self.path
