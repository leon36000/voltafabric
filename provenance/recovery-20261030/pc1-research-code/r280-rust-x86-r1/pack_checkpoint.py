"""Pack selected Qwen NVFP4 experts into per-rank Rust QPN2 residents.

The caller supplies an immutable compact checkpoint and its verified SHA256. This is
an offline CPU builder; it does not assert that a model or GPU is admitted.
"""
import hashlib,json,os,pathlib,subprocess

def _sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(1024*1024):h.update(chunk)
 return h.hexdigest()

def pack(*,source,source_sha,output,revision,layer,epoch,groups,cli,cli_sha):
 source=pathlib.Path(source);output=pathlib.Path(output);cli=pathlib.Path(cli)
 if source.is_symlink() or not source.is_file() or _sha(source)!=source_sha:raise ValueError('checkpoint identity mismatch')
 if cli.is_symlink() or not cli.is_file() or _sha(cli)!=cli_sha:raise ValueError('CLI identity mismatch')
 if not revision or type(layer)!=int or layer<0 or type(epoch)!=int or epoch<0 or not groups:raise ValueError('invalid binding')
 before=source.stat();fingerprint=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
 seen=set()
 for rank,experts in groups.items():
  if type(rank)!=int or rank<0 or not experts or len(experts)>64:raise ValueError('invalid rank')
  for expert in experts:
   if type(expert)!=int or expert<0 or expert in seen:raise ValueError('duplicate or invalid expert')
   seen.add(expert)
 output.mkdir() # no replace, even after failure
 result={}
 for rank,experts in sorted(groups.items()):
  copies=[{'expert':e,'rank':rank,'local':i} for i,e in enumerate(experts)];spec={'model_revision':revision,'layer':layer,'rank':rank,'registry':{'epoch':epoch,'copies':copies},'resident_order':list(range(len(experts))),'experts':[{'id':e,'local':i,'source_expert':e} for i,e in enumerate(experts)],'expected_metadata':{}}
  target=output/f'rank{rank}.safetensors';req={'source':{'kind':'checkpoint','request':{'path':str(source),'spec':spec}},'output':str(target)};body=json.dumps(req,separators=(',',':'),allow_nan=False).encode();(output/f'rank{rank}.intent.json').write_bytes(body)
  proc=subprocess.run([str(cli),'resident-export','-'],input=body,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120);(output/f'rank{rank}.stdout.json').write_bytes(proc.stdout);(output/f'rank{rank}.stderr').write_bytes(proc.stderr)
  if proc.returncode or not json.loads(proc.stdout).get('ok'):raise RuntimeError('Rust resident-export refused rank '+str(rank))
  current=source.stat()
  if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns)!=fingerprint:raise RuntimeError('checkpoint changed during export')
  result[str(rank)]={'path':str(target),'sha256':_sha(target),'size':target.stat().st_size,'experts':experts}
 (output/'manifest.json').write_text(json.dumps({'source_sha256':source_sha,'revision':revision,'layer':layer,'epoch':epoch,'ranks':result},indent=2,sort_keys=True)+'\n')
 return result
