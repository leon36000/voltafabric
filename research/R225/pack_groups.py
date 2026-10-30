"""Bounded per-rank QPN2 export for a verified complete layer-0 selection."""
import hashlib,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'research/R200'));from pack_checkpoint import pack

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(4*1024*1024):h.update(chunk)
 return h.hexdigest()

def build(manifest, source, dest, cli, cli_sha, revision):
 manifest=pathlib.Path(manifest);source=pathlib.Path(source);dest=pathlib.Path(dest);m=json.loads(manifest.read_bytes());groups=m['groups']
 if m['experts']!=256 or m['layer']!=0 or len(groups)!=32 or [e for g in groups for e in g['ids']]!=list(range(256)):raise ValueError('incomplete coverage')
 if any(g['name']!='group-%03d.safetensors'%g['first'] or len(g['ids'])!=8 for g in groups):raise ValueError('invalid groups')
 dest.mkdir();result={}
 for rank,g in enumerate(groups):
  path=source/g['name']
  if path.is_symlink() or path.stat().st_size!=g['bytes'] or sha(path)!=g['sha256']:raise ValueError('group identity mismatch')
  res=pack(source=path,source_sha=g['sha256'],output=dest/f'owner-{rank:02}',revision=revision,layer=0,epoch=1,groups={rank:g['ids']},cli=cli,cli_sha=cli_sha);result[str(rank)]=res[str(rank)]
 out={'source_manifest_sha256':sha(manifest),'revision':revision,'layer':0,'epoch':1,'ranks':result};(dest/'manifest.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');return out
if __name__=='__main__':
 p=pathlib.Path(sys.argv[1]);d=pathlib.Path(sys.argv[2]);cli=pathlib.Path(sys.argv[3]);out=build(p,p.parent,d,cli,'11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4','1355db6a052410cfd62085d94b58866fd0f2c3c5');print(json.dumps({'status':'SUCCESS','ranks':len(out['ranks'])}))
