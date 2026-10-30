import hashlib,json,os,sys
from pathlib import Path
import pip._internal.operations.prepare as prep
from pip._internal.cli.main import main
root=Path(sys.argv[1]);wheel=sys.argv[2];files=[Path(prep.__file__),Path(prep.__file__).parent.parent/'resolution/resolvelib/resolver.py'];expected=['8e8589c0f92ea86b1c42054d2262caef57bd8516a9c0abd108cf07725cac9af5','9cb24eb15304562da0414549a1414a31901ebb67fb19132318cbcd496cb3d017']
if [hashlib.sha256(x.read_bytes()).hexdigest() for x in files]!=expected:raise ValueError('pip internal source differs')
seen=[]
def metadata_only(self,reqs):
 if seen:raise ValueError('hook called more than once')
 out=[]
 for r in reqs:
  link=r.link
  out.append({'name':r.name,'requirement':str(r.req) if r.req else None,'link':link.url if link else None,'requested':bool(r.user_supplied)})
 if not out or any(not x['name'] or not x['requirement'] or not x['link'] for x in out):raise ValueError('incomplete candidates')
 data={'status':'PIP_RESOLVER_METADATA_CANDIDATES_NOT_INSTALLED','note':'Patched pip hook omits final wheel preparation; this is a candidate set, not a valid installation or completed pip report. Earlier metadata preparation may download files; caps apply.','pip_internal_sha256':expected,'candidates':out}
 with (root/'candidates.json').open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 seen.append(len(out))
prep.RequirementPreparer.prepare_linked_requirements_more=metadata_only
args=['install','--dry-run','--ignore-installed','--only-binary=:all:','--no-cache-dir','--timeout','8','--retries','0','--index-url','https://pypi.org/simple',wheel]
code=main(args);print('PATCHED_PIP_RETURN',code,'HOOK_COUNT',len(seen),'CANDIDATES',seen);sys.exit(code if seen else 88)
