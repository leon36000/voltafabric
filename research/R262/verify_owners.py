import hashlib,json,pathlib,sys
from scripts.resident_artifact import load_stack
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'owners/manifest.json').read_text());seen=set();results=[]
assert manifest['source_sha256']=='9b03db3babfc11af78d6c02fef2bba3435223f58a0f6d27bb952e60d41a11ddc' and set(manifest['ranks'])==set(map(str,range(16)))
for rank in range(16):
 a=manifest['ranks'][str(rank)];p=pathlib.Path(a['path']);h=hashlib.sha256()
 with p.open('rb') as f:
  while part:=f.read(1024*1024):h.update(part)
 assert h.hexdigest()==a['sha256'];r=load_stack(p,expected_sha256=a['sha256']);info=r.metadata;ids=[x['id'] for x in info['physical_experts']];assert info['rank']==rank and info['layer']==0 and info['epoch']==1 and ids==list(range(rank*16,(rank+1)*16));assert not seen.intersection(ids);seen.update(ids);results.append({'rank':rank,'bytes':p.stat().st_size,'sha256':a['sha256'],'experts':len(ids)})
assert seen==set(range(256));out={'status':'PASS','experts':len(seen),'ranks':results,'gpu':False};(root/'owners-verify.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','ranks':len(results),'experts':len(seen),'bytes':sum(x['bytes'] for x in results)}))
