import json,hashlib,pathlib
from scripts.resident_artifact import load_stack
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');rows=[]
for layer in range(16,20):
 manifest=json.loads((base/f'r294-layer{layer}-owners-r2/manifest.json').read_text());seen=set()
 for rank in range(4):
  v=manifest['ranks'][str(rank)];p=pathlib.Path(v['path']);h=hashlib.sha256()
  with p.open('rb') as f:
   while block:=f.read(1024*1024):h.update(block)
  assert h.hexdigest()==v['sha256'];m=load_stack(p,expected_sha256=v['sha256']).metadata;ids=[x['id'] for x in m['physical_experts']];assert m['layer']==layer and m['rank']==rank and m['epoch']==1 and ids==list(range(rank*64,(rank+1)*64));assert not seen.intersection(ids);seen.update(ids);rows.append({'layer':layer,'rank':rank,'sha256':v['sha256'],'bytes':p.stat().st_size})
 assert seen==set(range(256))
out={'status':'PASS','layers':4,'experts_per_layer':256,'owners_per_layer':4,'max_artifact_bytes':max(v['bytes'] for v in rows),'artifacts':rows,'gpu':False};(base/'r294-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='artifacts'}))
