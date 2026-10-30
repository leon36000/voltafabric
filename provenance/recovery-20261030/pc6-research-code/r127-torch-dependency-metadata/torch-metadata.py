import json,urllib.request,hashlib,email
name='torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl';want='787124e7db3b379d4f1ed54dd12ae7c741c16a4d29b49c0226a89bea50923ffb'; req=urllib.request.Request('https://pypi.org/simple/torch/',headers={'User-Agent':'VoltaFabric-dependency-admission','Accept':'application/vnd.pypi.simple.v1+json'})
with urllib.request.urlopen(req,timeout=15) as f:raw=f.read(25000001)
if len(raw)>25000000:raise ValueError('index cap')
a=[x for x in json.loads(raw)['files'] if x['filename']==name]
if len(a)!=1 or a[0]['hashes'].get('sha256')!=want:raise ValueError('artifact mismatch')
x=a[0]; mh=x.get('core-metadata') or x.get('dist-info-metadata'); expected=mh.get('sha256') if isinstance(mh,dict) else None
if not expected:raise ValueError('signed hash missing for metadata')
with urllib.request.urlopen(urllib.request.Request(x['url']+'.metadata',headers={'User-Agent':'VoltaFabric-dependency-admission'}),timeout=15) as f:b=f.read(2000001)
if len(b)>2000000 or hashlib.sha256(b).hexdigest()!=expected:raise ValueError('metadata hash mismatch')
m=email.message_from_bytes(b);print(json.dumps({'wheel':name,'wheel_sha256':want,'metadata_sha256':expected,'metadata_bytes':len(b),'name':m.get('Name'),'version':m.get('Version'),'requires_dist':m.get_all('Requires-Dist',[]),'scope':'exact wheel PEP 658 metadata, hash verified; no artifact download/import/GPU'}))
