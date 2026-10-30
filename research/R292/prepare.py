import json,hashlib,os,pathlib,subprocess
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');layer=int(__import__('sys').argv[1]);assert layer in (12,13,14,15);root=b/f'r292-layer{layer}-owners-r1';root.mkdir();binary=b/'r280-rust-x86-r1/target/debug/vf';template=b/'r284-layer4-owners-r1';rows={};source=b/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors';st=source.stat();identity=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
for rank in range(4):
 request=json.loads((template/f'rank{rank}.intent.json').read_text());request['source']['request']['spec']['layer']=layer;request['output']=str(root/f'rank{rank}.safetensors');(root/f'rank{rank}.intent.json').write_text(json.dumps(request,separators=(',',':'))+'\n')
 with (root/f'rank{rank}.stdout.json').open('xb') as out,(root/f'rank{rank}.stderr').open('xb') as err:
  p=subprocess.run([str(binary),'resident-export',str(root/f'rank{rank}.intent.json')],stdout=out,stderr=err,check=True)
 result=json.loads((root/f'rank{rank}.stdout.json').read_text());assert result['ok'] and not result['result']['cuda_executed'];path=pathlib.Path(request['output']);h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(1024*1024):h.update(block)
 rows[str(rank)]={'path':str(path),'sha256':h.hexdigest(),'experts':list(range(rank*64,(rank+1)*64))}
st=source.stat();assert identity==(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
manifest={'epoch':1,'layer':layer,'revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5','source_sha256':'07141c2db92e47bc08777132cd1a0323faf300eab3a7d7c111bc2bf075fda050','source_sha256_scope':'historical R284, not freshly rehashed','ranks':rows};(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'status':'PASS','layer':layer,'ranks':4,'source_stat_stable':True,'gpu':False}))
