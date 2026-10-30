import json,hashlib,os,pathlib,subprocess,sys
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');layer=int(sys.argv[1]);assert layer in (16,17,18,19);root=b/f'r294-layer{layer}-owners-r1';root.mkdir();binary=b/'r280-rust-x86-r1/target/debug/vf';template=b/'r284-layer4-owners-r1';model=b/'r188-qwen-checkpoint-pc1-20261028-r1/model';index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];rows={};stamps={}
def hfile(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while block:=f.read(1024*1024):h.update(block)
 return h.hexdigest()
def sig(p):
 s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def invoke(command,intent,name):
 ip=root/f'{name}.intent.json';ip.write_text(json.dumps(intent,separators=(',',':'))+'\n')
 with (root/f'{name}.stdout.json').open('xb') as out,(root/f'{name}.stderr').open('xb') as err:subprocess.run([str(binary),command,str(ip)],stdout=out,stderr=err,check=True)
 result=json.loads((root/f'{name}.stdout.json').read_text());assert result['ok'] and not result['result']['cuda_executed']
for rank in range(4):
 original=json.loads((template/f'rank{rank}.intent.json').read_text());groups=[]
 for e in range(rank*64,(rank+1)*64):
  pre=f'model.language_model.layers.{layer}.mlp.experts.{e}.'; shards={v for k,v in index.items() if k.startswith(pre)};assert len(shards)==1,(layer,e,shards);shard=shards.pop()
  if not groups or groups[-1][0]!=shard:groups.append((shard,[]))
  groups[-1][1].append(e)
 fragments=[]
 for group,(shard,experts) in enumerate(groups):
  source=model/shard;stamps.setdefault(shard,sig(source));request=json.loads(json.dumps(original));q=request['source']['request'];q['path']=str(source);q['spec']['layer']=layer;q['spec']['registry']['copies']=[v for v in q['spec']['registry']['copies'] if v['expert'] in experts];q['spec']['experts']=[v for v in q['spec']['experts'] if v['id'] in experts];q['spec']['resident_order']=[v['local'] for v in q['spec']['experts']];out=root/(f'rank{rank}.safetensors' if len(groups)==1 else f'rank{rank}-part{group}.safetensors');request['output']=str(out);invoke('resident-export',request,f'rank{rank}-part{group}');fragments.append({'path':str(out),'expected_sha256':hfile(out)})
 final=root/f'rank{rank}.safetensors'
 if len(fragments)>1:invoke('resident-artifact-merge',{'artifacts':fragments,'output':str(final)},f'rank{rank}-merge')
 rows[str(rank)]={'path':str(final),'sha256':hfile(final),'experts':list(range(rank*64,(rank+1)*64)),'shards':[g[0] for g in groups],'fragments':fragments}
assert all(sig(model/s)==v for s,v in stamps.items());m={'epoch':1,'layer':layer,'revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5','source_hash_scope':'R188 historical hashes, file stats stable during conversion','ranks':rows};(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'status':'PASS','layer':layer,'ranks':4,'groups':[len(rows[str(i)]['fragments']) for i in range(4)],'gpu':False}))
