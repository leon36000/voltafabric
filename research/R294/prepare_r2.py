import json,hashlib,pathlib,subprocess,sys
b=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');layer=int(sys.argv[1]);assert layer in (16,17,18,19);root=b/f'r294-layer{layer}-owners-r2';root.mkdir();binary=b/'r280-rust-x86-r1/target/debug/vf';template=b/'r284-layer4-owners-r1';ex=b/'r294-extracted-second-r1';source=ex/f'layer{layer}.safetensors';receipt=json.loads((ex/f'layer{layer}.json').read_text());rows={}
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
before=sig(source);assert hfile(source)==receipt['selected_sha256']
for rank in range(4):
 if layer==16 and rank<2:
  final=b/'r294-layer16-owners-r1'/f'rank{rank}.safetensors';assert final.is_file();fragments=[]
 else:
  original=json.loads((template/f'rank{rank}.intent.json').read_text());q=original['source']['request'];q['path']=str(source);q['spec']['layer']=layer
  if layer==16 and rank==2:
   ids=set(range(139,192));q['spec']['registry']['copies']=[v for v in q['spec']['registry']['copies'] if v['expert'] in ids];q['spec']['experts']=[v for v in q['spec']['experts'] if v['id'] in ids];q['spec']['resident_order']=[v['local'] for v in q['spec']['experts']];part=root/'rank2-part1.safetensors';original['output']=str(part);invoke('resident-export',original,'rank2-part1');first=b/'r294-layer16-owners-r1'/'rank2-part0.safetensors';fragments=[{'path':str(first),'expected_sha256':hfile(first)},{'path':str(part),'expected_sha256':hfile(part)}];final=root/'rank2.safetensors';invoke('resident-artifact-merge',{'artifacts':fragments,'output':str(final)},'rank2-merge')
  else:
   final=root/f'rank{rank}.safetensors';original['output']=str(final);invoke('resident-export',original,f'rank{rank}');fragments=[]
 rows[str(rank)]={'path':str(final),'sha256':hfile(final),'experts':list(range(rank*64,(rank+1)*64)),'fragments':fragments}
assert sig(source)==before;m={'epoch':1,'layer':layer,'revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5','selected_sha256':receipt['selected_sha256'],'ranks':rows};(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'status':'PASS','layer':layer,'ranks':4,'merged_rank2':layer==16,'gpu':False}),flush=True)
