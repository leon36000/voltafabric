import ast,hashlib,json,pathlib,zipfile
p=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r125-isolated-1cat-20261028/1cat_vllm-1.2.2-cp312-cp312-linux_x86_64.whl')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''): h.update(b)
assert h.hexdigest()=='8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6'
wanted=['vllm/model_executor/layers/fused_moe/moe_align_block_size.py','vllm/model_executor/layers/fused_moe/runner/shared_experts.py']
results=[]; tile=[];count=0;total=0
with zipfile.ZipFile(p) as z:
 for n in wanted:
  i=z.getinfo(n); assert i.file_size<1048576; data=z.read(i);t=ast.parse(data);imports=[]
  for x in ast.walk(t):
   if isinstance(x,ast.Import):imports.extend(y.name for y in x.names)
   elif isinstance(x,ast.ImportFrom):imports.append('.'*x.level+(x.module or ''))
  results.append({'path':n,'sha256':hashlib.sha256(data).hexdigest(),'imports':sorted(set(imports)),'direct_tilelang':any('tilelang' in s.lower() for s in imports)})
 for i in z.infolist():
  if not i.filename.startswith('vllm/') or not i.filename.endswith('.py'):continue
  count+=1;total+=i.file_size
  if total>64*1024*1024:raise ValueError('python source scan cap')
  if i.file_size>1048576:raise ValueError('single source cap')
  data=z.read(i)
  if b'tilelang' in data.lower():
   lines=[{'line':j,'text':line.strip()[:200]} for j,line in enumerate(data.decode('utf-8').splitlines(),1) if 'tilelang' in line.lower()]
   tile.append({'path':i.filename,'sha256':hashlib.sha256(data).hexdigest(),'hits':lines[:30]})
print(json.dumps({'wheel_sha256':h.hexdigest(),'targets':results,'tilelang_source_hits':tile,'python_files_scanned':count,'python_bytes_scanned':total,'scope':'static zipped Python sources only, no imports, no extensions, no GPU'},indent=2,sort_keys=True))
