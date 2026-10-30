import ast,hashlib,json,pathlib
p=pathlib.Path(__file__).parent; result=json.loads((p/'result.json').read_text()); assert result['python_files_scanned']==1876
src=(p/'gdn.py').read_bytes(); row=next(x for x in result['tilelang_source_hits'] if x['path'].endswith('/qwen_gdn_linear_attn.py'));assert hashlib.sha256(src).hexdigest()==row['sha256']; tree=ast.parse(src)
f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_sm70_flashqla_original_prefill_enabled'); assert any(isinstance(x,ast.Return) and isinstance(x.value,ast.Constant) and x.value.value is True for x in ast.walk(f))
names=[x for x in ast.walk(tree) if isinstance(x,ast.ImportFrom) and x.module=='flash_qla.ops.gated_delta_rule.chunk' and any(a.name=='chunk_gated_delta_rule_fwd_sm70_tilelang' for a in x.names)];assert len(names)==1
pinned=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/upstream-reuse-integration-r118-20261028/pinned-upstream'); donors=[]
for n in ['fork_patches__skinny_moe.py','fork_patches__marlin.py']:
 raw=(pinned/n).read_bytes();t=ast.parse(raw);imports=[]
 for x in ast.walk(t):
  if isinstance(x,ast.Import):imports.extend(y.name for y in x.names)
  if isinstance(x,ast.ImportFrom):imports.append(x.module or '')
 assert not any('tilelang' in x.lower() for x in imports)
 donors.append({'name':n,'sha256':hashlib.sha256(raw).hexdigest(),'direct_tilelang_imports':0})
assert all(not x['direct_tilelang'] for x in result['targets'])
print(json.dumps({'status':'STATIC_CALL_BOUNDARIES_IDENTIFIED','donors':donors,'gdn_sha256':row['sha256'],'gdn_default_enabled_line':f.lineno,'gdn_lazy_tilelang_import_line':names[0].lineno,'limits':'Direct static imports only; transitive, dynamic, compiled code and whether a selected model executes this path remain unproven.'},indent=2,sort_keys=True))
