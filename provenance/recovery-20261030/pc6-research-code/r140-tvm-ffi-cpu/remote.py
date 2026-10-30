from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,urllib.request
base=Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r140-tvm-ffi-cpu-20261028';py='/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.tools/uv_python_install_dir/cpython-3.12-linux-x86_64-gnu/bin/python3.12'
artifacts=[{'file':'apache_tvm_ffi-0.1.10-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl','sha256':'73d478395a8625dd92fde7b7fd92b4719f18f480b78336e422cb66cc7985213d','size':2589574,'url':'https://files.pythonhosted.org/packages/50/6f/23740f06829030704e6f8f1f7093a06b7a68f904baa40053a5f594705bae/apache_tvm_ffi-0.1.10-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'},{'file':'typing_extensions-4.15.0-py3-none-any.whl','sha256':'f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548','size':44614,'url':'https://files.pythonhosted.org/packages/18/67/36e9267722cc04a6b9f15c7f3441c2363321a3ea07da7ae0c0707beb2a9c/typing_extensions-4.15.0-py3-none-any.whl'}]
if shutil.disk_usage(base).free<5*1024**3:raise ValueError('disk reserve')
root.mkdir(mode=0o700);(root/'tmp').mkdir();out={'status':'STARTED_NO_RETRY','gpu':False,'steps':[],'artifacts':[],'scope':'isolated TVM FFI CPU import only; not TileLang, 1Cat or GPU admission'}
def save():
 with (root/'receipt.json').open('w') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(name,argv,allowed=(0,)):
 env={'HOME':str(root),'PATH':'/usr/bin:/bin','TMPDIR':str(root/'tmp'),'PIP_CONFIG_FILE':'/dev/null','PIP_NO_INDEX':'1','PIP_NO_INPUT':'1','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'};r=subprocess.run(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45);(root/(name+'.out')).write_bytes(r.stdout);(root/(name+'.err')).write_bytes(r.stderr);out['steps'].append({'name':name,'exit':r.returncode,'stdout_sha256':hashlib.sha256(r.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr).hexdigest()});save()
 if r.returncode not in allowed:raise RuntimeError(name+' failed')
try:
 save();run('venv',[py,'-m','venv',str(root/'venv')])
 for a in artifacts:
  req=urllib.request.Request(a['url'],headers={'User-Agent':'VoltaFabric-admission'});h=hashlib.sha256();n=0;dest=root/a['file']
  with urllib.request.urlopen(req,timeout=20) as f,dest.open('xb') as w:
   while chunk:=f.read(262144):
    n+=len(chunk)
    if n>a['size']:raise ValueError('download cap exceeded')
    h.update(chunk);w.write(chunk)
   w.flush();os.fsync(w.fileno())
  if n!=a['size'] or h.hexdigest()!=a['sha256']:raise ValueError('artifact mismatch')
  out['artifacts'].append({'file':a['file'],'bytes':n,'sha256':h.hexdigest()});save()
 v=str(root/'venv/bin/python');run('install',[v,'-m','pip','install','--no-index','--no-deps']+[str(root/a['file']) for a in artifacts]);run('check',[v,'-m','pip','check']);run('import',[v,'-c',"import importlib.metadata as m,tvm_ffi,json;print(json.dumps({'distribution':m.version('apache-tvm-ffi'),'module_file':tvm_ffi.__file__}))"])
 out['status']='TVM_FFI_010_CPU_IMPORT_IN_ISOLATION_PASSED';save();print(json.dumps(out))
except Exception as e:out['status']='FAILED_OR_INCOMPLETE_NO_RETRY';out['error_class']=type(e).__name__;save();print(json.dumps(out));sys.exit(1)
