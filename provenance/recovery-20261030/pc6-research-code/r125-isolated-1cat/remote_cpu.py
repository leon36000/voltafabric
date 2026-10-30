import hashlib,json,os,subprocess,sys,urllib.request,shutil
from pathlib import Path
root=Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r125-isolated-1cat-20261028'); py=Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.tools/uv_python_install_dir/cpython-3.12-linux-x86_64-gnu/bin/python3.12');sha='8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6'; size=245787536;name='1cat_vllm-1.2.2-cp312-cp312-linux_x86_64.whl';url='https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.2.2/'+name
if shutil.disk_usage(root.parent).free < 10*1024**3: raise RuntimeError('less than 10 GiB available')
root.mkdir(mode=0o700);(root/'tmp').mkdir();(root/'cache').mkdir();out={'status':'STARTED_NO_RETRY','gpu':False,'dependencies_installed':False,'patches_applied':False,'steps':[]}
def save():
 p=root/'receipt.json';p.write_text(json.dumps(out,indent=2)+'\n')
def run(label,args,allowed=(0,),timeout=120):
 env=dict(os.environ,TMPDIR=str(root/'tmp'),PIP_CACHE_DIR=str(root/'cache'),PIP_NO_INDEX='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout);(root/(label+'.out')).write_bytes(r.stdout);(root/(label+'.err')).write_bytes(r.stderr);out['steps'].append({'label':label,'exit':r.returncode,'stdout_sha256':hashlib.sha256(r.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr).hexdigest()});save()
 if r.returncode not in allowed:raise RuntimeError(label+' failed')
try:
 save();run('venv',[str(py),'-m','venv',str(root/'venv')]);part=root/(name+'.part');h=hashlib.sha256();n=0
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'VoltaFabric-admission'}),timeout=30) as r,part.open('xb') as f:
  while True:
   chunk=r.read(1048576)
   if not chunk:break
   n+=len(chunk)
   if n>size:raise ValueError('size cap')
   h.update(chunk);f.write(chunk)
  f.flush();os.fsync(f.fileno())
 if n!=size or h.hexdigest()!=sha:raise ValueError('wheel digest or size mismatch')
 wheel=root/name;os.link(part,wheel);part.unlink();out['wheel_sha256']=sha;out['wheel_bytes']=n;save()
 vpy=str(root/'venv/bin/python');run('wheel_install',[vpy,'-m','pip','install','--no-deps','--no-index',str(wheel)]);run('package_version',[vpy,'-c',"import importlib.metadata as m;print(m.version('1cat-vllm'))"]);run('pip_check',[vpy,'-m','pip','check'],allowed=(1,))
 out['status']='BASE_WHEEL_INSTALLED_NO_DEPS_UNQUALIFIED';save();print(json.dumps(out))
except Exception as e:
 out['status']='FAILED_OR_INCOMPLETE_NO_RETRY';out['error_class']=type(e).__name__;save();print(json.dumps(out));sys.exit(1)
