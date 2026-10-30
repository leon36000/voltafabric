import hashlib,json,os,resource,subprocess,sys
from pathlib import Path
base=Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');r125=base/'r125-isolated-1cat-20261028';root=base/'r128-resolver-dry-run-20261028';wheel=r125/'1cat_vllm-1.2.2-cp312-cp312-linux_x86_64.whl';py=r125/'venv/bin/python';expect='8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6'
with wheel.open('rb') as f:
 h=hashlib.sha256()
 while chunk:=f.read(1048576):h.update(chunk)
if h.hexdigest()!=expect:raise ValueError('wheel mismatch')
root.mkdir(mode=0o700);(root/'tmp').mkdir();(root/'cache').mkdir()
a=[str(py),'-m','pip','install','--dry-run','--ignore-installed','--only-binary=:all:','--no-cache-dir','--timeout','8','--retries','0','--index-url','https://pypi.org/simple','--report',str(root/'report.json'),str(wheel)]
env={'HOME':str(root),'PATH':'/usr/bin:/bin','TMPDIR':str(root/'tmp'),'PIP_CONFIG_FILE':'/dev/null','PIP_DISABLE_PIP_VERSION_CHECK':'1','PIP_NO_INPUT':'1','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}
intent={'argv':a,'wheel_sha256':expect,'write_limit_per_file_bytes':134217728,'timeout_seconds':110,'gpu':False,'install':False,'status':'STARTED'};(root/'intent.json').write_text(json.dumps(intent,indent=2)+'\n')
def limits():resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728))
try:
 with (root/'stdout.log').open('xb') as out,(root/'stderr.log').open('xb') as err:
  r=subprocess.run(a,env=env,stdout=out,stderr=err,timeout=110,preexec_fn=limits)
 status={'status':'TERMINAL','exit':r.returncode,'report_exists':(root/'report.json').exists()}
except subprocess.TimeoutExpired:status={'status':'TIMEOUT_NO_RETRY','report_exists':(root/'report.json').exists()}
(root/'terminal.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status))
