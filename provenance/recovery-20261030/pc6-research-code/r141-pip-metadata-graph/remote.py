from pathlib import Path
import hashlib,json,os,resource,subprocess,sys
b=Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=b/'r141-pip-metadata-graph-20261028';r125=b/'r125-isolated-1cat-20261028';wheel=r125/'1cat_vllm-1.2.2-cp312-cp312-linux_x86_64.whl';py=r125/'venv/bin/python';data=json.loads(sys.stdin.buffer.read(100000));code=bytes.fromhex(data['hook_hex']);assert hashlib.sha256(code).hexdigest()==data['hook_sha256'];h=hashlib.sha256()
with wheel.open('rb') as f:
 while c:=f.read(1048576):h.update(c)
if h.hexdigest()!='8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6':raise ValueError('wheel differs')
root.mkdir(mode=0o700);(root/'tmp').mkdir();(root/'cache').mkdir();(root/'hook.py').write_bytes(code);env={'HOME':str(root),'PATH':'/usr/bin:/bin','TMPDIR':str(root/'tmp'),'PIP_CONFIG_FILE':'/dev/null','PIP_DISABLE_PIP_VERSION_CHECK':'1','PIP_NO_INPUT':'1','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}
a=[str(py),'-B',str(root/'hook.py'),str(root),str(wheel)];(root/'intent.json').write_text(json.dumps({'argv':a,'hook_sha256':data['hook_sha256'],'max_file_bytes':134217728,'timeout':110,'gpu':False,'actual_install':False})+'\n')
def limits():resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728))
try:
 with (root/'stdout.log').open('xb') as out,(root/'stderr.log').open('xb') as err:r=subprocess.run(a,env=env,stdout=out,stderr=err,timeout=110,preexec_fn=limits)
 status={'status':'TERMINAL','exit':r.returncode,'candidate_file':(root/'candidates.json').is_file()}
except subprocess.TimeoutExpired:status={'status':'TIMEOUT_NO_RETRY','candidate_file':(root/'candidates.json').is_file()}
(root/'terminal.json').write_text(json.dumps(status)+'\n');print(json.dumps(status))
