"""Exercise a real local worker's startup guard and empty session; no experts."""
import hashlib,json,subprocess
from pathlib import Path
worker=Path('/home/pc1/NAOS/vf-r525-work/target/release/vf-rank-session')
digest=hashlib.sha256(worker.read_bytes()).hexdigest()
frames=b'{"op":"hello","protocol":1,"rank":2}\n{"op":"shutdown"}\n'
def run(expected):
    return subprocess.run([str(worker),'--guarded-local','--expected-exe-sha256',expected],input=frames,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15,check=False)
good=run(digest)
assert good.returncode==0 and good.stderr==b'', (good.returncode,good.stderr)
responses=[json.loads(x) for x in good.stdout.splitlines()]
assert len(responses)==2 and responses[0]['reply']=='hello' and responses[0]['hello']['rank']==2
assert responses[1]['reply']=='bye' and responses[1]['execs']==0, responses
bad=run('0'*64)
assert bad.returncode==2 and bad.stdout==b'' and b'LOCAL_EXECUTABLE_REFUSED' in bad.stderr
remote=subprocess.run([str(worker),'--guarded-remote','--expected-exe-sha256','0'*64],input=frames,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15,check=False)
assert remote.returncode==2 and remote.stdout==b'' and b'REMOTE_EXECUTABLE_REFUSED' in remote.stderr
print(json.dumps({'status':'PASS','worker_sha256':digest,'hello_rank':2,'bye_execs':0,'incorrect_hash_exit':bad.returncode}))
