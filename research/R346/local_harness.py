"""PC1-only test of R327 execute with local process transport, no SSH/network."""
import os,sys,subprocess,json,pathlib,importlib.util,hashlib
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c');runs=base/'.runs';d=runs/'r346-local-harness-r1';box=d/'box';code=runs/'r315-cross-host-code-r1';path=':'.join(map(str,[code,runs/'r276-package-r1',runs,runs/'r299-code-r1']))
sys.path[:0]=[str(d/'code'),str(code),str(runs/'r276-package-r1'),str(runs),str(runs/'r299-code-r1')]
spec=importlib.util.spec_from_file_location('r327_crosshost_local',d/'crosshost.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.B=box;m.R=box/'.runs/repository';m.BIN=runs/'r325-correlated-errors-r1/target/debug/examples/r261_multipart_cpu_session';m.BIN_SHA=hashlib.sha256(m.BIN.read_bytes()).hexdigest();m.REMOTE=str(box/'.remote');m.REMOTE_CODE=str(code);m.REMOTE_ART=str(runs/'r333-distinct-correlated-r1/rank9.safetensors')
def local_prepare(script):
 p=subprocess.run(['/usr/bin/python3','-'],input=script,text=True,capture_output=True,timeout=30)
 if p.returncode:raise RuntimeError('local preparation failed: '+p.stderr[-300:])
 return json.loads(p.stdout)
m.ssh_python=local_prepare
import adapters.session_bridge as bridge
original=bridge.DurableSession
def local_transport(binary,p,ledger,deadline_seconds=10,**kwargs):
 p['workers'][0]['argv'][1]='PYTHONPATH='+path
 remote=p['workers'][1]['argv']; receiver=remote[-1]
 p['workers'][1]['argv']=['/usr/bin/env','PYTHONPATH='+path,sys.executable,'-B',str(code/'worker.py'),receiver]
 return original(binary,p,ledger,deadline_seconds,**kwargs)
bridge.DurableSession=local_transport
snapshot=m.POLICY.check(box)
result=m.admit_both(lambda:m.POLICY.check(box),lambda:m.POLICY.check(box),m.execute)
assert result['status']=='PASS' and len(result['batches'])==1 and len(result['close'])==2
out={'status':'PASS','scope':'single host; local subprocess substituted for SSH','run_id':result['run_id'],'batches':result['batches'],'ranks':[x['rank'] for x in result['close']],'admission':snapshot}
(d/'receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
