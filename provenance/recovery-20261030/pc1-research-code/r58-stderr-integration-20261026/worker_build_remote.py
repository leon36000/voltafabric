"""Pinned actual Cargo worker gates; CPU only, isolated target, exact private ABI."""
from pathlib import Path
import json,os,hashlib,subprocess,time,signal,fcntl,shutil
ROOT=Path(__file__).resolve().parent
BASE=Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c')
RUST=BASE/'.tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mem():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
def main():
 plan=json.loads((ROOT/'BUILD_PLAN.json').read_text());assert all(sha(ROOT/n)==h for n,h in plan['files'].items())
 out=ROOT/'build';out.mkdir(exist_ok=False);(ROOT/'tmp').mkdir(exist_ok=False)
 lock=(ROOT/'compiler.lease').open('x+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 report=dict(status='CPU_WORKER_GATES_STARTED',pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),automatic_retry=False,gpu_executed=False,source_plan_sha256=sha(ROOT/'BUILD_PLAN.json'),steps=[])
 def save():(out/'operation.json').write_text(json.dumps(report,indent=2))
 env=dict(os.environ,PATH=str(RUST)+':'+os.environ['PATH'],RUSTC=str(RUST/'rustc'),CARGO_HOME=str(BASE/'.tools/cargo'),RUSTUP_HOME=str(BASE/'.tools/rustup'),CARGO_TARGET_DIR=str(ROOT/'target'),CARGO_BUILD_JOBS='1',CARGO_INCREMENTAL='0',CARGO_NET_OFFLINE='true',RUST_TEST_THREADS='1',TMPDIR=str(ROOT/'tmp'),VF_MXFP4_NATIVE_DIR=str(ROOT/'native'),CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=str(ROOT/'native'))
 for n in ['LD_PRELOAD','LD_AUDIT']:env.pop(n,None)
 def run(name,args):
  assert mem()>3*1024**3 and shutil.disk_usage(ROOT).free>4*1024**3
  argv=['/usr/bin/time','-v','-o',str(out/(name+'.time')),*args]
  with (out/(name+'.log')).open('x') as f:
   p=subprocess.Popen(argv,cwd=ROOT/'source',env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   step=dict(name=name,pid=p.pid,argv=argv,start_ns=time.monotonic_ns(),status='UNKNOWN');report['steps'].append(step);save()
   reason=None
   while p.poll() is None:
    if time.monotonic_ns()-step['start_ns']>360*10**9:reason='CPU_DEADLINE';break
    if mem()<1536*1024**2:reason='CPU_MEMORY_FLOOR';break
    time.sleep(.25)
   if reason:os.killpg(p.pid,signal.SIGKILL)
   code=p.wait(timeout=5)
   step.update(exit_code=code,status='TERMINAL',process_reconciled=True,end_ns=time.monotonic_ns(),resource_stop=reason);save()
   assert code==0 and reason is None,(name,code,reason)
 try:
  run('version',[str(RUST/'rustc'),'-vV'])
  c=[str(RUST/'cargo')];m=['--manifest-path','gpu-worker/Cargo.toml']
  run('fmt',c+['fmt',*m,'--','--check'])
  run('clippy',c+['clippy','--offline','--locked',*m,'--all-targets','--','-D','warnings'])
  for mode in ['debug','release']:
   flags=['--release'] if mode=='release' else []
   run('unit_'+mode,c+['test','--offline','--locked',*m,'--bin','vf-mxfp4-gpu-worker',*flags,'--','--nocapture'])
   run('stdio_'+mode,c+['test','--offline','--locked',*m,'--test','stdio',*flags,'--','--nocapture'])
   worker=ROOT/'target'/mode/'vf-mxfp4-gpu-worker';shutil.copyfile(worker,out/('worker_'+mode));(out/('worker_'+mode)).chmod(0o700)
   run('loader_'+mode,['ldd',str(worker)])
   assert 'not found' not in (out/('loader_'+mode+'.log')).read_text()
  shutil.copyfile(ROOT/'native/libvf_r13.so',out/'libvf_r13.so')
  assert all(sha(ROOT/n)==h for n,h in plan['files'].items())
  report.update(status='WORKER_CARGO_CPU_GATES_PASS_DEBUG_RELEASE',binaries={n:sha(out/n) for n in ['worker_debug','worker_release','libvf_r13.so']},source_stable=True)
 except Exception as e:report.update(status='WORKER_CARGO_CPU_GATES_FAILED',error=repr(e))
 finally:report['end_ns']=time.monotonic_ns();save();lock.close()
 print(json.dumps({'status':report['status'],'error':report.get('error')}))
 raise SystemExit(0 if report['status']=='WORKER_CARGO_CPU_GATES_PASS_DEBUG_RELEASE' else 1)
if __name__=='__main__':main()
