"""Private CPU compile/verify/cost runner. Never loads CUDA or runs a worker."""
import fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;BASE=R.parents[1]
CACHE=BASE/'.runs/r73-immediate-load-state-20261027'
T=BASE/'.tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=json.loads((R/'INPUTS.json').read_text());assert all(sha(R/n)==h for n,h in P['files'].items())
assert sha(CACHE/'BUILD_PLAN.json')==P['cache_build_plan_sha256']
assert sha(CACHE/'build/operation.json')==P['cache_operation_sha256']
old=json.loads((CACHE/'BUILD_PLAN.json').read_text());assert old['canonical_manifest_sha256']==P['canonical_manifest_sha256']
assert all(sha(CACHE/n)==h for n,h in old['files'].items())
for n,h in old['canonical_source_hashes'].items():
 if not n.startswith('gpu-worker/'):
  assert sha(CACHE/'source'/n)==h
assert json.loads((CACHE/'build/operation.json').read_text())['status']=='WORKER_CARGO_CPU_GATES_PASS_DEBUG_RELEASE'
(R/'out').mkdir();out=R/'out'
report=dict(status='CPU_STARTED_NO_GPU',pid=os.getpid(),steps=[],source_plan_sha256=sha(R/'INPUTS.json'),automatic_retry=False,gpu=False,host=os.uname().nodename,arch=os.uname().machine,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),selected_cpu=min(os.sched_getaffinity(0)),rlibs={})
def save():(out/'operation.json').write_text(json.dumps(report,indent=2)+'\n')
def run(name,argv,timeout=30):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(R/'tmp'));env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
 before=time.monotonic_ns();p=subprocess.Popen(argv,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{report['selected_cpu']}))
 report['steps'].append(dict(name=name,argv=argv,pid=p.pid,status='UNKNOWN',automatic_retry=False,start_ns=before));save()
 try:a,b=p.communicate(timeout=timeout)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGKILL);a,b=p.communicate(timeout=3);report['steps'][-1]['timeout']=True
 (out/(name+'.stdout')).write_bytes(a);(out/(name+'.stderr')).write_bytes(b)
 report['steps'][-1].update(exit_code=p.returncode,status='TERMINAL',process_reconciled=p.poll() is not None,end_ns=time.monotonic_ns());save();assert p.returncode==0,(name,b[-2000:])
 return a
save()
with (CACHE/'compiler.lease').open('rb') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
 run('version',[str(T/'rustc'),'-vV'])
 run('fmt',[str(T/'rustfmt'),'--edition','2021','--check',str(R/'quantizer_probe.rs')])
 for mode in ['debug','release']:
  deps=CACHE/'target'/mode/'deps'
  def unique(pattern):
   ps=list(deps.glob(pattern));assert len(ps)==1,(pattern,[p.name for p in ps]);return ps[0]
  libs={n:unique('lib'+n+'-*.rlib') for n in ['voltafabric','serde_json','libc']}
  report['rlibs'][mode]={str(p):sha(p) for p in libs.values()};save()
  args=['--edition=2021','-L','dependency='+str(deps),'-C','overflow-checks=yes','-C','opt-level='+('0' if mode=='debug' else '3'),'-D','warnings']
  for n,p in libs.items():args+=['--extern',n+'='+str(p)]
  if mode=='release':args+=['-C','lto=thin']
  binary=out/('probe_'+mode)
  run('compile_'+mode,[str(T/'rustc'),str(R/'quantizer_probe.rs'),*args,'-o',str(binary)],60)
  if mode=='release':
   run('clippy',[str(T/'clippy-driver'),str(R/'quantizer_probe.rs'),*args,'--emit=metadata','-o',str(out/'clippy.rmeta')],60)
  run('verify_'+mode,[str(binary),'verify',str(R/'oracle/CORPUS.json')])
 # Exactly ONE measured benchmark invocation, no retries or sample replacement.
 run('release_cost',['/usr/bin/time','-v','-o',str(out/'release_cost.time'),str(out/'probe_release'),'bench',str(R/'oracle/CORPUS.json')])
 assert all(sha(Path(p))==h for mode in report['rlibs'].values() for p,h in mode.items())
 assert all(sha(R/n)==h for n,h in P['files'].items())
 assert all(sha(CACHE/n)==h for n,h in old['files'].items())
 report.update(status='CPU_DEBUG_RELEASE_EXACT_AND_SINGLE_COST_SERIES_PASS',source_stable=True,rlibs_stable=True,source_files=P['files'],binaries={p.name:sha(p) for p in [out/'probe_debug',out/'probe_release']});save()
 print(json.dumps(dict(status=report['status'],steps=len(report['steps']),selected_cpu=report['selected_cpu'],gpu=False)))
