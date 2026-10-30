"""One fixed cost comparison only after actual CPU gates; never a GPU job."""
import fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;BASE=R.parents[1];CACHE=BASE/'.runs/r73-immediate-load-state-20261027'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=json.loads((R/'INPUTS.json').read_text());assert all(sha(R/n)==h for n,h in P['files'].items())
g=json.loads((R/'out/gates-final-operation.json').read_text());assert g['status']=='R75_EXISTING_CPU_PRODUCER_GATES_PASS_NO_BENCHMARK' and g['unit_tests']==26 and g['compiled_rust_mutants_detected']==2
assert all(s['exit_code']==s['expected_exit_code'] and s['status']=='TERMINAL' for s in g['steps'])
assert all(sha(R/'out'/n)==h for n,h in g['binaries'].items())
assert all(sha(Path(p))==h for group in g['rlibs'].values() for p,h in group.items())
receipt=R/'out/bench-operation.json'
report=dict(status='UNKNOWN_NO_RETRY',pid=os.getpid(),source_plan_sha256=sha(R/'INPUTS.json'),gates_sha256=sha(R/'out/gates-final-operation.json'),automatic_retry=False,gpu=False,selected_cpu=g['selected_cpu'])
with receipt.open('x') as f:json.dump(report,f)
assert report['selected_cpu'] in os.sched_getaffinity(0)
argv=['/usr/bin/time','-v','-o',str(R/'out/cost.time'),str(R/'out/probe_release'),'bench',str(R/'CORPUS.json')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(R/'tmp'));env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
with (CACHE/'compiler.lease').open('rb') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
 p=subprocess.Popen(argv,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{report['selected_cpu']}));report.update(argv=argv,child_pid=p.pid,start_ns=time.monotonic_ns());receipt.write_text(json.dumps(report,indent=2))
 try:a,b=p.communicate(timeout=30)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGKILL);a,b=p.communicate(timeout=3);report['timeout']=True
 (R/'out/cost.stdout').write_bytes(a);(R/'out/cost.stderr').write_bytes(b)
 report.update(status='PROCESS_TERMINAL',exit_code=p.returncode,process_reconciled=p.poll() is not None,end_ns=time.monotonic_ns(),source_stable=all(sha(R/n)==h for n,h in P['files'].items()),binaries_stable=all(sha(R/'out'/n)==h for n,h in g['binaries'].items()),rlibs_stable=all(sha(Path(p))==h for group in g['rlibs'].values() for p,h in group.items()))
 receipt.write_text(json.dumps(report,indent=2)+'\n');assert p.returncode==0 and not report.get('timeout') and report['source_stable'] and report['binaries_stable'] and report['rlibs_stable'],(report,b[-2000:]);print(json.dumps(dict(status=report['status'],exit_code=p.returncode,gpu=False)))
