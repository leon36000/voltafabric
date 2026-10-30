import json,hashlib,os,subprocess,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent;BASE=R.parents[1];T=BASE/'.tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin';out=R/'live-harness';out.mkdir()
plan=json.loads((R/'LIVE_HARNESS_PLAN.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert all(sha(R/n)==h for n,h in plan['files'].items())
report=dict(status='CPU_LIVE_READER_STARTED',pid=os.getpid(),steps=[],cases=[],gpu_executed=False)
def save():(out/'operation.json').write_text(json.dumps(report,indent=2))
def run(n,argv,env,expected=0):
 p=subprocess.run(argv,capture_output=True,env=env,cwd=R,timeout=60);(out/(n+'.stdout')).write_bytes(p.stdout);(out/(n+'.stderr')).write_bytes(p.stderr);report['steps'].append(dict(name=n,argv=argv,exit_code=p.returncode,process_reconciled=True));save();assert p.returncode==expected,(n,p.stderr[-3000:]);return p
save()
with (R/'compiler.lease').open('rb') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
 for mode in ['debug','release']:
  deps=R/'target'/mode/'deps';core=list(deps.glob('libvoltafabric-*.rlib'));sj=list(deps.glob('libserde_json-*.rlib'));assert len(core)==len(sj)==1
  flags=['--edition=2021','-L','dependency='+str(deps),'--extern','voltafabric='+str(core[0]),'--extern','serde_json='+str(sj[0]),'-C','overflow-checks=yes','-C','opt-level='+('0' if mode=='debug' else '3')]
  if mode=='release':flags+=['-C','lto=thin']
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(R/'tmp'))
  exe=out/('cpu_reader_'+mode);run('reader_compile_'+mode,[str(T/'rustc'),str(R/'cpu_live_reader.rs'),*flags,'-o',str(exe)],env)
  for case in ['healthy','dirty','foreign','missing','partial','badjson','cap']:
   p=run('reader_'+mode+'_'+case,[str(exe),case],env);report['cases'].append(dict(mode=mode,result=json.loads(p.stdout.decode().splitlines()[-1])))
  if mode=='release':
   env.update(VF_MXFP4_LIBRARY_SHA256='6127d41a0744792ae48fa56220f528f624d30be14fff641f9e1f7f4d015feea3',CARGO_MANIFEST_DIR=str(R/'source/gpu-worker'));env['CARGO_BIN_EXE_vf-mxfp4-gpu-worker']='/NO_NATIVE_EXECUTABLE'
   run('harness_compile',[str(T/'rustc'),'--test',str(R/'hardware_r73_live.rs'),*flags,'-D','warnings','-o',str(out/'hardware_release')],env)
   run('harness_list',[str(out/'hardware_release'),'--list'],env)
 assert all(sha(R/n)==h for n,h in plan['files'].items())
 report.update(status='CPU_LIVE_PROBE_ADMISSION_BEFORE_REPAIR_PASS',source_stable=True,cpu_cases=len(report['cases']),harness_sha256=sha(out/'hardware_release'));save();print(json.dumps(dict(status=report['status'],cpu_cases=report['cpu_cases'],harness_sha256=report['harness_sha256'])))
