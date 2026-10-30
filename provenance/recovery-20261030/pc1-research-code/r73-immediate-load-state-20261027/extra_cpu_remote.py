"""After Cargo: CPU-only adversarial executor statements and private harness build."""
import json,os,subprocess,hashlib,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent;BASE=R.parents[1];T=BASE/'.tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin';out=R/'extra-cpu';out.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((R/'BUILD_PLAN.json').read_text());assert all(sha(R/n)==h for n,h in plan['files'].items());assert json.loads((R/'build/operation.json').read_text())['status']=='WORKER_CARGO_CPU_GATES_PASS_DEBUG_RELEASE'
report=dict(status='EXTRA_CPU_STARTED',pid=os.getpid(),steps=[],cases=[],GPU_executed=False)
def save():(out/'operation.json').write_text(json.dumps(report,indent=2))
def run(n,argv,env,expected=0,contains=None):
 p=subprocess.run(argv,cwd=R,env=env,capture_output=True,timeout=60);(out/(n+'.stdout')).write_bytes(p.stdout);(out/(n+'.stderr')).write_bytes(p.stderr)
 report['steps'].append(dict(name=n,argv=argv,exit_code=p.returncode,expected_exit_code=expected,process_reconciled=True));save();assert p.returncode==expected,(n,p.stderr[-2000:])
 if contains:assert contains in (p.stdout+p.stderr).decode()
 return p
save()
with (R/'compiler.lease').open('rb') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
 for mode in ['debug','release']:
  deps=R/'target'/mode/'deps'
  def unique(pattern):
   ps=list(deps.glob(pattern));assert len(ps)==1,(pattern,ps);return ps[0]
  core=unique('libvoltafabric-*.rlib');sj=unique('libserde_json-*.rlib');before={str(p):sha(p) for p in [core,sj]}
  common=['--edition=2021','-L','dependency='+str(deps),'--extern','voltafabric='+str(core),'--extern','serde_json='+str(sj),'-C','overflow-checks=yes','-C','opt-level='+('0' if mode=='debug' else '3')]
  if mode=='release':common+=['-C','lto=thin']
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(R/'tmp'))
  for label in ['baseline','candidate']:
   target=out/(label+'_'+mode)
   run('compile_'+label+'_'+mode,[str(T/'rustc'),str(R/('cpu_contract_'+label+'.rs')),*common,'-o',str(target)],env)
   if label=='baseline':
    e=dict(env,PREFIX='1',AUDIT='1',IO='healthy',CLOSE_FAIL='0');run('red_baseline_'+mode,[str(target)],e,101,'missing immediate refusal audit');continue
   for prefix in [0,1,2]:
    variants=[(True,'healthy',False),(True,'broken',False),(True,'partial',False),(False,'healthy',False),(False,'broken',False)]
    if prefix:variants.append((True,'healthy',True))
    for audit,io,fail in variants:
     name=f'cpu_{mode}_{prefix}_{int(audit)}_{io}_{int(fail)}';e=dict(env,PREFIX=str(prefix),AUDIT=str(int(audit)),IO=io,CLOSE_FAIL=str(int(fail)))
     p=run(name,[str(target)],e);row=json.loads(p.stdout)
     if audit and io=='healthy' and not fail:
      lines=[json.loads(x) for x in p.stderr.decode().splitlines() if x.startswith('{')];assert len(lines)==1;probe=lines[0];assert probe['before']['lifecycle']==[prefix,0,0] and probe['after']['lifecycle']==[0,0,0];assert probe['after']['published']==[] and probe['after']['binding'] is None
     report['cases'].append(dict(mode=mode,result=row))
  assert all(sha(Path(p))==h for p,h in before.items())
 # Compile actual private Rust harness against the newly built, source-linked release core.
 deps=R/'target/release/deps';core=list(deps.glob('libvoltafabric-*.rlib'))[0];sj=list(deps.glob('libserde_json-*.rlib'))[0]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(R/'tmp'),VF_MXFP4_LIBRARY_SHA256=plan['native_sha256'],CARGO_MANIFEST_DIR=str(R/'source/gpu-worker'))
 env['CARGO_BIN_EXE_vf-mxfp4-gpu-worker']='/NO_GPU_EXECUTABLE'
 args=[str(T/'rustc'),'--edition=2021','--test',str(R/'hardware_r73.rs'),'--crate-name','hardware_r73','-L','dependency='+str(deps),'--extern','voltafabric='+str(core),'--extern','serde_json='+str(sj),'-C','opt-level=3','-C','lto=thin','-C','overflow-checks=yes','-D','warnings','-o',str(out/'hardware_release')]
 run('harness_compile',args,env)
 run('harness_list',[str(out/'hardware_release'),'--list'],env,contains='6 tests')
 argv=[str(out/'hardware_release'),'--ignored','--exact','native_load_numeric_refusal_rolls_back_each_partial_owner_prefix','--nocapture']
 for n,ns,flag,decl,msg in [('missing_namespace',None,'1',None,'VF_GPU_TEST_RUN_ID must supply'),('no_gpu_flag','r53-r73-cpu-only',None,None,'set VF_RUN_GPU_TESTS=1'),('missing_hashes','r53-r73-cpu-only','1',None,'predeclared numeric LOAD hashes required')]:
  e=dict(env,VF_FIXTURE_DIR=str(R/'source/fixtures'),VF_GPU_WORKER='/NO_NATIVE_OR_CUDA_EXECUTABLE');e.pop('VF_GPU_TEST_RUN_ID',None);e.pop('VF_RUN_GPU_TESTS',None);e.pop('VF_R72_ROLLBACK_SHA256',None)
  if ns:e['VF_GPU_TEST_RUN_ID']=ns
  if flag:e['VF_RUN_GPU_TESTS']=flag
  run(n,argv,e,101,msg)
 assert all(sha(R/n)==h for n,h in plan['files'].items())
 report.update(status='CPU_EXACT_PROBE_STATEMENTS_DEBUG_RELEASE_AND_HARNESS_PASS',source_stable=True,cpu_cases=len(report['cases']),harness_sha256=sha(out/'hardware_release'));save();print(json.dumps(dict(status=report['status'],cpu_cases=report['cpu_cases'],harness_sha256=report['harness_sha256'])))
