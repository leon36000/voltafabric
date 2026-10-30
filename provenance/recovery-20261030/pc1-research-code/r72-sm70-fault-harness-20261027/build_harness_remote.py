import os,json,hashlib,fcntl,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1];OLD=BASE/'.runs/r69-sm70-current-worker-20261027'
RUST=BASE/'.tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin/rustc'
SHA='6127d41a0744792ae48fa56220f528f624d30be14fff641f9e1f7f4d015feea3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(n,args,env,expected=0,contains=None):
 p=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,timeout=60)
 (ROOT/(n+'.stdout')).write_bytes(p.stdout);(ROOT/(n+'.stderr')).write_bytes(p.stderr)
 row=dict(name=n,argv=args,exit_code=p.returncode,expected=expected,process_reconciled=True)
 result['steps'].append(row);save()
 assert p.returncode==expected,n
 if contains:assert contains in (p.stdout+p.stderr).decode(),n
 return p
result=dict(status='BUILD_UNKNOWN_NO_REPLAY',steps=[],pid=os.getpid(),no_gpu=True)
def save():(ROOT/'operation.json').write_text(json.dumps(result,indent=2))
assert not (ROOT/'operation.json').exists();save()
with (OLD/'compiler.lease').open('rb') as lease:
 fcntl.flock(lease,fcntl.LOCK_SH|fcntl.LOCK_NB)
 deps=OLD/'target/release/deps'
 def unique(p):
  xs=list(deps.glob(p));assert len(xs)==1,(p,xs);return xs[0]
 core=unique('libvoltafabric-*.rlib');sj=unique('libserde_json-*.rlib')
 plan=json.loads((OLD/'BUILD_PLAN.json').read_text());assert plan['canonical_manifest_sha256']=='41f179ce7d7076748e907e566182aa1d3298b4deb776cb52ecd0f694edba526d'
 before={str(p):sha(p) for p in [core,sj,RUST,ROOT/'hardware_r72.rs']}
 assert all(sha(OLD/n)==h for n,h in plan['files'].items())
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',VF_MXFP4_LIBRARY_SHA256=SHA,CARGO_MANIFEST_DIR=str(OLD/'source/gpu-worker'),CARGO_BIN_EXE_vf_mxfp4_gpu_worker='/NO_GPU_WORKER')
 env['CARGO_BIN_EXE_vf-mxfp4-gpu-worker']='/NO_GPU_WORKER'
 env['TMPDIR']=str(ROOT/'tmp');(ROOT/'tmp').mkdir()
 run('version',[str(RUST),'-vV'],env,contains='rustc 1.90.0')
 run('compile',[str(RUST),'--edition=2021','--test',str(ROOT/'hardware_r72.rs'),'--crate-name','hardware_r72','-L','dependency='+str(deps),'--extern','voltafabric='+str(core),'--extern','serde_json='+str(sj),'-C','opt-level=3','-C','lto=thin','-C','overflow-checks=yes','-D','warnings','-o',str(ROOT/'hardware_release')],env)
 run('list',[str(ROOT/'hardware_release'),'--list'],env,contains='6 tests')
 test='native_worker_rebind_preserves_loaded_owner_and_never_resends_weights'
 argv=[str(ROOT/'hardware_release'),'--ignored','--exact',test,'--nocapture']
 for n,ns,flag,msg in [('no-namespace',None,'1','VF_GPU_TEST_RUN_ID must supply'),('invalid-namespace','foreign','1','outside the declared'),('missing-gpu-flag','r53-r72-cpu-admission-only',None,'set VF_RUN_GPU_TESTS=1')]:
  e=dict(env,VF_GPU_WORKER='/NONEXISTENT_MUST_NEVER_SPAWN',VF_FIXTURE_DIR='/NONEXISTENT_MUST_NEVER_READ')
  e.pop('VF_GPU_TEST_RUN_ID',None);e.pop('VF_RUN_GPU_TESTS',None)
  if ns:e['VF_GPU_TEST_RUN_ID']=ns
  if flag:e['VF_RUN_GPU_TESTS']=flag
  run(n,argv,e,101,msg)
 assert all(sha(Path(p))==h for p,h in before.items())
 assert all(sha(OLD/n)==h for n,h in plan['files'].items())
 result.update(status='HARNESS_CPU_BUILD_AND_ADMISSION_PASS_NO_GPU',inputs=before,source_stable=True,binary_sha256=sha(ROOT/'hardware_release'))
 save()
