"""Build the existing CPU producer unchanged, exact corpus/units and real mutants."""
import fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;BASE=R.parents[1];CACHE=BASE/'.runs/r73-immediate-load-state-20261027';T=BASE/'.tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=json.loads((R/'INPUTS.json').read_text());assert all(sha(R/n)==h for n,h in P['files'].items())
assert sha(CACHE/'BUILD_PLAN.json')==P['cache_build_plan_sha256'] and sha(CACHE/'build/operation.json')==P['cache_operation_sha256']
old=json.loads((CACHE/'BUILD_PLAN.json').read_text());assert old['canonical_manifest_sha256']==P['canonical_manifest_sha256'] and all(sha(CACHE/n)==h for n,h in old['files'].items())
out=R/'out';assert out.is_dir()
initial=json.loads((out/'gates-resume-operation.json').read_text());assert sha(out/'gates-resume-operation.json')=='952d82b605c87dfae2c359eebaac005ddf751012bebc16ff3110cb883b247e7d'
assert len(initial['steps'])==9 and initial['steps'][-1]['name']=='probe_build_debug' and initial['steps'][-1]['exit_code']==1 and all(s['exit_code']==0 and s['status']=='TERMINAL' for s in initial['steps'][:-1])
assert not Path('/proc',str(initial['pid'])).exists() and all(not Path('/proc',str(s['pid'])).exists() for s in initial['steps'])
assert all(sha(out/n)==h for n,h in {'libcodec_debug.rlib': 'a78ffe0cac0792852cc1ececfb39b4de65ee1ccd9d7a77dfa50a5a85259be9b4', 'codec_tests_debug': '5a4b1be30c2adb977409f0ef5304c0d8a2d61595f0b6d99464557a19ef0a7c8f', 'libproducer_debug.rlib': '617222241c8c581939e29ae020709d768b70019cc26540827456760973cec630', 'producer_tests_debug': '0e9388b8b109bcdf2962bd24ea51ad482da8318042db6f79a8cd1f59e8ea29fc'}.items())
assert all(sha(Path(p))==h for group in initial['rlibs'].values() for p,h in group.items())
report=dict(initial,status='CPU_GATES_CONTINUE_AFTER_PROVEN_RLIB_NAMING_FIX',pid=os.getpid(),initial_operation_sha256='952d82b605c87dfae2c359eebaac005ddf751012bebc16ff3110cb883b247e7d',reused_successful_steps=[s['name'] for s in initial['steps'][:-1]],source_plan_sha256=sha(R/'INPUTS.json'),known_compiler_failure=initial['steps'][-1])
report['steps']=[dict(s,reused_from_prior_terminal_receipt=True) for s in initial['steps'][:-1]]
report['initial_pid_absent']=True

def save():(out/'gates-final-operation.json').write_text(json.dumps(report,indent=2)+'\n')
def run(n,argv,expected=0,contains=None,timeout=30):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(R/'tmp'));env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
 p=subprocess.Popen(argv,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{report['selected_cpu']}))
 report['steps'].append(dict(name=n,argv=argv,pid=p.pid,status='UNKNOWN',automatic_retry=False,start_ns=time.monotonic_ns(),expected_exit_code=expected));save()
 try:a,b=p.communicate(timeout=timeout)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGKILL);a,b=p.communicate(timeout=3);report['steps'][-1]['timeout']=True
 (out/(n+'.stdout')).write_bytes(a);(out/(n+'.stderr')).write_bytes(b)
 report['steps'][-1].update(status='TERMINAL',exit_code=p.returncode,process_reconciled=p.poll() is not None,end_ns=time.monotonic_ns());save();assert p.returncode==expected,(n,b[-2000:])
 if contains:assert contains.encode() in a+b,(n,a[-1000:],b[-1000:])
 return a
save()
with (CACHE/'compiler.lease').open('rb') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
 # Version/fmt and debug codec units already terminal0, admitted above; no replay.
 for mode in ['debug','release']:
  deps=CACHE/'target'/mode/'deps'
  def unique(pattern):
   xs=list(deps.glob(pattern));assert len(xs)==1,(pattern,[x.name for x in xs]);return xs[0]
  libs={n:unique('lib'+n+'-*.rlib') for n in ['voltafabric','serde_json','libc']};report['rlibs'][mode]={str(p):sha(p) for p in libs.values()};save()
  flags=['--edition=2021','-C','overflow-checks=yes','-C','opt-level='+('0' if mode=='debug' else '3'),'-D','warnings']
  if mode=='release':flags+=['-C','lto=thin','-C','embed-bitcode=yes']
  codec=out/('libcodec_'+mode+'.rlib');producer=out/('libproducer_'+mode+'.rlib');probe=out/('probe_'+mode)
  cs=R/'producer/claude_codec/src/lib.rs';ps=R/'producer/quantizer_bridge/src/lib.rs'
  if mode!='debug':
   run('codec_build_'+mode,[str(T/'rustc'),str(cs),'--crate-name','vf_lab_e4m3_codec','--crate-type=rlib','-F','unsafe_code',*flags,'-o',str(codec)],timeout=60)
   run('codec_tests_build_'+mode,[str(T/'rustc'),'--test',str(cs),'--crate-name','vf_lab_e4m3_codec','-F','unsafe_code',*flags,'-o',str(out/('codec_tests_'+mode))],timeout=60)
   run('codec_tests_'+mode,[str(out/('codec_tests_'+mode))],contains='9 passed; 0 failed; 1 ignored;')
  ext=['--extern','vf_lab_e4m3_codec='+str(codec),'-L','dependency='+str(out)]
  if mode!='debug':
   run('producer_build_'+mode,[str(T/'rustc'),str(ps),'--crate-name','vf_lab_quantizer_bridge','--crate-type=rlib',*flags,*ext,'-o',str(producer)],timeout=60)
   run('producer_tests_build_'+mode,[str(T/'rustc'),'--test',str(ps),'--crate-name','vf_lab_quantizer_bridge',*flags,*ext,'-o',str(out/('producer_tests_'+mode))],timeout=60)
   run('producer_tests_'+mode,[str(out/('producer_tests_'+mode))],contains='4 passed')
  alias=out/('deps_'+mode);alias.mkdir(exist_ok=True);dest=alias/'libvf_lab_e4m3_codec.rlib'
  if not dest.exists():os.link(codec,dest)
  assert codec.stat().st_ino==dest.stat().st_ino
  args=[*flags,'-L','dependency='+str(alias),'-L','dependency='+str(deps),'-L','dependency='+str(out),'--extern','vf_lab_quantizer_bridge='+str(producer),'--extern','vf_lab_e4m3_codec='+str(codec)]
  for n,p in libs.items():args+=['--extern',n+'='+str(p)]
  run('probe_build_'+mode,[str(T/'rustc'),str(R/'producer_probe.rs'),*args,'-o',str(probe)],timeout=60)
  if mode=='release':run('clippy',[str(T/'clippy-driver'),str(R/'producer_probe.rs'),*args,'--emit=metadata','-o',str(out/'clippy.rmeta')],timeout=60)
  run('verify_'+mode,[str(probe),'verify',str(R/'CORPUS.json')])
 # Mutants are source-private; normal artifacts remain intact. No timing occurs here.
 deps=CACHE/'target/debug/deps';flags=['--edition=2021','-C','overflow-checks=yes','-C','opt-level=0','-D','warnings']
 libs={n:unique('lib'+n+'-*.rlib') for n in ['voltafabric','serde_json','libc']}
 for name in ['lose_sign','write_on_refusal']:
  md=out/name;md.mkdir();codec=md/'libcodec.rlib';prod=md/'libproducer.rlib'
  if name=='lose_sign':
   source=R/'mutations/lose_sign/candidate.rs'; # module tree copied with exact normal lib/reference/tests
   cs=source.parent/'lib.rs'
  else:cs=R/'producer/claude_codec/src/lib.rs'
  run(name+'_codec_compile',[str(T/'rustc'),str(cs),'--crate-name','vf_lab_e4m3_codec','--crate-type=rlib','-F','unsafe_code',*flags,'-o',str(codec)],timeout=60)
  os.link(codec,md/'libvf_lab_e4m3_codec.rlib')
  ps=R/'mutations/write_on_refusal/lib.rs' if name=='write_on_refusal' else R/'producer/quantizer_bridge/src/lib.rs'
  run(name+'_producer_compile',[str(T/'rustc'),str(ps),'--crate-name','vf_lab_quantizer_bridge','--crate-type=rlib',*flags,'--extern','vf_lab_e4m3_codec='+str(codec),'-L','dependency='+str(md),'-o',str(prod)],timeout=60)
  args=[*flags,'-L','dependency='+str(deps),'-L','dependency='+str(md),'--extern','vf_lab_quantizer_bridge='+str(prod),'--extern','vf_lab_e4m3_codec='+str(codec)]
  for n,p in libs.items():args+=['--extern',n+'='+str(p)]
  binary=md/'probe';run(name+'_probe_compile',[str(T/'rustc'),str(R/'producer_probe.rs'),*args,'-o',str(binary)],timeout=60)
  run(name+'_detected',[str(binary),'verify',str(R/'CORPUS.json')],expected=101,contains='candidate codes' if name=='lose_sign' else 'refusal wrote caller output')
 assert all(sha(Path(p))==h for group in report['rlibs'].values() for p,h in group.items())
 assert all(sha(R/n)==h for n,h in P['files'].items()) and all(sha(CACHE/n)==h for n,h in old['files'].items())
 run('elf_cpu_dependencies',['/usr/bin/ldd',str(out/'probe_release')])
 nm=run('C_ABI_symbol',['/usr/bin/nm','--defined-only',str(out/'probe_release')]);assert b' vf_lab_quantize' in nm
 report.update(status='R75_EXISTING_CPU_PRODUCER_GATES_PASS_NO_BENCHMARK',source_stable=True,rlibs_stable=True,unit_tests=26,compiled_rust_mutants_detected=2,binaries={p.name:sha(p) for p in [out/'probe_debug',out/'probe_release',out/'libcodec_release.rlib',out/'libproducer_release.rlib']});save();print(json.dumps(dict(status=report['status'],steps=len(report['steps']))))
