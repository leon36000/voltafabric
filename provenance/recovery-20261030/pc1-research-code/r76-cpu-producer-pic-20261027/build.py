from pathlib import Path
import os,json,hashlib,subprocess,fcntl,signal,time
R=Path(__file__).resolve().parent;B=R.parents[1];T=B/'.tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin';CACHE=B/'.runs/r73-immediate-load-state-20261027';out=R/'out';out.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();P=json.loads((R/'INPUTS.json').read_text());assert all(sha(R/n)==h for n,h in P['files'].items())
assert sha(CACHE/'BUILD_PLAN.json')==P['cache_build_plan_sha256'] and sha(CACHE/'build/operation.json')==P['cache_operation_sha256']
p=json.loads((CACHE/'BUILD_PLAN.json').read_text());assert all(sha(CACHE/n)==h for n,h in p['files'].items())
report=dict(status='CPU_PIC_ARCHIVE_STARTED',pid=os.getpid(),steps=[],source_plan_sha256=sha(R/'INPUTS.json'),gpu=False,automatic_retry=False)
def save():(out/'operation.json').write_text(json.dumps(report,indent=2)+'\n')
def run(n,args):
 q=subprocess.Popen(args,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(R/'tmp')),stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True);report['steps'].append(dict(name=n,argv=args,pid=q.pid,status='UNKNOWN'));save()
 try:a,b=q.communicate(timeout=60)
 except subprocess.TimeoutExpired:
  os.killpg(q.pid,signal.SIGKILL);a,b=q.communicate(timeout=3);report['steps'][-1]['timeout']=True
 (out/(n+'.stdout')).write_bytes(a);(out/(n+'.stderr')).write_bytes(b);report['steps'][-1].update(status='TERMINAL',exit_code=q.returncode,process_reconciled=q.poll() is not None);save();assert q.returncode==0,(n,b[-3000:]);return a
save()
with (CACHE/'compiler.lease').open('rb') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
 flags=['--edition=2021','-C','opt-level=3','-C','overflow-checks=yes','-C','relocation-model=pic','-C','embed-bitcode=yes','-C','lto=thin','-D','warnings']
 codec=out/'libvf_lab_e4m3_codec.rlib'
 run('codec_PIC',[str(T/'rustc'),str(R/'producer/claude_codec/src/lib.rs'),'--crate-name','vf_lab_e4m3_codec','--crate-type=rlib','-F','unsafe_code',*flags,'-o',str(codec)])
 run('producer_PIC_static_and_rlib',[str(T/'rustc'),str(R/'producer/quantizer_bridge/src/lib.rs'),'--crate-name','vf_lab_quantizer_bridge','--crate-type=staticlib,rlib',*flags,'--extern','vf_lab_e4m3_codec='+str(codec),'-L','dependency='+str(out),'--out-dir',str(out),'--print','native-static-libs'])
 deps=CACHE/'target/release/deps'
 def unique(pattern):
  x=list(deps.glob(pattern));assert len(x)==1;return x[0]
 libs={n:unique('lib'+n+'-*.rlib') for n in ['voltafabric','serde_json','libc']};report['rlibs']={str(p):sha(p) for p in libs.values()}
 args=[*flags,'-L','dependency='+str(out),'-L','dependency='+str(deps),'--extern','vf_lab_quantizer_bridge='+str(out/'libvf_lab_quantizer_bridge.rlib'),'--extern','vf_lab_e4m3_codec='+str(codec)]
 for n,p in libs.items():args+=['--extern',n+'='+str(p)]
 run('probe_PIC',[str(T/'rustc'),str(R/'producer_probe.rs'),*args,'-o',str(out/'probe')])
 run('probe_PIC_exact_cpu',[str(out/'probe'),'verify',str(R/'CORPUS.json')])
 assert all(sha(R/n)==h for n,h in P['files'].items()) and all(sha(Path(p))==h for p,h in report['rlibs'].items())
 report.update(status='R76_CPU_PIC_ARCHIVE_AND_EXACT_CORPUS_PASS_NO_GPU',source_stable=True,artifacts={n:dict(sha256=sha(out/n),bytes=(out/n).stat().st_size) for n in ['libvf_lab_quantizer_bridge.a','libvf_lab_quantizer_bridge.rlib','libvf_lab_e4m3_codec.rlib','probe']});save();print(json.dumps(dict(status=report['status'],archive_bytes=(out/'libvf_lab_quantizer_bridge.a').stat().st_size)))
