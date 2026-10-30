"""Private sequential Rust1.90 build; existing cached dependencies are read-only."""
from pathlib import Path
import json,hashlib,os,subprocess,shutil,fcntl,time,signal
ROOT=Path(__file__).resolve().parent
BASE=Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c')
OLD=BASE/'.runs/r52-sm120-rust-worker-20261026'
RUST=BASE/'.tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=json.loads((ROOT/'BUILD_PLAN.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in plan['private_source_hashes'].items())
    assert all(sha(OLD/'source'/n)==h for n,h in plan['canonical_rust_sources'].items())
    lib=OLD/'native/libvf_r13.so';assert sha(lib)==plan['native_sha256']
    out=ROOT/'bin';out.mkdir(exist_ok=False);(ROOT/'tmp').mkdir(exist_ok=False)
    shutil.copyfile(lib,out/'libvf_r13.so')
    lock=(ROOT/'compiler.lease').open('x+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,TMPDIR=str(ROOT/'tmp'),CUDA_VISIBLE_DEVICES='',CARGO_PKG_VERSION='0.1.0',VF_MXFP4_LIBRARY_SHA256=plan['native_sha256'])
    report={'status':'BUILD_STARTED_NO_GPU','pid':os.getpid(),'steps':[],'automatic_retry':False,'gpu_executed':False}
    def save():(ROOT/'BUILD_RESULT.json').write_text(json.dumps(report,indent=2))
    def run(name,argv):
        mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        assert mem>3*1024**2 and shutil.disk_usage(ROOT).free>4*1024**3
        with (out/(name+'.stdout')).open('x') as so,(out/(name+'.stderr')).open('x') as se:
            p=subprocess.Popen(argv,stdout=so,stderr=se,env=env,start_new_session=True)
            entry={'name':name,'argv':argv,'pid':p.pid,'start_ns':time.monotonic_ns(),'status':'UNKNOWN'};report['steps'].append(entry);save()
            try:code=p.wait(timeout=90)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5);raise
            entry.update(exit_code=code,process_reconciled=True,end_ns=time.monotonic_ns(),status='TERMINAL');save()
            assert code==0,(name,code)
    run('version',[str(RUST/'rustc'),'-vV'])
    run('fmt',[str(RUST/'rustfmt'),'--edition','2021','--check',str(ROOT/'main.rs'),str(ROOT/'identity.rs'),str(ROOT/'resident.rs')])
    for mode,opt in [('debug','0'),('release','3')]:
        deps=OLD/'target'/mode/'deps'
        externs={}
        for n in ['serde','serde_json','num_bigint','safetensors','libc']:
            ps=list(deps.glob('lib'+n+'-*.rlib'));assert len(ps)==1,(n,ps);externs[n]=ps[0]
        common=['--edition=2021','-C','opt-level='+opt,'-C','lto=thin','-C','overflow-checks=yes','-D','warnings','-L','dependency='+str(deps)]
        args=[str(RUST/'rustc'),'--crate-name','voltafabric','--crate-type','rlib',str(OLD/'source/src/lib.rs'),*common,'-o',str(out/('libvoltafabric_'+mode+'.rlib'))]
        for n,p in externs.items():args+=['--extern',n+'='+str(p)]
        run('library_'+mode,args)
        args=[str(RUST/'rustc'),'--crate-name','vf_mxfp4_gpu_worker',str(ROOT/'main.rs'),*common,'--extern','voltafabric='+str(out/('libvoltafabric_'+mode+'.rlib')),'--extern','serde_json='+str(externs['serde_json']),'--extern','libc='+str(externs['libc']),'-L','native='+str(out),'-C','link-arg=-Wl,-rpath,$ORIGIN','-o',str(out/('worker_'+mode))]
        run('worker_'+mode,args)
        run('loader_'+mode,['ldd',str(out/('worker_'+mode))])
        assert 'not found' not in (out/('loader_'+mode+'.stdout')).read_text()
        report.setdefault('dependency_hashes',{})[mode]={str(p):sha(p) for p in externs.values()}
    assert all(sha(ROOT/n)==h for n,h in plan['private_source_hashes'].items())
    assert all(sha(OLD/'source'/n)==h for n,h in plan['canonical_rust_sources'].items())
    report.update(status='PRIVATE_CANDIDATE_COMPILED_NO_GPU',binaries={n:sha(out/n) for n in ['worker_debug','worker_release','libvf_r13.so']},end_ns=time.monotonic_ns());save()
    print(json.dumps(report))
if __name__=='__main__':main()
