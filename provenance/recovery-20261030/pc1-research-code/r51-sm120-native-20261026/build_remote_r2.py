"""Isolated offline SM120 CUDA compilation; no CUDA executable or GPU context."""
from pathlib import Path
import hashlib,json,os,shutil,signal,subprocess,time
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(name,args):
    receipt=ROOT/('r2-'+name+'.json');j={'status':'UNKNOWN_NO_REPLAY','argv':args,'automatic_retry':False,'gpu_executed':False,'start_ns':time.monotonic_ns()}
    with receipt.open('x') as f:json.dump(j,f)
    p=subprocess.Popen(args,cwd=ROOT/'native',env=dict(os.environ,TMPDIR=str(ROOT/'tmp'),CUDA_VISIBLE_DEVICES='',CUDA_CACHE_DISABLE='1'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    j['pid']=p.pid;receipt.write_text(json.dumps(j))
    try:o,e=p.communicate(timeout=150)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);o,e=p.communicate(timeout=3);j['timeout']=True
    (ROOT/('r2-'+name+'.stdout')).write_bytes(o);(ROOT/('r2-'+name+'.stderr')).write_bytes(e)
    j.update(status='PROCESS_TERMINAL',exit_code=p.returncode,end_ns=time.monotonic_ns(),process_reconciled=p.poll() is not None);receipt.write_text(json.dumps(j,indent=2))
    assert p.returncode==0,(name,e.decode()[-2000:])

def main():
    expected=json.loads((ROOT/'native-inputs.json').read_text())['files'];assert all(sha(ROOT/'native'/k)==h for k,h in expected.items())
    nvcc='/usr/local/cuda-13/bin/nvcc';cxx=shutil.which('g++-13') or shutil.which('g++');assert cxx
    run('nvcc-version',[nvcc,'--version']);run('cxx-version',[cxx,'--version'])
    base=[nvcc,'-I'+str(ROOT/'cuda-compat'),'-ccbin',cxx,'-std=c++17','-lineinfo','--fmad=false','--ftz=false','-arch=sm_120','--shared','-Xcompiler','-fPIC','-lcuda']
    for name,src,flags in [('baseline','resident-original.cu',['-O3']),('release','resident.cu',['-O3']),('debug','resident.cu',['-O0','-G']),('mutant','resident.cu',['-O3','-DVF_R13_MUTATE_SKIP_WEIGHT']),('fault','resident.cu',['-O3','-DVF_R13_TEST_SUBMISSION_FAILURE'])]:
        run('build-'+name,base+[src,*flags,'-o',str(ROOT/('lib-'+name+'.so'))])
    run('elf',['readelf','-h',str(ROOT/'lib-release.so')]);run('needed',['readelf','-d',str(ROOT/'lib-release.so')]);run('device-images',['/usr/local/cuda-13/bin/cuobjdump','--list-elf',str(ROOT/'lib-release.so')])
    assert all(sha(ROOT/'native'/k)==h for k,h in expected.items())
    j={'status':'CUDA_COMPILED_SM120_NO_GPU_EXECUTION','source_hashes':expected,'binaries':{p.name:sha(p) for p in ROOT.glob('*.so')},'gpu_executed':False,'native_abi':1,'architecture':'sm_120','host_arch':'x86_64','compiler':nvcc}
    (ROOT/'BUILD.json').write_text(json.dumps(j,indent=2));print(json.dumps(j))
if __name__=='__main__':main()
