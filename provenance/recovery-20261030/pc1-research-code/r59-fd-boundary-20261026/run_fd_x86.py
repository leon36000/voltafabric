"""CPU-only paired FD regression on PC1. No CUDA, HCA, driver or route operations."""
from pathlib import Path
import importlib.util,unittest,json,hashlib,os,platform
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert platform.machine()=='x86_64'
 plan=json.loads((ROOT/'INPUTS.json').read_text());assert all(sha(ROOT/n)==h for n,h in plan.items())
 receipt=ROOT/'operation.json'
 with receipt.open('x') as f:json.dump({'status':'CPU_RUNNING_NO_GPU_RDMA','pid':os.getpid(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'automatic_retry':False},f)
 results=[]
 for variant in ['baseline','candidate']:
  spec=importlib.util.spec_from_file_location('fd_'+variant,ROOT/'test_rdma_worker_fds.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  m.BRIDGE=ROOT/('bridge-'+variant+'.c');m.WORK=ROOT/('fd-'+variant);m.WORK.mkdir(exist_ok=False);m.KEEP=True
  if variant=='candidate':m.PROFILES += [('sanitized',['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'])]
  with (ROOT/(variant+'.log')).open('x') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
  rows=getattr(m.WorkerFdBoundary,'results',[])
  d=dict(variant=variant,errors=len(r.errors),failures=len(r.failures),success=r.wasSuccessful(),tests_run=r.testsRun,executions=rows)
  results.append(d);(ROOT/'results.json').write_text(json.dumps(results,indent=2))
  if variant=='baseline':assert not r.wasSuccessful() and not r.errors and len(r.failures)==6
  else:assert r.wasSuccessful()
 assert all(sha(ROOT/n)==h for n,h in plan.items())
 d=json.loads(receipt.read_text());d.update(status='CPU_FD_BOUNDARY_X86_EXACT_AND_FAULTS_VERIFIED',source_inputs=plan,results=results,gpu_executed=False,rdma_executed=False,automatic_retry=False)
 receipt.write_text(json.dumps(d,indent=2));print(json.dumps({'status':d['status'],'variants':len(results)}))
if __name__=='__main__':main()
