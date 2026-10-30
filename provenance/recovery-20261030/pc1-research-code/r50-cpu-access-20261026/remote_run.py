"""One bounded, private PC1 CPU correctness session. Exclusive receipts forbid replay."""
from pathlib import Path
import hashlib,json,os,platform,signal,struct,subprocess,time
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(j,g,shape):
    assert j['check_only'] is True and j['weights_loaded']==1 and j['samples']==[]
    if shape=='real':
        assert j['status']=='EXACT_REAL_R9_STANDALONE_PROJECTIONS' and len(j['outputs'])==3
        assert j['artifact_sha256']==g['artifact_sha256']
        for a,c in zip(j['outputs'],g['cases']):
            assert a['name']==c['name'] and (a['rows'],a['cols'])==(c['rows'],c['cols'])
            assert a['output_bits']==c['output_bits']
        return 9728
    assert j['status']=='EXACT_R9_COMPONENT_PROFILE' and j['cases']==45 and len(j['outputs'])==45
    assert j['artifact_sha256']==g['weights_sha256']
    for a,c in zip(j['outputs'],g['cases']):
        assert (a['expert'],a['input'],a['coefficient_bits'])==(c['expert'],c['input'],struct.unpack('<I',struct.pack('<f',c['coefficient']))[0])
        assert a['output_bits']==[int(x,16) for x in c['output_bits']]
    return 1440

def main():
    assert platform.machine()=='x86_64'
    m=json.loads((ROOT/'ASSETS.json').read_text())
    assert all(sha(ROOT/p)==digest for p,digest in m['files'].items())
    out=ROOT/'results';out.mkdir(mode=0o700)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();summary=[]
    for mode in ['debug','release']:
        for shape in ['real','tiny']:
            name=mode+'-'+shape;receipt=out/(name+'.json')
            binary=ROOT/('probe-x86-'+mode);artifact=ROOT/(shape+'.safetensors');golden=ROOT/(shape+'-golden.json')
            args=[str(binary),str(artifact),str(golden),'--check']
            record={'operation':name,'run_id':m['run_id'],'argv':args,'status':'UNKNOWN_NO_REPLAY','automatic_retry':False,'pid':None,'input_hashes':{p.name:sha(p) for p in [binary,artifact,golden]},'boot_id':boot,'uid':os.getuid(),'start_ns':time.monotonic_ns(),'host':platform.node(),'machine':platform.machine()}
            with receipt.open('x') as f:json.dump(record,f,indent=2)
            process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
            record['pid']=process.pid
            record['start_ticks']=Path('/proc',str(process.pid),'stat').read_text().split(') ',1)[1].split()[19]
            receipt.write_text(json.dumps(record,indent=2))
            try:stdout,stderr=process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);stdout,stderr=process.communicate(timeout=3);record['timeout']=True
            (out/(name+'.stdout')).write_bytes(stdout);(out/(name+'.stderr')).write_bytes(stderr)
            record.update(status='PROCESS_TERMINAL',exit_code=process.returncode,end_ns=time.monotonic_ns(),process_reconciled=process.poll() is not None)
            receipt.write_text(json.dumps(record,indent=2))
            assert process.returncode==0,(name,process.returncode,stderr.decode()[-1000:])
            exact=verify(json.loads(stdout),json.loads(golden.read_text()),shape)
            record['exact_values']=exact;record['stdout_sha256']=hashlib.sha256(stdout).hexdigest();receipt.write_text(json.dumps(record,indent=2))
            summary.append(record)
    result={'status':'PC1_X86_PUBLIC_PROJECTIONS_AND_TINY_EXACT','run_id':m['run_id'],'records':summary,'active_operations':[],'gpu':False,'rdma':False,'performance_claim':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'records':len(summary),'exact_values':sum(r['exact_values'] for r in summary)}))
if __name__=='__main__':main()
