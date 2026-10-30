"""Experimental two-host correlated CPU session; snapshots do not reserve resources.

Run explicitly only after checking host policy. No unknown operation is retried.
"""
import hashlib,json,os,pathlib,struct,subprocess,sys,uuid
from adapters.resource_admission import ResourcePolicy,AdmissionError

B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012')
R=B/'.runs/gitlab-import-r113-20261028/repository'
BIN=B/'.runs/r326-arm-build-20261029/target/debug/examples/r261_multipart_cpu_session'
BIN_SHA='016ae8d312a71ad05e284ec389517d3301e52299a445698405ed8019a5196a38'
REMOTE='/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs'
REMOTE_CODE=REMOTE+'/r315-cross-host-code-r1'
REMOTE_ART=REMOTE+'/r315-cross-host-remote-r1/rank9.safetensors'
REMOTE_SHA='ebc3cd90c0578ebe130ba52d0281a39a9154e9683b7e241b99591f58cb8ed1f8'
LOCAL_SHA='3f841052b0b3e09aabd09876f1fc8a5ef50cca2c18a4b7c17e9f2744afe8bec4'
INPUT_SHA='27e94ad2a07d39e4ceb08990154174babb717f74cb9c8f0a57ddfc64662d2af5'
EXPECTED_SHA='a40fe7cfb4f54da40fc57de87d38b379d8533fd2f4c8eb063711abe67dc42bc8'
POLICY=ResourcePolicy(2*1024**3,1024**3,2*1024**3,256*1024**2)

def admit_both(local,remote,start):
    local_snapshot=local()
    remote_snapshot=remote()
    return start(local_snapshot,remote_snapshot)

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def validate_fixture_input(p):
    rows=p.get('inputs')
    if not isinstance(rows,list) or len(rows)!=1 or not isinstance(rows[0],list) or len(rows[0])!=2048:
        raise ValueError('fixture input shape changed')
    try:
        encoded=b''.join(struct.pack('<e',x) for x in rows[0])
    except (TypeError,ValueError,OverflowError,struct.error) as exc:
        raise ValueError('invalid fixture input') from exc
    if hashlib.sha256(encoded).hexdigest()!=INPUT_SHA:
        raise ValueError('fixture input differs from oracle')

def ssh_python(script):
    # stdin is data, never a shell interpolation; a timeout is not retried.
    p=subprocess.run(['/usr/bin/ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','pc1-lan','python3','-'],input=script,text=True,capture_output=True,timeout=30)
    if p.returncode: raise AdmissionError('remote check or preparation failed; inspect before retry')
    return json.loads(p.stdout)

def remote_check():
    script='import json,sys\nsys.path.insert(0,'+repr(REMOTE_CODE)+')\nfrom adapters.resource_admission import ResourcePolicy\nprint(json.dumps(ResourcePolicy(2*1024**3,1024**3,2*1024**3,256*1024**2).check('+repr(REMOTE)+')))\n'
    return ssh_python(script)

def persist(path,value):
    with open(path,'x',encoding='utf8') as f:
        json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def execute(local_snapshot,remote_snapshot):
    if digest(BIN)!=BIN_SHA:raise ValueError('coordinator hash changed')
    local_art=R/'research/R190/artifacts/rank2.safetensors';all_art=R/'research/R190/artifacts/rank9.safetensors'
    if digest(local_art)!=LOCAL_SHA or digest(all_art)!=REMOTE_SHA:raise ValueError('local artifact hash changed')
    p=json.loads((R/'research/R159/input.json').read_bytes());validate_fixture_input(p)
    run='r327-'+uuid.uuid4().hex;root=B/'.runs'/run;root.mkdir(mode=0o700);(root/'receiver').mkdir()
    remote_root=REMOTE+'/'+run
    state={'status':'REMOTE_PREPARE_INTENT','run_id':run,'remote_root':remote_root,'local_admission':local_snapshot,'remote_admission':remote_snapshot,'gpu':False};persist(root/'intent.json',state)
    print(json.dumps({'status':'REMOTE_PREPARE_INTENT','run_id':run,'remote_root':remote_root,'local_root':str(root)}),flush=True)
    persist(root/'reuse-map.json',{'0:2':{'path':str(local_art),'sha256':LOCAL_SHA,'bytes':local_art.stat().st_size}})
    script=("import os,pathlib,json,hashlib\n"
      +"p=pathlib.Path("+repr(REMOTE_ART)+"); h=hashlib.sha256(p.read_bytes()).hexdigest(); assert h=="+repr(REMOTE_SHA)+"\n"
      +"d=pathlib.Path("+repr(remote_root)+"); d.mkdir(mode=0o700); (d/'receiver').mkdir()\n"
      +"f=open(d/'reuse-map.json','x'); json.dump({'0:9':{'path':str(p),'sha256':h,'bytes':p.stat().st_size}},f);f.flush();os.fsync(f.fileno());f.close()\n"
      +"fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps({'ok':True}))\n")
    try:
        if ssh_python(script).get('ok') is not True:raise RuntimeError('remote prepare unconfirmed')
        persist(root/'remote-prepared.json',{'remote_root':remote_root})
        import torch
        from adapters.session_bridge import DurableSession
        assert torch.version.cuda is None;torch.set_num_threads(1)
        p['routing']['stamp'].update(run_id=run,plan_id='r327-cross-host',model_revision='qwen-r132-eight-copies');p['routing']['experts']=list(range(8));p['routing']['routes']=[[{'expert':e,'weight':.125} for e in range(8)]];p['routing']['registry']['copies']=[{'expert':e,'rank':2 if e<4 else 9,'local':e%4} for e in range(8)];p['routing']['resident_order']={'2':list(range(4)),'9':list(range(4))};p['artifacts']=[str(local_art),str(all_art)];p['fp16_completions']=True
        remote_py=REMOTE.replace('/.runs','/.tools/adapter-cpu/bin/python');remote_path=':'.join([REMOTE_CODE,REMOTE+'/r276-package-r1',REMOTE,REMOTE+'/r299-code-r1'])
        p['workers']=[{'rank':2,'argv':['/usr/bin/env','PYTHONPATH='+str(R),sys.executable,'-B',str(B/'.runs/r315-cross-host-code-r1/worker.py'),str(root/'receiver')]},{'rank':9,'argv':['/usr/bin/ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','pc1-lan','env','CUDA_VISIBLE_DEVICES=-1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+remote_path,remote_py,'-B',REMOTE_CODE+'/worker.py',remote_root+'/receiver']}]
        x=torch.tensor(p['inputs'],dtype=torch.float16);ids=torch.tensor([list(range(8))],dtype=torch.int64);weights=torch.full((1,8),.125,dtype=torch.float32)
        s=DurableSession(BIN,p,root/'ledger',deadline_seconds=180,admission=lambda:POLICY.check(root),correlated=True);hashes=[]
        for batch in range(2):
            output=s(None,x,weights,ids,None,None)
            done=json.loads((s.directory/f'batch-{batch:020d}.done.json').read_text());raw=(s.directory/f'batch-{batch:020d}.fp16').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=done['sha256'] or output.contiguous().numpy().tobytes()!=raw:raise AssertionError('durable output mismatch')
            if done['sha256']!=EXPECTED_SHA:raise AssertionError('independent expert oracle mismatch')
            hashes.append(done['sha256'])
        closed=s.close()
        if len(closed['close'])!=2 or any(c['exit_code']!=0 or c['bye_execs']!=2 for c in closed['close']):raise RuntimeError('worker close failed')
        state.update(status='PASS',batches=hashes,close=closed['close'])
    except BaseException as e:
        state.update(status='FAILED_OR_UNKNOWN',error_type=type(e).__name__,error=str(e)[:300])
        persist(root/'receipt.json',state)
        raise
    persist(root/'receipt.json',state);return state

def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        result=admit_both(lambda:POLICY.check(B),remote_check,execute)
        print(json.dumps(result));return 0
    except AdmissionError as e:
        print(json.dumps({'status':'ADMISSION_REFUSED_OR_REMOTE_UNCONFIRMED','error':str(e)}));return 1

if __name__=='__main__':sys.exit(main())
