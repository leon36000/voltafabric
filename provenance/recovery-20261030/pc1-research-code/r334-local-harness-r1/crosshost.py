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
REMOTE_ART=REMOTE+'/r333-distinct-correlated-r1/rank9.safetensors'
REMOTE_SHA='4d8eaf68cda68c998c7c6f7bfc98e8c22ba1aa7bbea6b271cdea33a22db67482'
LOCAL_SHA='cd947531488c90034c3b9ba8213673cbbf628d59354437dd49d1ceba1aac87c1'
INPUT_SHA='440fb5ada1472ce67fecfe589fb620912ba7bd018a13b6585cb4d3a953429ff8'
SOURCE_SHA='0e597ecf47a568d6066ac2490bb09b37f615a1f725f9bdac9364cab4c195928b'
EXPERTS=[198,204,109,121,171,76,166,173]
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

def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def persist(path,value):
    with open(path,'x',encoding='utf8') as f:
        json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    sync_dir(path.parent)

def execute(local_snapshot,remote_snapshot):
    if digest(BIN)!=BIN_SHA:raise ValueError('coordinator hash changed')
    local_art=B/'.runs/r206-gate-route-20261028/owners/rank2.safetensors';all_art=B/'.runs/r206-gate-route-20261028/owners/rank9.safetensors';source=B/'.runs/r206-gate-route-20261028/selected.safetensors'
    if digest(local_art)!=LOCAL_SHA or digest(all_art)!=REMOTE_SHA or digest(source)!=SOURCE_SHA:raise ValueError('local artifact hash changed')
    p=json.loads((R/'research/R333/input.json').read_bytes());validate_fixture_input(p)
    run='r333x-'+uuid.uuid4().hex;root=B/'.runs'/run;root.mkdir(mode=0o700);sync_dir(root.parent);(root/'receiver').mkdir()
    remote_root=REMOTE+'/'+run
    state={'status':'REMOTE_PREPARE_INTENT','run_id':run,'remote_root':remote_root,'local_admission':local_snapshot,'remote_admission':remote_snapshot,'gpu':False};persist(root/'intent.json',state)
    print(json.dumps({'status':'REMOTE_PREPARE_INTENT','run_id':run,'remote_root':remote_root,'local_root':str(root)}),flush=True)
    persist(root/'reuse-map.json',{'0:2':{'path':str(local_art),'sha256':LOCAL_SHA,'bytes':local_art.stat().st_size}})
    script=("import os,pathlib,json,hashlib\n"
      +"p=pathlib.Path("+repr(REMOTE_ART)+"); h=hashlib.sha256(p.read_bytes()).hexdigest(); assert h=="+repr(REMOTE_SHA)+"\n"
      +"d=pathlib.Path("+repr(remote_root)+"); d.mkdir(mode=0o700); fd=os.open(d.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd); (d/'receiver').mkdir()\n"
      +"f=open(d/'reuse-map.json','x'); json.dump({'0:9':{'path':str(p),'sha256':h,'bytes':p.stat().st_size}},f);f.flush();os.fsync(f.fileno());f.close()\n"
      +"fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps({'ok':True}))\n")
    try:
        if ssh_python(script).get('ok') is not True:raise RuntimeError('remote prepare unconfirmed')
        persist(root/'remote-prepared.json',{'remote_root':remote_root})
        import torch
        from adapters.session_bridge import DurableSession
        assert torch.version.cuda is None;torch.set_num_threads(1)
        p['routing']['stamp'].update(run_id=run,plan_id='r333-distinct-cross-host',model_revision='1355db6a052410cfd62085d94b58866fd0f2c3c5',layer=0,batch=0,epoch=1);p['routing']['experts']=EXPERTS;p['routing']['routes']=[[{'expert':e,'weight':.125} for e in EXPERTS]];p['routing']['registry']['epoch']=1;p['routing']['registry']['copies']=[{'expert':e,'rank':2 if i<4 else 9,'local':i%4} for i,e in enumerate(EXPERTS)];p['routing']['resident_order']={'2':list(range(4)),'9':list(range(4))};p['artifacts']=[str(local_art),str(all_art)];p['fp16_completions']=True
        remote_py=REMOTE.replace('/.runs','/.tools/adapter-cpu/bin/python');remote_path=':'.join([REMOTE_CODE,REMOTE+'/r276-package-r1',REMOTE,REMOTE+'/r299-code-r1'])
        p['workers']=[{'rank':2,'argv':['/usr/bin/env','PYTHONPATH='+str(R),sys.executable,'-B',str(B/'.runs/r315-cross-host-code-r1/worker.py'),str(root/'receiver')]},{'rank':9,'argv':['/usr/bin/ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','pc1-lan','env','CUDA_VISIBLE_DEVICES=-1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+remote_path,remote_py,'-B',REMOTE_CODE+'/worker.py',remote_root+'/receiver']}]
        x=torch.tensor(p['inputs'],dtype=torch.float16);cases=[(EXPERTS,[.125]*8),(list(reversed(EXPERTS)),[.2,.17,.14,.12,.11,.1,.09,.07])];sys.path.insert(0,str(R/'research/R199'));from nvfp4_oracle import expert_forward
        s=DurableSession(BIN,p,root/'ledger',deadline_seconds=180,admission=lambda:POLICY.check(root),correlated=True);hashes=[];errors=[]
        for batch,(order,coefficients) in enumerate(cases):
            ids=torch.tensor([order],dtype=torch.int64);weights=torch.tensor([coefficients],dtype=torch.float32)
            output=s(None,x,weights,ids,None,None)
            done=json.loads((s.directory/f'batch-{batch:020d}.done.json').read_text());raw=(s.directory/f'batch-{batch:020d}.fp16').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=done['sha256'] or output.contiguous().numpy().tobytes()!=raw:raise AssertionError('durable output mismatch')
            expected=torch.zeros(x.shape[1],dtype=torch.float64)
            for expert,weight in zip(order,weights[0].tolist()):expected+=expert_forward(source,0,expert,x[0]).double()*weight
            error=float((output[0].double()-expected.half().double()).abs().max())
            if error>=0.0001:raise AssertionError('independent expert oracle mismatch')
            errors.append(error)
            hashes.append(done['sha256'])
        closed=s.close()
        if len(closed['close'])!=2 or any(c['exit_code']!=0 or c['bye_execs']!=2 for c in closed['close']):raise RuntimeError('worker close failed')
        state.update(status='PASS',batches=hashes,oracle_errors=errors,close=closed['close'])
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
