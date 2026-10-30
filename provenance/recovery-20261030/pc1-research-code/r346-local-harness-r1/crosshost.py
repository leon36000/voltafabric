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
WORKER_PINS={'worker.py':'8e068244d3a45a46c416f2ddbda38733f30d5a2ce79930f928345b2755cc3961','owner_service.py':'f8d10efa14d366bb5a6992667cb001eac7cd86b4e0b2e46fd76bb6fc90ebd083','wire.py':'fe12930345bffd0ffebf3f43d634e8537164cb4dc1f3e28171608572e0230fad','local_reuse.py':'4b65f551694a582ee06c088d40efafd19e72d28aa628cab235e93fa936c36d4e'}
ADAPTER_PINS={'resource_admission.py':'0c0430e19843022f7584a10b07361d48fa3275ef84340c56b79d027cdd43e389','skinny_moe.py':'a149a72c4b38edebd07515d0dd74f60881042f9e6b781d33cba4a3089b452350','single_owner_admission.py':'087903e54cfc79c0ca198c6efd7cff95c6a1c6592929d21da9fef4a9e0a6598a','resident_multipart.py':'927693b534a9bd0b5aa59bc577f220b2bef5513a7742ed3c218685bce145bb84','persistent_gpu_owner.py':'e12945ae9167843f7f056bcc12734ee222af5f361def1f108138960168d823f3','qpn2_cpu_reference.py':'5f2f625913d261e1e94adc82506dfdf9cab47a098b1e2f778c082cd2e34868a4'}
ORACLE_SHA='939c75880465e9ea901f20c875228ea7dc55307a953d1a0a7c70abb68f0f6e4f'

def admit_both(local,remote,start):
    local_snapshot=local()
    remote_snapshot=remote()
    return start(local_snapshot,remote_snapshot)

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def verify_pins(files):
    for path,expected in files.items():
        try:
            actual=digest(path)
        except OSError as exc:
            raise ValueError('required code unavailable') from exc
        if actual!=expected:raise ValueError('required code hash changed')

def local_code_pins():
    result={B/'.runs/r315-cross-host-code-r1'/name:sha for name,sha in WORKER_PINS.items()}
    result.update({R/'adapters'/name:sha for name,sha in ADAPTER_PINS.items()})
    # The local preflight now reports structured refusal details; the remote
    # worker still uses its separately pinned prior snapshot.
    result[R/'adapters/resource_admission.py']='5223ed38206e2eb92d6ed857c1c9425c00d82eb6c2c331646fbbb4ec3bdbe28f'
    result[R/'adapters/__init__.py']='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    result[R/'adapters/session_bridge.py']='febf8057c4ffd6f28eaed137cf3cc7693cb112176180bc6fa726185e4e00639e'
    result[R/'research/R199/nvfp4_oracle.py']=ORACLE_SHA
    return result

def remote_code_pins():
    result={REMOTE_CODE+'/'+name:sha for name,sha in WORKER_PINS.items()}
    result[REMOTE_CODE+'/adapters/__init__.py']='4032724b98ed267185b12b8ec78dd9bcb0423ab1c89a6f0dad99397fd51159d9'
    for name,sha in ADAPTER_PINS.items():
        folder=REMOTE_CODE+'/adapters' if name in ('resource_admission.py','skinny_moe.py','single_owner_admission.py') else REMOTE+'/r299-code-r1/adapters'
        result[folder+'/'+name]=sha
    return result

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
    script='import hashlib,json,pathlib,sys\nfor path,sha in '+repr(remote_code_pins())+'.items():\n if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()!=sha: raise ValueError("remote code hash changed")\nsys.path.insert(0,'+repr(REMOTE_CODE)+')\nfrom adapters.resource_admission import ResourcePolicy\nprint(json.dumps(ResourcePolicy(2*1024**3,1024**3,2*1024**3,256*1024**2).check('+repr(REMOTE)+')))\n'
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
    verify_pins(local_code_pins())
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
        from adapters.qwen_decode import TransactionalDecoder,DecodeError
        from adapters.qwen_kv_transaction import TransactionalKvAttention,KvTransactionError
        from adapters.qwen_gdn_transaction import TransactionalGdnAttention,GdnTransactionError
        bound={'run_id':run,'model_revision':p['routing']['stamp']['model_revision'],'plan_id':p['routing']['stamp']['plan_id'],'layer':0,'epoch':1}
        other={**bound,'layer':1}
        def kv_ref(z,old):
            return z,(torch.zeros((1,2,1,256)),torch.zeros((1,2,1,256)))
        def gdn_ref(z,history,previous):return z,history,torch.zeros((1,32,128,128))
        class FailGdn(TransactionalGdnAttention):
            def commit(self):raise RuntimeError('injected commit failure')
        kv=TransactionalKvAttention(kv_ref,bound);gdn=FailGdn(gdn_ref,other);results=[]
        order,coefficients=cases[0];ids=torch.tensor([order],dtype=torch.int64);weights=torch.tensor([coefficients],dtype=torch.float32)
        def forward(token,batch):
            kv(x,torch.empty_like(x));gdn(x,torch.empty_like(x))
            out=s(None,x,weights,ids,None,None);results.append(out);return out
        decoder=TransactionalDecoder([kv,gdn],[bound,other],forward,lambda _:7,context_limit=16,eos_token_ids=(7,))
        try:decoder.generate([1],1)
        except RuntimeError as exc:
            if str(exc)!='injected commit failure':raise
        else:raise AssertionError('partial generation returned')
        try:decoder.generate([1],1)
        except DecodeError:pass
        else:raise AssertionError('failed generation replayed')
        try:kv.begin({**bound,'batch':1,'attempt':1})
        except KvTransactionError:pass
        else:raise AssertionError('committed KV reused')
        try:gdn.begin({**other,'batch':0,'attempt':2})
        except GdnTransactionError:pass
        else:raise AssertionError('GDN reused')
        output=results[0];done=json.loads((s.directory/'batch-00000000000000000000.done.json').read_text());raw=(s.directory/'batch-00000000000000000000.fp16').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=done['sha256'] or output.contiguous().numpy().tobytes()!=raw:raise AssertionError('durable output mismatch')
        expected=torch.zeros(x.shape[1],dtype=torch.float64)
        for expert,weight in zip(order,weights[0].tolist()):expected+=expert_forward(source,0,expert,x[0]).double()*weight
        error=float((output[0].double()-expected.half().double()).abs().max())
        if error>=0.0001:raise AssertionError('independent expert oracle mismatch')
        errors.append(error);hashes.append(done['sha256']);closed=s.close()
        claim=json.loads((s.directory/'close.claim.json').read_text());assert claim['identity']==s.identity and claim['next_batch']==1 and (s.directory/'closed.json').exists();state['durable_close_claim_confirmed']=True
        if len(closed['close'])!=2 or any(c['exit_code']!=0 or c['bye_execs']!=1 for c in closed['close']):raise RuntimeError('worker close failed')
        state.update(injected_partial_commit_refused=True,replay_refused=True,kv_fenced=True,gdn_fenced=True)
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
        print(json.dumps({'status':'ADMISSION_REFUSED_OR_REMOTE_UNCONFIRMED','error':str(e),
                          'snapshot':e.snapshot,'limits':e.limits,'reasons':e.reasons}));return 1

if __name__=='__main__':sys.exit(main())
