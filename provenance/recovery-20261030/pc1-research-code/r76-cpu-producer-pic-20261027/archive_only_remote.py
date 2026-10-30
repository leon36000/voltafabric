from pathlib import Path
import os,json,hashlib,subprocess,fcntl,signal
R=Path(__file__).resolve().parent;B=R.parents[1];T=B/'.tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin';CACHE=B/'.runs/r73-immediate-load-state-20261027';out=R/'out'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();P=json.loads((R/'INPUTS.json').read_text());assert all(sha(R/n)==h for n,h in P['files'].items())
old=json.loads((out/'operation.json').read_text());assert [s['name'] for s in old['steps']]==['codec_PIC','producer_PIC_static_and_rlib'] and [s['exit_code'] for s in old['steps']]==[0,1] and not Path('/proc',str(old['pid'])).exists()
assert not (out/'libvf_lab_quantizer_bridge.a').exists()
receipt=out/'archive-only-operation.json';j=dict(status='CPU_STATIC_ARCHIVE_BUILDING',pid=os.getpid(),initial_failed_operation_sha256=sha(out/'operation.json'),source_plan_sha256=sha(R/'INPUTS.json'),codec_reused_no_rebuild=True,codec_sha256=sha(out/'libvf_lab_e4m3_codec.rlib'),GPU_executed=False,archive_not_exercised_yet=True)
with receipt.open('x') as f:json.dump(j,f)
flags=['--edition=2021','-C','opt-level=3','-C','overflow-checks=yes','-C','relocation-model=pic','-C','embed-bitcode=yes','-C','lto=thin','-D','warnings']
argv=[str(T/'rustc'),str(R/'producer/quantizer_bridge/src/lib.rs'),'--crate-name','vf_lab_quantizer_bridge','--crate-type=staticlib',*flags,'--extern','vf_lab_e4m3_codec='+str(out/'libvf_lab_e4m3_codec.rlib'),'-L','dependency='+str(out),'--out-dir',str(out),'--print','native-static-libs']
with (CACHE/'compiler.lease').open('rb') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
 q=subprocess.Popen(argv,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(R/'tmp')),stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True);j.update(argv=argv,child_pid=q.pid);receipt.write_text(json.dumps(j,indent=2))
 try:a,b=q.communicate(timeout=60)
 except subprocess.TimeoutExpired:
  os.killpg(q.pid,signal.SIGKILL);a,b=q.communicate(timeout=3);j['timeout']=True
 (out/'archive-only.stdout').write_bytes(a);(out/'archive-only.stderr').write_bytes(b);j.update(status='PROCESS_TERMINAL',exit_code=q.returncode,process_reconciled=q.poll() is not None);receipt.write_text(json.dumps(j,indent=2));assert q.returncode==0,b[-2000:]
 assert all(sha(R/n)==h for n,h in P['files'].items())
 p=out/'libvf_lab_quantizer_bridge.a';assert p.stat().st_size<64*1024**2;j.update(status='R76_CPU_PIC_STATIC_ARCHIVE_BUILT_NOT_YET_EXERCISED',source_stable=True,archive_sha256=sha(p),archive_bytes=p.stat().st_size);receipt.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
