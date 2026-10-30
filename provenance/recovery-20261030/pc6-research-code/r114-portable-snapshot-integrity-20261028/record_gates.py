from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent;W=R.parent/'gitlab-import-r113-20261028/repository'
names=['scripts/check_gitlab_snapshot.py','tests/python/test_gitlab_snapshot_integrity.py','docs/GITLAB_SNAPSHOT_INTEGRITY.md']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={n:digest(W/n) for n in names}
commands=[['python3','-B','-m','unittest','-v','tests.python.test_gitlab_snapshot_integrity'],['python3','-B','scripts/check_gitlab_snapshot.py']];rows=[]
for i,argv in enumerate(commands):
 start=time.monotonic_ns();p=subprocess.run(argv,cwd=W,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,check=False)
 if len(p.stdout)+len(p.stderr)>1024*1024:raise ValueError('output cap')
 for label,raw in [('stdout',p.stdout),('stderr',p.stderr)]:
  with (R/f'{i}-{label}.log').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 rows.append(dict(argv=argv,cwd=str(W),exit_code=p.returncode,stdout_sha256=hashlib.sha256(p.stdout).hexdigest(),stderr_sha256=hashlib.sha256(p.stderr).hexdigest(),elapsed_ns=time.monotonic_ns()-start))
 if p.returncode:break
after={n:digest(W/n) for n in names}
result=dict(status='R114_CPU_CHECKS_PASS' if len(rows)==2 and all(r['exit_code']==0 for r in rows) and before==after else 'R114_CPU_CHECKS_FAIL',source_sha256=before,sources_unchanged=before==after,commands=rows,gpu=False,network=False,full_cpu_suite=False)
with (R/'CHECKS.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(result));raise SystemExit(0 if result['status']=='R114_CPU_CHECKS_PASS' else 1)
