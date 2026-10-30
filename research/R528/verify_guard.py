"""Synthetic lock inheritance through actual no-worker SC117 preflight on PC1."""
import fcntl,hashlib,json,os,pathlib,subprocess
p=pathlib.Path('/home/pc1/NAOS/vf-r528-work'); pack=pathlib.Path('/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s')
worker='/home/pc1/NAOS/vf-r525-work/target/release/vf-rank-session'; binary=str(p/'target/release/vf-sc117-prefill-greedy')
manifest='/home/pc1/NAOS/vf-r443-work/manifest.json'; ledger=p/'ledger'; ledger.mkdir(exist_ok=True)
assert not list(ledger.iterdir())
lease=p/'synthetic-lease-2'; fd=os.open(lease,os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_CLOEXEC,0o600)
try:
 m=os.fstat(fd); spec={'path':str(lease),'dev':m.st_dev,'ino':m.st_ino,'uid':m.st_uid}
 keys=(p/'known-hosts').read_bytes(); digest=hashlib.sha256(keys).hexdigest()
 env=os.environ.copy()
 for key in list(env):
  if key.startswith('VF_RANK') or key.startswith('VF_SHARED_LEASE') or key=='VF_EXPECTED_PLAN_ID': env.pop(key)
 env.update(VF_CPU_EXPERT_BUDGET='3221225472',VF_RANK2_BUDGET='1073741824',VF_RANK3_BUDGET='2147483648',VF_LOCAL_WORKER_SHA256='49216f6ce76a637be323dd0aa88c69e7788b4fa8ff13f5663ee103c5e4a22409',VF_RANK3_SSH_HOST='pc1-lan',VF_RANK3_SSH_WORKER='/opt/vf-worker',VF_RANK3_SSH_WORKER_SHA256='a'*64,VF_RANK3_SSH_KNOWN_HOSTS=str(p/'known-hosts'),VF_RANK3_SSH_KNOWN_HOSTS_SHA256=digest,VF_SHARED_LEASE_SPEC=json.dumps(spec,separators=(',',':')))
 tokens='248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271'
 # /dev/null deliberately fails PLE source-size validation after the lease gate.
 args=[binary,str(pack/'dense.bin'),'/dev/null',str(pack/'experts.bin'),str(pack/'native_experts.txt'),worker,str(ledger),'/dev/null',manifest,'48',tokens,'1','248046','chat']
 pre=subprocess.run([binary,'--plan-only',*args[1:]],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30)
 plan=json.loads(pre.stdout)['plan_id']; assert plan.startswith('r528-')
 env['VF_EXPECTED_PLAN_ID']=plan
 def run(lease_fd):
  e=dict(env,VF_SHARED_LEASE_FD=str(lease_fd))
  return subprocess.run(args,env=e,stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(lease_fd,),timeout=30)
 unlocked=run(fd); assert unlocked.returncode!=0 and not unlocked.stdout and b'exclusive shared lease flock required' in unlocked.stderr, unlocked.stderr
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 other=os.open(lease,os.O_RDWR|os.O_CLOEXEC)
 try:
  wrong=run(other); assert wrong.returncode!=0 and not wrong.stdout and b'exclusive shared lease flock required' in wrong.stderr,wrong.stderr
 finally: os.close(other)
 held=run(fd); assert held.returncode!=0 and not held.stdout and b'PLE source not regular' in held.stderr,held.stderr
 assert not list(ledger.iterdir())
 print(json.dumps({'status':'PASS','plan_id':plan,'synthetic_lease':spec,'hostkeys_sha256':digest,'unlocked':'refused','different_description':'refused','locked':'passed_gate_then_ple_type_refusal','ledger_claims':0},sort_keys=True))
finally:
 os.close(fd)
