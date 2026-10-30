"""Actual R53 utility and no-worker coordinator gate with synthetic private files."""
import fcntl,hashlib,importlib.util,json,os,pathlib,subprocess
p=pathlib.Path('/home/pc1/NAOS/vf-r529-work'); pack=pathlib.Path('/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s')
worker='/home/pc1/NAOS/vf-r525-work/target/release/vf-rank-session'; binary=str(p/'target/release/vf-sc117-prefill-greedy'); manifest='/home/pc1/NAOS/vf-r443-work/manifest.json'
root=p/'synthetic-r53'; root.mkdir(mode=0o700); ledger=p/'native-ledger'; ledger.mkdir(mode=0o700)
lease_path=p/'synthetic-shared-lease'; fd=os.open(lease_path,os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_CLOEXEC,0o600)
try:
 m=os.fstat(fd); lease={'path':str(lease_path),'dev':m.st_dev,'ino':m.st_ino,'uid':m.st_uid}
 m=root.stat(); ns={'root':str(root),'dev':m.st_dev,'ino':m.st_ino,'uid':m.st_uid,'base_id':'r53-r529-integration-a'}
 keys=(p/'known-hosts').read_bytes(); kh=hashlib.sha256(keys).hexdigest()
 env=os.environ.copy()
 for key in list(env):
  if key.startswith('VF_RANK') or key.startswith('VF_SHARED_LEASE') or key.startswith('VF_R53_') or key=='VF_EXPECTED_PLAN_ID': env.pop(key)
 env.update(VF_CPU_EXPERT_BUDGET='3221225472',VF_RANK2_BUDGET='1073741824',VF_RANK3_BUDGET='2147483648',VF_LOCAL_WORKER_SHA256='49216f6ce76a637be323dd0aa88c69e7788b4fa8ff13f5663ee103c5e4a22409',VF_RANK3_SSH_HOST='pc1-lan',VF_RANK3_SSH_WORKER='/opt/vf-worker',VF_RANK3_SSH_WORKER_SHA256='a'*64,VF_RANK3_SSH_KNOWN_HOSTS=str(p/'known-hosts'),VF_RANK3_SSH_KNOWN_HOSTS_SHA256=kh,VF_SHARED_LEASE_SPEC=json.dumps(lease,separators=(',',':')),VF_R53_NAMESPACE_SPEC=json.dumps(ns,separators=(',',':')))
 tokens='248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271'
 args=[binary,str(pack/'dense.bin'),'/dev/null',str(pack/'experts.bin'),str(pack/'native_experts.txt'),worker,str(ledger),'/dev/null',manifest,'48',tokens,'1','248046','chat']
 pre=subprocess.run([binary,'--plan-only',*args[1:]],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30)
 plan=json.loads(pre.stdout)['plan_id']; assert plan.startswith('r529-')
 env.update(VF_EXPECTED_PLAN_ID=plan,VF_SHARED_LEASE_FD=str(fd)); fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def run(): return subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(fd,),timeout=30)
 missing=run(); assert missing.returncode!=0 and not missing.stdout and not list(root.iterdir()) and not list(ledger.iterdir())
 modspec=importlib.util.spec_from_file_location('namespace_ledger',p/'namespace_ledger.py'); module=importlib.util.module_from_spec(modspec); modspec.loader.exec_module(module)
 ids=module.reserve(root,ns['base_id'],{'plan_id':plan,'shared_lease':lease})
 assert ids==[ns['base_id'],ns['base_id']+'-duplicate-slot']
 result=run(); assert result.returncode!=0 and not result.stdout and b'PLE source not regular' in result.stderr,result.stderr
 marker=root/(ns['base_id']+'-sc117-started.json'); mark=json.loads(marker.read_text())
 assert mark=={'status':'RESERVED_UNKNOWN_NO_REPLAY','base_id':ns['base_id'],'plan_id':plan}
 assert not list(ledger.iterdir())
 hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(os.listdir(root))}
 print(json.dumps({'status':'PASS','plan_id':plan,'namespace':ns,'synthetic_lease':lease,'hostkeys_sha256':kh,'records_sha256':hashes,'missing_reservations':'refused','accepted':'consumed_marker_then_ple_type_refusal','native_claims':0},sort_keys=True))
finally: os.close(fd)
