"""Compare opt-in native reference against the previously captured corpus."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
host=sys.argv[1]; assert host in ('pc1','pc6')
root=Path('/home/pc1/NAOS') if host=='pc1' else Path(__file__).resolve().parents[2]/'.runs'
work=root/'vf-r592-work';prev=root/'vf-r591-work'; prior=root/'vf-r590-work'
plan=json.loads((work/'PLAN.json').read_bytes()); manifest=(work/'INPUTS.json').read_bytes();records=json.loads(manifest)
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(Path(__file__).read_bytes())==plan['runner_sha256']
assert sha(manifest)==plan['inputs_sha256']
binary=work/'check-reference';assert sha(binary.read_bytes())==plan[host+'_binary_sha256']
experts={12:prior/'expert12.bin',75:prior/'expert75.bin',255:(root/'vf-r575-work/expert255.bin' if host=='pc1' else root/'vf-r574-input/expert255.bin')}
for n,path in experts.items():assert sha(path.read_bytes())==plan['expert_sha256'][str(n)]
fd=os.open(work/'SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.write(fd,(sha((work/'PLAN.json').read_bytes())+'\n').encode());os.fsync(fd);os.close(fd)
for item in records:
    activation=prev/'inputs'/item['file'];assert sha(activation.read_bytes())==item['sha256']
    for n,blob in experts.items():
        capture=prev/'results'/f"{item['file'][:-4]}-e{n}.bin"
        subprocess.run([str(binary),str(blob),str(activation),str(capture)],check=True,timeout=30)
print('PASS reference matches',len(records)*len(experts),'full f64 rows')
