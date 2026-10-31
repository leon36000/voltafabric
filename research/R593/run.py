"""One-shot bounded scalar capture of observed format pairs."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
host=sys.argv[1];assert host in ('pc1','pc6')
root=Path('/home/pc1/NAOS') if host=='pc1' else Path(__file__).resolve().parents[2]/'.runs'
work=root/'vf-r593-work';inputs=root/'vf-r591-work/inputs';p=json.loads((work/'PLAN.json').read_bytes());cases_raw=(work/'CASES.json').read_bytes();cases=json.loads(cases_raw)
h=lambda b:hashlib.sha256(b).hexdigest()
assert h(Path(__file__).read_bytes())==p['runner_sha256'] and h(cases_raw)==p['cases_sha256']
binary=work/'capture-reference';assert h(binary.read_bytes())==p[host+'_binary_sha256']
for c in cases:
 assert h((work/'experts'/('layer'+str(c['layer'])+'.bin')).read_bytes())==c['sha256']
for name,digest in p['inputs'].items():assert h((inputs/name).read_bytes())==digest
assert not any((work/'results').iterdir())
fd=os.open(work/'SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.write(fd,(h((work/'PLAN.json').read_bytes())+'\n').encode());os.fsync(fd);os.close(fd)
for c in cases:
 for name in p['inputs']:
  output=work/'results'/(f"l{c['layer']}-"+name[:-4]+'.bin')
  subprocess.run([str(binary),str(c['gate_up']),str(c['down']),str(work/'experts'/('layer'+str(c['layer'])+'.bin')),str(inputs/name),str(output)],check=True,timeout=30)
print('PASS captures',len(cases)*len(p['inputs']))
