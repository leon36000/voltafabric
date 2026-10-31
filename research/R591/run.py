"""One-shot, CPU-only scalar capture; preserves unique outputs for each host."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
host=sys.argv[1]
assert host in ('pc1','pc6')
work=Path('/home/pc1/NAOS/vf-r591-work') if host=='pc1' else Path(__file__).resolve().parents[2]/'.runs/vf-r591-work'
prior=Path('/home/pc1/NAOS/vf-r590-work') if host=='pc1' else Path(__file__).resolve().parents[2]/'.runs/vf-r590-work'
plan=json.loads((work/'PLAN.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(Path(__file__).read_bytes()) == plan['runner_sha256']
binary=prior/'scalar-capture'
assert sha(binary.read_bytes()) == plan[host+'_binary_sha256']
inputs=json.loads((work/'INPUTS.json').read_bytes())
assert sha((work/'INPUTS.json').read_bytes())==plan['input_manifest_sha256']
experts={12:prior/'expert12.bin',75:prior/'expert75.bin',255:(Path('/home/pc1/NAOS/vf-r575-work/expert255.bin') if host=='pc1' else Path(__file__).resolve().parents[2]/'.runs/vf-r574-input/expert255.bin')}
for record in inputs:
    assert sha((work/'inputs'/record['file']).read_bytes())==record['sha256']
for expert,path in experts.items():
    assert sha(path.read_bytes())==plan['expert_sha256'][str(expert)]
assert not any((work/'results').iterdir())
fd=os.open(work/'SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
os.write(fd,(sha((work/'PLAN.json').read_bytes())+'\n').encode());os.fsync(fd);os.close(fd)
fd=os.open(work,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
for record in inputs:
    for expert,path in experts.items():
        dest=work/'results'/f"{record['file'][:-4]}-e{expert}.bin"
        subprocess.run([str(binary),str(path),str(work/'inputs'/record['file']),str(dest)],check=True,timeout=30)
print('completed captures',len(inputs)*len(experts))
