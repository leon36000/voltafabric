import json,subprocess
from pathlib import Path
cli=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli')
def run(name,obj):
 r=subprocess.run([str(cli),name,'-'],input=json.dumps(obj).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10);assert r.returncode==0;o=json.loads(r.stdout);assert o['ok'] is True;return o['result']
def test(first,scale,factor,want):
 codes=[0]*1024;codes[0]=first;scales=[0]*128;scales[0]=scale;x={'rows':32,'columns':64,'codes':codes,'scales':scales,'global_dequant':factor};a=run('qpn2-dequant',run('qpn2-pack',x));assert a[0][:2]==want
for x in [(0x21,0x38,2.0,[1.0,2.0]),(0x12,0x38,2.0,[2.0,1.0]),(0x21,0x40,2.0,[2.0,4.0]),(0x21,0x38,1.0,[0.5,1.0])]:test(*x)
print(json.dumps({'status':'FOUR_MANUAL_SENTINELS_PASS','tests':['low_high_nibble','reversed_nibble','scale_change','global_factor_change']}))
