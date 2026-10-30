from pathlib import Path
import hashlib,json,math,struct,subprocess
b=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');p=b/'.runs/r136-real-expert-dequant';s=b/'.runs/r132-qwen-expert-sample';cli=b/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';h=lambda x:hashlib.sha256(x).hexdigest()
assert h((p/'plan.json').read_bytes())=='448a51b42a8629284df5d7b9f836ec148225f4172b09e20befc996861778abc1'
assert h(cli.read_bytes())=='11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4'
c=(s/'codes.bin').read_bytes();ss=(s/'block_scales.bin').read_bytes();gs=(s/'scalars.bin').read_bytes()
assert h(c)=='c1bbdf2d6c0c15d7da34352689d65530b4daf981aa6392bfb17d17010dd7e993' and h(ss)=='98aa1d1823f373ff28fc078a0443c1de7fa45594a9cfee1751e6d3f4cacc85aa' and h(gs)=='bcf1e13241cb2960c903be775ca03e1a2776d3cc560b394e92137a7fb7b715b2'
factors=struct.unpack('<6f',gs);results=[]
def fp4(v):
 m=v&7;x=0.5*m if m<2 else math.ldexp(1.0+0.5*(m&1),(m>>1)-1)
 return -x if v&8 else x
def scale(v):
 if v&128 or v==127:raise ValueError('invalid scale')
 e=(v>>3)&15;m=v&7
 return math.ldexp(m,-9) if e==0 else math.ldexp(8+m,e-10)
def call(name,obj):
 raw=json.dumps(obj,separators=(',',':')).encode();r=subprocess.run([str(cli),name,'-'],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20);return r,json.loads(r.stdout)
for i,(name,cols,fi) in enumerate([('down',512,1),('gate',2048,3),('up',2048,5)]):
 fullc=c[i*524288:(i+1)*524288];fulls=ss[i*65536:(i+1)*65536];smallc=b''.join(fullc[r*(cols//2):r*(cols//2)+32] for r in range(32));smalls=b''.join(fulls[r*(cols//16):r*(cols//16)+4] for r in range(32));plane={'rows':32,'columns':64,'codes':list(smallc),'scales':list(smalls),'global_dequant':factors[fi]}
 rc,packed=call('qpn2-pack',plane)
 if rc.returncode or packed.get('ok') is not True:raise ValueError('pack failed')
 rd,decoded=call('qpn2-dequant',packed['result'])
 if rd.returncode or decoded.get('ok') is not True:raise ValueError('dequant failed')
 mat=decoded['result'];assert len(mat)==32 and all(len(row)==64 for row in mat);mismatch=0;nonzero=0
 for row in range(32):
  for col in range(64):
   byte=smallc[row*32+col//2];n=(byte>>4 if col&1 else byte&15);want=fp4(n)*scale(smalls[row*4+col//16])*factors[fi];actual=mat[row][col];nonzero+=want!=0.0
   if struct.pack('<d',want)!=struct.pack('<d',actual):mismatch+=1
 result={'projection':name,'values':2048,'nonzero_values':nonzero,'bit_mismatches':mismatch,'pack_stdout_sha256':h(rc.stdout),'dequant_stdout_sha256':h(rd.stdout)};results.append(result)
 if mismatch:raise ValueError('mismatch: '+name)
out={'status':'REAL_EXPERT_SUBTILES_RUST_PYTHON_DEQUANT_F64_BIT_EXACT','projections':results,'values':6144,'gpu':False,'limitations':'Three 32x64 slices only, ideal binary64 decode, no kernel FP16 intermediate/forward, model or full shard LFS hash.'};(p/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
