from pathlib import Path
import hashlib,json,subprocess,sys
b=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');p=b/'.runs/r135-real-expert-prepack-cpu';s=b/'.runs/r132-qwen-expert-sample';cli=b/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';h=lambda x:hashlib.sha256(x).hexdigest()
if h(cli.read_bytes())!='11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4':raise ValueError('cli mismatch')
c=(s/'codes.bin').read_bytes();sc=(s/'block_scales.bin').read_bytes()
if h(c)!='c1bbdf2d6c0c15d7da34352689d65530b4daf981aa6392bfb17d17010dd7e993' or h(sc)!='98aa1d1823f373ff28fc078a0443c1de7fa45594a9cfee1751e6d3f4cacc85aa':raise ValueError('sample mismatch')
shapes=[(2048,512),(512,2048),(512,2048)];result=[]
for i,(rows,cols) in enumerate(shapes):
 inp={'rows':rows,'columns':cols,'codes':list(c[i*524288:(i+1)*524288]),'scales':list(sc[i*65536:(i+1)*65536]),'global_dequant':1.0};raw=json.dumps(inp,separators=(',',':')).encode();(p/(str(i)+'-rust-intent.json')).write_text(json.dumps({'input_sha256':h(raw),'cli_sha256':h(cli.read_bytes()),'gpu':False})+'\n')
 try:r=subprocess.run([str(cli),'qpn2-pack','-'],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
 except subprocess.TimeoutExpired:(p/(str(i)+'-rust-receipt.json')).write_text('{"status":"UNKNOWN_NO_RETRY"}\n');raise
 (p/(str(i)+'-rust-receipt.json')).write_text(json.dumps({'exit':r.returncode,'stdout_sha256':h(r.stdout),'stderr_sha256':h(r.stderr)})+'\n')
 if r.returncode or len(r.stdout)>12000000:raise ValueError('rust process failure or output cap')
 obj=json.loads(r.stdout)
 if obj.get('ok') is not True:raise ValueError('rust not ok')
 rc=bytes(obj['result']['codes']);rs=bytes(obj['result']['scales']);dc=(p/(str(i)+'-donor-codes.bin')).read_bytes();ds=(p/(str(i)+'-donor-scales.bin')).read_bytes()
 if rc!=dc or rs!=ds:raise ValueError('packing mismatch at projection '+str(i))
 result.append({'projection':['down','gate','up'][i],'rows':rows,'cols':cols,'codes_bytes':len(rc),'scale_bytes':len(rs),'codes_sha256':h(rc),'scale_sha256':h(rs),'equal':True})
out={'status':'DONOR_TORCH_CPU_AND_RUST_QPN2_PACK_BIT_EXACT','projections':result,'gpu':False,'scope':'Actual donor Torch CPU permutation vs actual historical Rust CLI; not CUDA or numerical forward, complete shard SHA unverified.'};(p/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
