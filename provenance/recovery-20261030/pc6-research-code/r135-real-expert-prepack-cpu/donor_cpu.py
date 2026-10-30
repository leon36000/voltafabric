import ast,base64,hashlib,json,sys
p=json.loads(sys.stdin.buffer.read(4000000));expected={'marlin':'b2e1f7c402d83f133244337fc2352a640e5cb5b67e8279938e2da18f4a7cf147','scales':'98aa1d1823f373ff28fc078a0443c1de7fa45594a9cfee1751e6d3f4cacc85aa','codes':'c1bbdf2d6c0c15d7da34352689d65530b4daf981aa6392bfb17d17010dd7e993'}
if set(p)!=set(expected):raise ValueError('payload set mismatch')
b={k:base64.b64decode(v,validate=True) for k,v in p.items()}
for k,v in b.items():
 if hashlib.sha256(v).hexdigest()!=expected[k]:raise ValueError('payload hash mismatch')
import torch
torch.set_num_threads(1)
if torch.version.cuda is not None:raise ValueError('CPU-only torch required')
tree=ast.parse(b['marlin']);f=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=='_qpn_prepack']
if len(f)!=1:raise ValueError('function missing or ambiguous')
code=ast.fix_missing_locations(ast.Module(body=f,type_ignores=[]));env={'torch':torch};exec(compile(code,'<pinned-marlin-qpn>','exec'),env);out=[]
for i,(rows,cols) in enumerate([(2048,512),(512,2048),(512,2048)]):
 c=torch.frombuffer(bytearray(b['codes'][i*524288:(i+1)*524288]),dtype=torch.uint8).reshape(rows,cols//2);s=torch.frombuffer(bytearray(b['scales'][i*65536:(i+1)*65536]),dtype=torch.uint8).reshape(rows,cols//16);qc,qs=env['_qpn_prepack'](c,s)
 if qc is None or qs is None:raise ValueError('donor rejected shape')
 cb=bytes(qc.tolist());sb=bytes(qs.tolist());out.append({'codes':base64.b64encode(cb).decode(),'scales':base64.b64encode(sb).decode(),'codes_sha256':hashlib.sha256(cb).hexdigest(),'scales_sha256':hashlib.sha256(sb).hexdigest(),'rows':rows,'cols':cols})
print(json.dumps({'status':'DONOR_QPN_PREPACK_CPU_EXECUTED','planes':out,'torch':torch.__version__,'gpu':False}))
