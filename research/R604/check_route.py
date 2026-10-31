"""Read-only check of the retained R428 result on PC1; print no model output."""
import json,hashlib,os
p='/home/pc1/NAOS/vf-r428-work/result.json'
with open(p,'rb') as f:
 a=os.fstat(f.fileno());raw=f.read();b=os.fstat(f.fileno())
identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
assert identity(a)==identity(b)
digest=hashlib.sha256(raw).hexdigest();assert digest=='31800dc58ccac050049ce417dd0bdb61143e3d022c8ab9cae210f02ccfec8cd4'
v=json.loads(raw)['tokens'][0];ids=[75,12,209,181,257,272,480,159,492,255];weights=[0.3420320749282837,0.1739722639322281,0.09913824498653412,0.08050233870744705,0.06733449548482895,0.061283160001039505,0.05222782492637634,0.042672742158174515,0.04069479927420616,0.04014204069972038]
assert v['ids']==ids and v['weights']==weights and v['activation_sha']=='c8a0744a87fade2652acfb34c802a49354379f558791bdf63e53f1f124bcc88a'
print(json.dumps({'status':'PASS','result_sha256':digest,'activation_sha256':v['activation_sha'],'route_count':len(ids),'ids_and_coefficients_exact':True}))
