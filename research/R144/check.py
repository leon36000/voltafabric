import email.parser,hashlib,json,pathlib,re
from packaging.requirements import Requirement
from packaging.version import Version
p=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs'); metadata=(p/'r143-pep658-metadata/1cat.metadata').read_bytes(); donor=(p/'upstream-reuse-integration-r118-20261028/pinned-upstream/scripts__bootstrap-sm70.sh').read_bytes()
assert hashlib.sha256(metadata).hexdigest()=='8c21ab87c8454d229c175deb0b8927f55529bc40f468113fb1f015f0001b0cd9'
m=email.parser.BytesParser().parsebytes(metadata); assert m['Name']=='1cat-vllm' and m['Version']=='1.2.2'
rows=[]
for name in ('tilelang','apache-tvm-ffi'):
 reqs=[Requirement(s) for s in m.get_all('Requires-Dist',[]) if Requirement(s).name.lower().replace('_','-')==name and Requirement(s).marker is None]
 assert len(reqs)==1 and str(reqs[0].specifier)=='==0.1.9'
 s=donor.decode(); assert re.search(r'pip install "tilelang==0\.1\.10" "apache-tvm-ffi==0\.1\.10"',s)
 req=reqs[0]; assert Version('0.1.9') in req.specifier and Version('0.1.10') not in req.specifier
 rows.append({'package':name,'1cat_requirement':str(req),'donor_requested':'==0.1.10','positive_control_0_1_9':True,'donor_version_satisfies_1cat':False})
result={'status':'INCOMPATIBLE_DECLARED_REQUIREMENTS','1cat_version':'1.2.2','metadata_sha256':hashlib.sha256(metadata).hexdigest(),'donor_bootstrap_sha256':hashlib.sha256(donor).hexdigest(),'donor_revision':'b1a2181702e609dc92591343e46c63d1324ade43','rows':rows,'scope':'Exact 1Cat wheel metadata and donor bootstrap only. No claim about other wheel versions or proof of actual runtime incompatibility; no install or GPU.'}
print(json.dumps(result,indent=2,sort_keys=True))
