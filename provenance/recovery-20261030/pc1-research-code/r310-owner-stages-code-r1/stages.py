"""CPU-only, bounded stage samples for one existing QPN2 resident artifact."""
import gc,json,os,pathlib,sys,torch
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs')
root=base/'r310-owner-stages-r1';root.mkdir(mode=0o700)
from adapters.resource_admission import ResourcePolicy
from adapters.persistent_gpu_owner import PersistentQPN2Owner
from scripts.resident_artifact import load_stack
from owner_service import Provider
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and not torch.cuda.is_initialized();torch.set_num_threads(1)
policy=ResourcePolicy(8*1024**3,6*1024**3,5*1024**3,1*1024**3);admission=policy.check(base)
def sample(stage):
 result={'stage':stage}
 for line in pathlib.Path('/proc/self/status').read_text().splitlines():
  fields=line.split()
  if fields and fields[0] in ('VmRSS:','VmHWM:','RssAnon:','RssFile:','VmSwap:'):
   result[fields[0][:-1]+'_kib']=int(fields[1])
 for line in pathlib.Path('/proc/self/smaps_rollup').read_text().splitlines():
  fields=line.split()
  if fields and fields[0] in ('Pss:','Pss_Anon:','Pss_File:','Swap:'):
   result[fields[0][:-1]+'_kib']=int(fields[1])
 return result
manifest=json.loads((base/'r283-four-layer-owners-r1/layer0/manifest.json').read_text())['ranks']['0'];path=pathlib.Path(manifest['path']);sha=manifest['sha256'];samples=[sample('imports')]
artifact=load_stack(path,expected_sha256=sha);info=artifact.metadata;size=sum(t.numel()*t.element_size() for t in artifact.tensors.values());samples.append(sample('validated_artifact_retained'))
del artifact;gc.collect();samples.append(sample('validation_released'))
mapping=torch.full((256,),-1,dtype=torch.int64)
for physical,expert in enumerate(info['physical_experts']):mapping[expert['id']]=physical
owner=PersistentQPN2Owner.load(artifact_path=path,expected_info=info,expected_sha256=sha,provider=Provider(),expert_map=mapping,device='cpu');samples.append(sample('owner_ready'))
assert owner.state=='READY';owner.close();del owner;gc.collect();samples.append(sample('owner_closed'))
assert not torch.cuda.is_initialized()
result={'status':'PASS','artifact_sha256':sha,'artifact_bytes':path.stat().st_size,'tensor_bytes':size,'admission':admission,'samples':samples,'gpu':False,'scope':'single isolated CPU process; stage-boundary samples, not continuous peaks'}
(root/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result),flush=True)
