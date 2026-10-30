"""Read only bounded safetensors headers and classify checkpoint payloads."""
import hashlib,json,pathlib,struct,sys
root=pathlib.Path(sys.argv[1]);files=sorted(root.glob('*.safetensors'));assert files
seen=set();locations={};groups={};total_payload=0;shards=[]
for path in files:
 size=path.stat().st_size
 with path.open('rb') as f:
  raw=f.read(8);assert len(raw)==8;n=struct.unpack('<Q',raw)[0];assert 0<n<=64*1024*1024 and n+8<=size;header=f.read(n);assert len(header)==n
 meta=json.loads(header);regions=[];payload=size-8-n
 for name,v in meta.items():
  if name=='__metadata__':continue
  if name in seen:raise ValueError('duplicate tensor '+name)
  seen.add(name);locations[name]=path.name;lo,hi=v['data_offsets'];assert 0<=lo<=hi<=payload;regions.append((lo,hi));total_payload+=hi-lo
  if name.startswith('mtp.') or name.startswith('model.mtp.'):kind='mtp'
  elif '.mlp.experts.' in name:kind='routed_experts'
  elif '.mlp.shared_expert.' in name:kind='shared_expert'
  elif '.mlp.shared_expert_gate.' in name:kind='shared_gate'
  else:kind='other_model'
  g=groups.setdefault(kind,{'tensors':0,'payload_bytes':0,'dtypes':{}});g['tensors']+=1;g['payload_bytes']+=hi-lo;g['dtypes'][v['dtype']]=g['dtypes'].get(v['dtype'],0)+(hi-lo)
 for (_,end),(start,_) in zip(sorted(regions),sorted(regions)[1:]):
  if end>start:raise ValueError('overlapping regions in '+str(path))
 shards.append({'name':path.name,'bytes':size,'header_sha256':hashlib.sha256(header).hexdigest(),'tensors':len(regions),'payload_bytes':payload})
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];assert set(index)==seen;assert index==locations
print(json.dumps({'groups':groups,'shards':shards,'tensors':len(seen),'payload_bytes':total_payload,'limits':'checkpoint payload sizes only; not allocations, packing overhead, peak memory, KV, CPU copies, activations or GPU bandwidth'},sort_keys=True))
