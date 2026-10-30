# SPDX-License-Identifier: Apache-2.0
"""Extract a bounded selection only after hashing a quiescent safetensors shard.

No format conversion; labels and tensor values are copied byte-for-byte. The shard
must already be a complete, immutable regular file, not a live download or symlink.
"""
import argparse, hashlib, json, os, pathlib, stat, struct

MAX_HEADER = 16 * 1024 * 1024
MAX_SELECTED = 512 * 1024 * 1024

def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError('duplicate JSON key')
        out[key] = value
    return out

def _stable(st): return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

def extract(source, expected_sha256, destination, experts, layer=0):
    if len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('expected lowercase SHA256 required')
    if not experts or len(experts)>256 or len(set(experts))!=len(experts) or any(type(e)!=int or e<0 for e in experts):
        raise ValueError('invalid expert selection')
    source=pathlib.Path(source); destination=pathlib.Path(destination)
    fd=os.open(source,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    try:
        initial=os.fstat(fd)
        if not stat.S_ISREG(initial.st_mode): raise ValueError('source must be regular')
        with os.fdopen(fd,'rb',closefd=False) as f:
            digest=hashlib.sha256()
            while chunk:=f.read(4*1024*1024): digest.update(chunk)
            if digest.hexdigest()!=expected_sha256: raise ValueError('source SHA256 mismatch')
            if _stable(initial)!=_stable(os.fstat(fd)): raise ValueError('source changed during hash')
            f.seek(0); b=f.read(8)
            if len(b)!=8: raise ValueError('truncated header length')
            hlen=struct.unpack('<Q',b)[0]
            if not 0<hlen<=MAX_HEADER or 8+hlen>initial.st_size: raise ValueError('invalid header size')
            header=f.read(hlen)
            if len(header)!=hlen: raise ValueError('truncated header')
            meta=json.loads(header,object_pairs_hook=_unique)
            names=[f'model.language_model.layers.{layer}.mlp.experts.{e}.{plane}.{field}' for e in experts for plane in ('gate_proj','up_proj','down_proj') for field in ('weight','weight_scale','weight_scale_2','input_scale')]
            source_spans=[]
            for name in names:
                entry=meta[name]; start,end=entry['data_offsets']
                if type(start)!=int or type(end)!=int or not 0<=start<=end<=initial.st_size-8-hlen: raise ValueError('invalid tensor offsets')
                source_spans.append((start,end))
            ordered=sorted(source_spans)
            if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])): raise ValueError('overlapping tensors')
            total=sum(end-start for start,end in source_spans)
            if total>MAX_SELECTED: raise ValueError('selection exceeds byte cap')
            outmeta={'__metadata__':meta.get('__metadata__',{})}; offset=0
            for name,(start,end) in zip(names,source_spans):
                ent=meta[name];outmeta[name]={'dtype':ent['dtype'],'shape':ent['shape'],'data_offsets':[offset,offset+end-start]};offset+=end-start
            raw=json.dumps(outmeta,separators=(',',':'),ensure_ascii=False).encode();raw+=b' '*((-len(raw))%8)
            with destination.open('xb') as out:
                out.write(struct.pack('<Q',len(raw)));out.write(raw)
                for start,end in source_spans:
                    f.seek(8+hlen+start);remaining=end-start
                    while remaining:
                        chunk=f.read(min(1024*1024,remaining))
                        if not chunk:raise ValueError('truncated tensor')
                        out.write(chunk);remaining-=len(chunk)
                out.flush();os.fsync(out.fileno())
            if _stable(initial)!=_stable(os.fstat(fd)): raise ValueError('source changed during extraction; discard result')
        digest=hashlib.sha256()
        with destination.open('rb') as f:
            while chunk:=f.read(4*1024*1024):digest.update(chunk)
        return {'source_sha256':expected_sha256,'selected_sha256':digest.hexdigest(),'selected_bytes':destination.stat().st_size,'tensor_bytes':total,'tensor_count':len(names),'experts':experts,'layer':layer}
    finally: os.close(fd)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--experts',required=True);p.add_argument('--layer',type=int,default=0);a=p.parse_args();print(json.dumps(extract(a.source,a.sha256,a.output,[int(x) for x in a.experts.split(',')],a.layer),sort_keys=True))
if __name__=='__main__':main()
