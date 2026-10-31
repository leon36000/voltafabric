"""Read-only coverage check against a frozen R53 snapshot; never grants a lease."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

MAX_FILES=10000
MAX_BYTES=65536
SAFE_NAME=re.compile(r'r53-[A-Za-z0-9_-]{8,100}\.json\Z')
STATUSES={'RESERVED_UNKNOWN_NO_REPLAY','CONSUMED_RETROSPECTIVE_CPU_NO_REPLAY'}

def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('duplicate JSON key')
        result[key]=value
    return result

def _read_dir(path,private):
    flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC
    try: fd=os.open(path,flags)
    except OSError as e: raise ValueError('cannot safely open directory') from e
    try:
        meta=os.fstat(fd)
        try: path_meta=os.stat(path,follow_symlinks=False)
        except OSError as e: raise ValueError('directory path changed') from e
        if (path_meta.st_dev,path_meta.st_ino)!=(meta.st_dev,meta.st_ino):raise ValueError('directory path changed')
        if not stat.S_ISDIR(meta.st_mode) or (private and (meta.st_uid!=os.geteuid() or meta.st_mode&0o077)):
            raise ValueError('invalid directory owner or permissions')
        names=sorted(os.listdir(fd))
        if len(names)>MAX_FILES:raise ValueError('too many entries')
        entries={}
        for name in names:
            if SAFE_NAME.fullmatch(name) is None: raise ValueError('unexpected entry name')
            try: child=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,dir_fd=fd)
            except OSError as e: raise ValueError('unsafe entry') from e
            try:
                before=os.fstat(child)
                if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or not 0<before.st_size<=MAX_BYTES
                    or (private and (before.st_uid!=os.geteuid() or before.st_mode&0o077))):
                    raise ValueError('invalid entry identity or permissions')
                chunks=[];remaining=before.st_size
                while remaining:
                    part=os.read(child,remaining)
                    if not part:raise ValueError('short entry')
                    chunks.append(part);remaining-=len(part)
                after=os.fstat(child)
                key=lambda m:(m.st_dev,m.st_ino,m.st_size,m.st_mtime_ns,m.st_ctime_ns)
                if key(before)!=key(after):raise ValueError('entry changed during read')
                entries[name]=b''.join(chunks)
            finally:os.close(child)
        end=os.fstat(fd)
        if (meta.st_dev,meta.st_ino,meta.st_mtime_ns,meta.st_ctime_ns)!=(end.st_dev,end.st_ino,end.st_mtime_ns,end.st_ctime_ns):
            raise ValueError('directory changed during read')
        try: path_end=os.stat(path,follow_symlinks=False)
        except OSError as e: raise ValueError('directory path changed') from e
        if (path_end.st_dev,path_end.st_ino)!=(meta.st_dev,meta.st_ino):raise ValueError('directory path changed')
        return entries, {'device':meta.st_dev,'inode':meta.st_ino,'uid':meta.st_uid}
    finally:os.close(fd)

def audit(snapshot,candidate,expected_snapshot_sha256):
    frozen,_=_read_dir(Path(snapshot),False)
    if not frozen:raise ValueError('empty snapshot')
    for name,raw in frozen.items():
        try: value=json.loads(raw,object_pairs_hook=_unique)
        except (ValueError,UnicodeDecodeError) as e:raise ValueError('invalid snapshot record') from e
        if not isinstance(value,dict):raise ValueError('invalid snapshot record')
        effective=name[:-5];base=value.get('base_id')
        if (value.get('status') not in STATUSES or value.get('effective_run_id')!=effective
            or not isinstance(base,str) or effective not in (base,base+'-duplicate-slot')):
            raise ValueError('snapshot identity mismatch')
    manifest=hashlib.sha256()
    for name in sorted(frozen):manifest.update(name.encode()+b'\0'+hashlib.sha256(frozen[name]).digest())
    if (not isinstance(expected_snapshot_sha256,str) or re.fullmatch(r'[0-9a-f]{64}',expected_snapshot_sha256) is None
            or manifest.hexdigest()!=expected_snapshot_sha256):
        raise ValueError('snapshot manifest mismatch')
    current,identity=_read_dir(Path(candidate),True)
    missing=sorted(frozen.keys()-current.keys())
    changed=sorted(name for name in frozen.keys()&current.keys() if frozen[name]!=current[name])
    extra=sorted(current.keys()-frozen.keys())
    return {'snapshot_records':len(frozen),'candidate_records':len(current),
            'snapshot_manifest_sha256':manifest.hexdigest(),'candidate_identity':identity,'missing':missing,'changed':changed,'additional':extra,
            'snapshot_covered':not missing and not changed,'live_ledger_qualified':False,
            'scope':'Frozen-byte coverage only; reconcile all hosts, receipts and shared lease independently.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot');parser.add_argument('candidate');parser.add_argument('--expected-snapshot-sha256',required=True);args=parser.parse_args()
    try:result=audit(args.snapshot,args.candidate,args.expected_snapshot_sha256)
    except ValueError as error:parser.exit(2,str(error)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result['snapshot_covered'] else 1
if __name__=='__main__':raise SystemExit(main())
