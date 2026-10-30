"""Private fixed-profile tar packaging, no model execution or network backend.

Linux cooperative same-UID files. Never extracts an archive pathname: only the
pinned manifest's regular files are written exclusively into a private sibling.
Publication uses renameat2(RENAME_NOREPLACE). Pre-publication failure leaves an
identified partial sibling. Post-publication failure is explicit UNKNOWN with the
owned final preserved. No retry, auto-cleanup or lease creation.
"""
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tarfile

ROOT=Path(__file__).resolve().parent
R61=ROOT.parent/'gpu-rdma-case-contract-r61'
R61_SEAL='679574b2e3fedcf88ae4dfee25bdd21e2ef31d8b19698db880ec831739d32a85'
R62_SEAL='146f545c1385e41e2437bc6800f0ff23d6d70afe24312495eadd01732a325d37'
SOURCE_SHA='41f179ce7d7076748e907e566182aa1d3298b4deb776cb52ecd0f694edba526d'
MAX_BYTES=20*1024*1024
ADAPTERS=['case_contract.py','client_endpoint.py','worker_exec.py','remote_entry.py','guards/lease_contract.py','guards/rping_guard.py','guards/server_group.py']
ROLE_FILES={'client':['bin/aarch64/vf_cli','bin/aarch64/rdma_bridge','fixtures/preweighted_mxfp4_tiny.safetensors','fixtures/preweighted_mxfp4_tiny_golden.json'],'server':['bin/x86_64/worker_release','bin/x86_64/rdma_bridge','bin/x86_64/libvf_r13.so']}


def need(value,message):
    if not value:raise ValueError(message)


def no_duplicates(pairs):
    d={}
    for k,v in pairs:
        need(k not in d,'duplicate manifest field');d[k]=v
    return d


def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def sha_stream(stream,limit):
    h=hashlib.sha256();total=0
    while True:
        chunk=stream.read(min(65536,limit-total+1))
        if not chunk:break
        total+=len(chunk);need(total<=limit,'file size cap');h.update(chunk)
    return h.hexdigest(),total


def assets():
    manifest=(R61/'snapshot/MANIFEST.json').read_bytes()
    need(hashlib.sha256(manifest).hexdigest()==R61_SEAL,'frozen R61 manifest changed')
    body=(R61/'snapshot/ASSETS.json').read_bytes()
    need(hashlib.sha256(body).hexdigest()==json.loads(manifest)['files']['ASSETS.json'],'frozen assets changed')
    return json.loads(body)['files']


def expected_manifest(role):
    need(role in ROLE_FILES,"unknown package role")
    return json.loads('{"client":{"architecture":"aarch64","elf_class":64,"endian":"little","files":{"bin/aarch64/rdma_bridge":{"bytes":72352,"mode":320,"sha256":"5d02b20c3cc10c4768f4008eb4e78aef349c5ca2515cf6fc4f4712e31224669c","source":"bin/aarch64/rdma_bridge"},"bin/aarch64/vf_cli":{"bytes":6047808,"mode":320,"sha256":"11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4","source":"bin/aarch64/vf_cli"},"case_contract.py":{"bytes":7077,"mode":320,"sha256":"5d55a7e85f8bcca183201d5c4b0279eaae48687a2d2c06a98f02edcfb97ccba1","source":"adapter/case_contract.py"},"client_endpoint.py":{"bytes":1032,"mode":320,"sha256":"c4f75f7730e48fab997e9e632174bec1e3b69489eb413c4182b25be950421506","source":"adapter/client_endpoint.py"},"fixtures/preweighted_mxfp4_tiny.safetensors":{"bytes":6688,"mode":256,"sha256":"f0bfbf4220713b7aae0200e15f5232d118f0abfc65d7c322a6540a837fb279a7","source":"fixtures/preweighted_mxfp4_tiny.safetensors"},"fixtures/preweighted_mxfp4_tiny_golden.json":{"bytes":327132,"mode":256,"sha256":"51e4c60545c4f29dcbe03d57f017e522095ec2d37e19f67d2fe890670227d1ed","source":"fixtures/preweighted_mxfp4_tiny_golden.json"},"guards/lease_contract.py":{"bytes":6117,"mode":320,"sha256":"d8254aef7fa2d8d528ccf5da027fb04f3e4f850a6f94fcab859f5dfe237b8a11","source":"adapter/guards/lease_contract.py"},"guards/rping_guard.py":{"bytes":7580,"mode":320,"sha256":"9943d2e66db276fb311b6f8d6577f20096a1863ed8d2d23063014d8240e50876","source":"adapter/guards/rping_guard.py"},"guards/server_group.py":{"bytes":6223,"mode":320,"sha256":"7ebb1a6ce46abce27b895d6428b417614427c3aa5f5d11a8918029225704b0bf","source":"adapter/guards/server_group.py"},"remote_entry.py":{"bytes":3653,"mode":320,"sha256":"15be60f3a72034c38212988ca4ede84965679f8204898345c128c95aa7977053","source":"adapter/remote_entry.py"},"worker_exec.py":{"bytes":960,"mode":320,"sha256":"593674a54ce69e42febf2c800fa93c480da510f00ea24e6daa4ead26d340d8b1","source":"adapter/worker_exec.py"}},"no_model_submitted":true,"role":"client","schema_version":1,"trust":{"R61":"679574b2e3fedcf88ae4dfee25bdd21e2ef31d8b19698db880ec831739d32a85","R62":"146f545c1385e41e2437bc6800f0ff23d6d70afe24312495eadd01732a325d37","canonical_R59":"41f179ce7d7076748e907e566182aa1d3298b4deb776cb52ecd0f694edba526d"}},"server":{"architecture":"x86_64","elf_class":64,"endian":"little","files":{"bin/x86_64/libvf_r13.so":{"bytes":1379016,"mode":256,"sha256":"f856c9bdf5c4d021203b2b7ce5b1f393058370e3f5d2d0f05072fb067e739213","source":"bin/x86_64/libvf_r13.so"},"bin/x86_64/rdma_bridge":{"bytes":21976,"mode":320,"sha256":"de59d04b9226652b53718125959be972da81e79c7c353c829709307c5d578230","source":"bin/x86_64/rdma_bridge"},"bin/x86_64/worker_release":{"bytes":1916560,"mode":320,"sha256":"0f2c51dba34bf0e42cdf2740ce7e34d5a87c433de0df6afa6271151e6c55528a","source":"bin/x86_64/worker_release"},"case_contract.py":{"bytes":7077,"mode":320,"sha256":"5d55a7e85f8bcca183201d5c4b0279eaae48687a2d2c06a98f02edcfb97ccba1","source":"adapter/case_contract.py"},"client_endpoint.py":{"bytes":1032,"mode":320,"sha256":"c4f75f7730e48fab997e9e632174bec1e3b69489eb413c4182b25be950421506","source":"adapter/client_endpoint.py"},"guards/lease_contract.py":{"bytes":6117,"mode":320,"sha256":"d8254aef7fa2d8d528ccf5da027fb04f3e4f850a6f94fcab859f5dfe237b8a11","source":"adapter/guards/lease_contract.py"},"guards/rping_guard.py":{"bytes":7580,"mode":320,"sha256":"9943d2e66db276fb311b6f8d6577f20096a1863ed8d2d23063014d8240e50876","source":"adapter/guards/rping_guard.py"},"guards/server_group.py":{"bytes":6223,"mode":320,"sha256":"7ebb1a6ce46abce27b895d6428b417614427c3aa5f5d11a8918029225704b0bf","source":"adapter/guards/server_group.py"},"remote_entry.py":{"bytes":3653,"mode":320,"sha256":"15be60f3a72034c38212988ca4ede84965679f8204898345c128c95aa7977053","source":"adapter/remote_entry.py"},"worker_exec.py":{"bytes":960,"mode":320,"sha256":"593674a54ce69e42febf2c800fa93c480da510f00ea24e6daa4ead26d340d8b1","source":"adapter/worker_exec.py"}},"no_model_submitted":true,"role":"server","schema_version":1,"trust":{"R61":"679574b2e3fedcf88ae4dfee25bdd21e2ef31d8b19698db880ec831739d32a85","R62":"146f545c1385e41e2437bc6800f0ff23d6d70afe24312495eadd01732a325d37","canonical_R59":"41f179ce7d7076748e907e566182aa1d3298b4deb776cb52ecd0f694edba526d"}}}')[role]


def elf_header(raw,machine):
    need(len(raw)>=64 and bytes(raw[:7])==b'\x7fELF\x02\x01\x01','ELF64 little-endian required')
    need(int.from_bytes(raw[18:20],'little')==machine,'wrong ELF machine')
    need(int.from_bytes(raw[16:18],'little')==3,'ELF DYN executable/library required')
    return dict(machine=machine,elf_class=64,endian='little',type='DYN')


def _member(name,size,mode):
    m=tarfile.TarInfo(name);m.size=size;m.mode=mode;m.uid=m.gid=0;m.mtime=0;m.type=tarfile.REGTYPE
    return m


def build_bundle(role,path):
    path=Path(path);manifest=expected_manifest(role);body=encoded(manifest)
    need(len(body)<=65536,'manifest cap')
    with path.open('xb') as target:
        with tarfile.open(fileobj=target,mode='w:',format=tarfile.USTAR_FORMAT) as archive:
            archive.addfile(_member('PACKAGE.json',len(body),0o400),io.BytesIO(body))
            for name,spec in manifest['files'].items():
                source=R61/'assets'/spec['source'];fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW)
                with os.fdopen(fd,'rb') as stream:
                    s=os.fstat(stream.fileno());need(stat.S_ISREG(s.st_mode) and s.st_size==spec['bytes'],'source size changed')
                    digest,size=sha_stream(stream,spec['bytes']);need(digest==spec['sha256'] and size==spec['bytes'],'source hash changed')
                    if name.startswith('bin/'):stream.seek(0);elf_header(stream.read(64),183 if role=='client' else 62)
                    stream.seek(0);archive.addfile(_member(name,spec['bytes'],spec['mode']),stream)
        target.flush();os.fsync(target.fileno())
    with path.open('rb') as stream:digest,size=sha_stream(stream,MAX_BYTES+65536)
    return dict(role=role,archive_sha256=digest,archive_bytes=size,manifest_sha256=hashlib.sha256(body).hexdigest(),payload_bytes=sum(v['bytes'] for v in manifest['files'].values()),model_submitted=False)


class PublishedDurabilityUnknown(RuntimeError):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__(json.dumps(evidence,sort_keys=True))


def publish(parent_fd,partial_name,final_name):
    libc=ctypes.CDLL(None,use_errno=True)
    call=getattr(libc,'renameat2',None)
    if call is None:raise OSError(errno.ENOSYS,'renameat2 required, no unsafe fallback')
    call.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];call.restype=ctypes.c_int
    if call(parent_fd,os.fsencode(partial_name),parent_fd,os.fsencode(final_name),1)!=0:
        e=ctypes.get_errno();raise OSError(e,os.strerror(e),final_name)


def install(archive,parent,name,role,archive_sha,manifest_sha):
    archive=Path(archive);parent=Path(parent)
    need(re.fullmatch(r'r53-r61-[A-Za-z0-9_-]{8,55}',name) is not None and len(name)<=64,'case name domain')
    need(all(re.fullmatch('[0-9a-f]{64}',x) for x in [archive_sha,manifest_sha]),'external SHA required')
    manifest=expected_manifest(role);body=encoded(manifest)
    need(hashlib.sha256(body).hexdigest()==manifest_sha,'fixed manifest/hash differs')
    st=parent.lstat();need(parent.is_absolute() and parent.resolve()==parent and stat.S_ISDIR(st.st_mode) and st.st_uid==os.getuid(),'owned non-symlink parent required')
    final=parent/name
    if final.exists() or final.is_symlink():raise FileExistsError(str(final))
    parent_fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    partial_name='.vf-r63-stage-'+secrets.token_hex(8);partial=parent/partial_name;partial_inode=None
    try:
        current=os.fstat(parent_fd);need((current.st_dev,current.st_ino)==(st.st_dev,st.st_ino),'parent replaced')
        fd=os.open(archive,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno());need(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid(),'regular owned archive required')
            total=512+((len(body)+511)//512)*512+sum(512+((s['bytes']+511)//512)*512 for s in manifest['files'].values())+1024
            total=((total+10239)//10240)*10240
            need(info.st_size==total and total<=MAX_BYTES+65536,'canonical archive size')
            digest,size=sha_stream(stream,total);need(digest==archive_sha and size==total,'archive digest differs');stream.seek(0)
            os.mkdir(partial_name,mode=0o700,dir_fd=parent_fd)
            created=partial.lstat();partial_inode=(created.st_dev,created.st_ino)
            with tarfile.open(fileobj=stream,mode='r:') as tar:
                names=['PACKAGE.json',*manifest['files']]
                for name_in_tar in names:
                    member=tar.next();need(member is not None,'missing member')
                    spec=dict(bytes=len(body),mode=0o400,sha256=hashlib.sha256(body).hexdigest()) if name_in_tar=='PACKAGE.json' else manifest['files'][name_in_tar]
                    need(member.name==name_in_tar and member.type==tarfile.REGTYPE and not member.linkname and not member.pax_headers and not tar.pax_headers and member.uid==member.gid==member.mtime==0 and member.mode==spec['mode'] and member.size==spec['bytes'],'member path/type/size/mode differs')
                    src=tar.extractfile(member);need(src is not None,'member payload absent')
                    target=partial/name_in_tar;target.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
                    out_fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                    h=hashlib.sha256();n=0;prefix=bytearray()
                    with os.fdopen(out_fd,'wb') as out:
                        while True:
                            chunk=src.read(min(65536,spec['bytes']-n+1))
                            if not chunk:break
                            n+=len(chunk);need(n<=spec['bytes'],'payload size overflow')
                            if len(prefix)<64:prefix.extend(chunk[:64-len(prefix)])
                            out.write(chunk);h.update(chunk)
                        need(n==spec['bytes'] and h.hexdigest()==spec['sha256'],'payload hash differs')
                        if name_in_tar.startswith('bin/'):elf_header(prefix,183 if role=='client' else 62)
                        out.flush();os.fsync(out.fileno());os.fchmod(out.fileno(),spec['mode'])
                    src.close()
                need(tar.next() is None,'unexpected additional member')
                remaining=stream.read(MAX_BYTES+1);need(not remaining.strip(b'\0'),'nonzero archive tail')
            state=dict(status='STAGED_EXACT_NO_MODEL_SUBMITTED',role=role,archive_sha256=archive_sha,manifest_sha256=manifest_sha,files=len(manifest['files']),payload_bytes=sum(v['bytes'] for v in manifest['files'].values()),model_submitted=False,automatic_retry=False)
            with (partial/'STAGE_RECEIPT.json').open('x') as f:json.dump({**state,'status':'PAYLOAD_VERIFIED_BEFORE_PUBLICATION'},f,indent=2);f.flush();os.fsync(f.fileno())
            for d in sorted([p for p in partial.rglob('*') if p.is_dir()],reverse=True)+[partial]:
                dfd=os.open(d,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                try:os.fsync(dfd)
                finally:os.close(dfd)
            current=parent.lstat();need((current.st_dev,current.st_ino)==(st.st_dev,st.st_ino),'parent replaced before publish')
            publish(parent_fd,partial_name,name);os.fsync(parent_fd)
            return state
    except BaseException as exc:
        try:
            fstat=final.lstat();published_owned=partial_inode is not None and (fstat.st_dev,fstat.st_ino)==partial_inode
        except OSError:published_owned=False
        if published_owned:
            outcome=dict(status='UNKNOWN_PUBLISHED_DURABILITY_NO_RETRY',published=True,model_submitted=False,automatic_retry=False,final_path=str(final),archive_sha256=archive_sha,manifest_sha256=manifest_sha,error=type(exc).__name__+': '+str(exc))
            try:
                with (final/'PUBLICATION_UNKNOWN.json').open('x') as f:json.dump(outcome,f,indent=2)
            except OSError:pass
            raise PublishedDurabilityUnknown(outcome) from exc
        if partial.is_dir():
            try:
                with (partial/'FAILED_NO_AUTOMATIC_RETRY.json').open('x') as f:json.dump(dict(error=type(exc).__name__,model_submitted=False,final_name=name),f)
            except OSError:pass
        raise
    finally:os.close(parent_fd)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    for key in ['archive','parent','name','role','archive-sha','manifest-sha']:parser.add_argument('--'+key,required=True)
    a=parser.parse_args()
    need(__debug__,'optimized installer refused')
    need(a.parent=='/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs' and a.role=='server','approved PC1 server parent/role required')
    try:
        result=install(a.archive,a.parent,a.name,a.role,a.archive_sha,a.manifest_sha)
        print(json.dumps(result),flush=True)
    except PublishedDurabilityUnknown as exc:
        print(json.dumps(exc.evidence),flush=True)
        raise SystemExit(2)
