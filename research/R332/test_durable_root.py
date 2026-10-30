import contextlib,hashlib,io,pathlib,subprocess,tempfile,unittest,os
from unittest.mock import patch
from research.R327 import crosshost

class DurableRootTest(unittest.TestCase):
 def test_parent_directories_synced_before_uncertain_remote_ack(self):
  with tempfile.TemporaryDirectory() as td:
   base=pathlib.Path(td);(base/'.runs').mkdir();remote=base/'remote';remote.mkdir();repo=base/'repo';art=repo/'research/R190/artifacts';art.mkdir(parents=True)
   (art/'rank2.safetensors').write_bytes(b'a');(art/'rank9.safetensors').write_bytes(b'b');ip=repo/'research/R159';ip.mkdir(parents=True);(ip/'input.json').write_bytes((crosshost.R/'research/R159/input.json').read_bytes())
   remote_art=base/'remote-art';remote_art.write_bytes(b'remote');remote_hash=hashlib.sha256(b'remote').hexdigest()
   def digest(p):
    p=pathlib.Path(p);return crosshost.BIN_SHA if p.name=='binary' else (crosshost.LOCAL_SHA if p.name=='rank2.safetensors' else crosshost.REMOTE_SHA)
   real_fsync=os.fsync;seen=[]
   def tracked(fd):
    seen.append(os.readlink('/proc/self/fd/'+str(fd)));return real_fsync(fd)
   def remote_no_ack(script):
    exec(script,{});raise subprocess.TimeoutExpired('ssh',30)
   with patch.object(crosshost,'B',base),patch.object(crosshost,'R',repo),patch.object(crosshost,'BIN',base/'binary'),patch.object(crosshost,'REMOTE',str(remote)),patch.object(crosshost,'REMOTE_ART',str(remote_art)),patch.object(crosshost,'REMOTE_SHA',remote_hash),patch.object(crosshost,'digest',side_effect=digest),patch.object(crosshost,'ssh_python',side_effect=remote_no_ack) as rpc,patch.object(os,'fsync',side_effect=tracked),contextlib.redirect_stdout(io.StringIO()):
    with self.assertRaises(subprocess.TimeoutExpired):crosshost.execute({'local':1},{'remote':1})
   rpc.assert_called_once();self.assertIn(str(base/'.runs'),seen);self.assertIn(str(remote),seen)
   root=next((base/'.runs').glob('r327-*'));self.assertFalse((root/'ledger').exists());self.assertTrue((root/'receipt.json').is_file())
