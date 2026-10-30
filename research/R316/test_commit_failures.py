import hashlib,pathlib,sys,tempfile,unittest
from unittest import mock
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/adapter'));sys.path.insert(0,str(ROOT/'research/R291'))
from test_a92_resident_artifact_install import ARTIFACT,INFO
from local_reuse import LocalReuse
import local_reuse
from adapters.single_owner_admission import admit_owner
from adapters.persistent_gpu_owner import PersistentQPN2Owner
from adapters.resident_multipart import MultipartError
class Provider:
 device='cpu';admissions=launches=synchronized_launches=0
 def gemm_qpn2_moe(self,*args):raise AssertionError('not called')
 def synchronize(self):pass
 def close(self):pass
class TestCommit(unittest.TestCase):
 def setup_receiver(self,root):
  source=root/'source.safetensors';source.write_bytes(ARTIFACT.read_bytes());payload=source.read_bytes();sha=hashlib.sha256(payload).hexdigest();identity={'run_id':'r316','model_revision':INFO['model_revision'],'plan_id':'p','layer':0,'epoch':1,'rank':0,'attempt':1};approved={'path':str(source),'bytes':len(payload),'sha256':sha};recv=LocalReuse(root,identity,len(payload),sha,approved);recv.append(0,payload);return recv,source,sha,identity,approved
 def test_mutation_after_owner_load_refuses_and_closes(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);recv,source,sha,identity,approved=self.setup_receiver(root);original_load=PersistentQPN2Owner.load;original_close=PersistentQPN2Owner.close;created=[]
   def modified_load(**kw):
    owner=original_load(**kw);created.append(owner)
    with source.open('r+b') as f:f.seek(-1,2);byte=f.read(1);f.seek(-1,2);f.write(bytes([byte[0]^1]));f.flush()
    return owner
   with mock.patch.object(PersistentQPN2Owner,'load',side_effect=modified_load),mock.patch.object(PersistentQPN2Owner,'close',autospec=True,side_effect=original_close):
    with self.assertRaises(MultipartError):admit_owner(recv,sha,Provider())
   self.assertFalse((recv.directory/'resident.json').exists());self.assertEqual(recv.state,'FAILED');self.assertEqual(created[0].state,'CLOSED')
   with self.assertRaises(FileExistsError):LocalReuse(root,identity,approved['bytes'],sha,approved)
 def test_receipt_fsync_failure_is_unknown_even_if_receipt_exists(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);recv,source,sha,identity,approved=self.setup_receiver(root);original_close=PersistentQPN2Owner.close;closed=[]
   def close(owner):closed.append(owner);return original_close(owner)
   with mock.patch.object(PersistentQPN2Owner,'close',close),mock.patch.object(local_reuse,'_fsync_dir',side_effect=OSError('injected fsync failure')):
    with self.assertRaises(OSError):admit_owner(recv,sha,Provider())
   self.assertEqual(recv.state,'UNKNOWN');self.assertTrue((recv.directory/'resident.json').exists());self.assertEqual(closed[0].state,'CLOSED')
   with self.assertRaises(FileExistsError):LocalReuse(root,identity,approved['bytes'],sha,approved)
if __name__=='__main__':unittest.main()
