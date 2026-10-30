import pathlib,sys,tempfile,unittest
from unittest import mock
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'tests/adapter'))
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'research/R291'))
from test_a92_resident_artifact_install import ARTIFACT,ARTIFACT_SHA256,INFO
from local_reuse import LocalReuse
from adapters.single_owner_admission import admit_owner
from adapters.persistent_gpu_owner import PersistentQPN2Owner

class Provider:
 device='cpu';admissions=launches=synchronized_launches=0
 def gemm_qpn2_moe(self,*args):raise AssertionError('not used')
 def synchronize(self):pass
 def close(self):pass

def receiver(root,sha=ARTIFACT_SHA256):
 data=ARTIFACT.read_bytes();identity={'run_id':'r311','model_revision':INFO['model_revision'],'plan_id':'plan','layer':0,'epoch':1,'rank':0,'attempt':1}
 recv=LocalReuse(root,identity,len(data),sha,{'path':str(ARTIFACT),'bytes':len(data),'sha256':sha});recv.append(0,data);return recv
class Tests(unittest.TestCase):
 def test_one_load_and_commit(self):
  with tempfile.TemporaryDirectory() as tmp:
   from scripts import resident_artifact
   recv=receiver(pathlib.Path(tmp));original=resident_artifact.load_stack
   with mock.patch.object(resident_artifact,'load_stack',wraps=original) as load:
    owner=admit_owner(recv,ARTIFACT_SHA256,Provider());self.assertEqual(load.call_count,1)
   self.assertEqual(owner.state,'READY');self.assertTrue((recv.directory/'resident.json').is_file());owner.close()
 def test_wrong_identity_is_refused_before_publish(self):
  with tempfile.TemporaryDirectory() as tmp:
   recv=receiver(pathlib.Path(tmp));recv.identity['layer']=1
   with self.assertRaises(ValueError):admit_owner(recv,ARTIFACT_SHA256,Provider())
   self.assertFalse((recv.directory/'resident.json').exists())
 def test_wrong_digest_refuses_before_receipt(self):
  with tempfile.TemporaryDirectory() as tmp:
   recv=receiver(pathlib.Path(tmp))
   with self.assertRaises(ValueError):admit_owner(recv,'0'*64,Provider())
   self.assertFalse((recv.directory/'resident.json').exists())
 def test_failure_after_validation_closes_unpublished_owner(self):
  class FailingReceiver:
   def finish(self,validate):
    validate(ARTIFACT,{'model_revision':INFO['model_revision'],'layer':0,'epoch':1,'rank':0});raise OSError('injected failure')
  closed=[];original=PersistentQPN2Owner.close
  def observe(owner):closed.append(owner);return original(owner)
  with mock.patch.object(PersistentQPN2Owner,'close',observe):
   with self.assertRaises(OSError):admit_owner(FailingReceiver(),ARTIFACT_SHA256,Provider())
  self.assertEqual(len(closed),1);self.assertEqual(closed[0].state,'CLOSED')
if __name__=='__main__':unittest.main()
