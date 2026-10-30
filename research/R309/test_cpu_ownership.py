import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tests/adapter'))
from test_a92_resident_artifact_install import ARTIFACT,ARTIFACT_SHA256,INFO
from adapters import skinny_moe
from scripts import resident_artifact

class CpuOwnership(unittest.TestCase):
 def test_private_validated_planes_transfer_to_cpu_owner_without_copy(self):
  loaded=[]; original=resident_artifact.load_stack
  def capture(*args,**kwargs):
   result=original(*args,**kwargs);loaded.append(result);return result
  with mock.patch.object(resident_artifact,'load_stack',side_effect=capture):
   owner=skinny_moe.load_resident_artifact(SimpleNamespace(expert_map=None),ARTIFACT,expected_info=INFO,expected_sha256=ARTIFACT_SHA256,device='cpu')
  mapping={'codes13':owner._qc13,'scales13':owner._qs13,'gscales13':owner._gs13,'codes2':owner._qc2,'scales2':owner._qs2,'gscales2':owner._gs2}
  for name,tensor in mapping.items():
   self.assertEqual(tensor.data_ptr(),loaded[0].tensors[name].data_ptr(),name)
  independently_loaded=original(ARTIFACT,expected=INFO,expected_sha256=ARTIFACT_SHA256)
  for name,tensor in mapping.items():
   self.assertNotEqual(tensor.data_ptr(),independently_loaded.tensors[name].data_ptr(),name)
if __name__=='__main__':unittest.main()
