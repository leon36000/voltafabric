import pathlib,sys,tempfile,types,unittest
from unittest.mock import Mock,patch
sys.modules.setdefault('torch',types.ModuleType('torch'))
from adapters.session_bridge import DurableSession,SessionStartUnknown

class StartFailure(unittest.TestCase):
 def test_process_receipt_failure_exposes_live_process_and_identity(self):
  init={'routing':{'stamp':{'run_id':'r','model_revision':'m','plan_id':'p','layer':0,'epoch':1,'batch':0},'experts':[0]},'inputs':[[0.0]]}
  proc=types.SimpleNamespace(pid=1234,kill=Mock(),terminate=Mock())
  calls=[]
  def write(path,data):
   calls.append(path.name)
   if path.name=='process.json':raise OSError('disk unavailable')
  with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as tmp,patch('adapters.session_bridge._exclusive',side_effect=write),patch('adapters.session_bridge._sync_dir'),patch('subprocess.Popen',return_value=proc):
   with self.assertRaises(SessionStartUnknown) as caught:DurableSession('binary',init,tmp)
   e=caught.exception;self.assertIs(e.process,proc);self.assertEqual(e.pid,1234);self.assertEqual(e.identity['run_id'],'r');self.assertTrue(e.directory.is_dir());self.assertEqual(calls,['identity.json','process.json']);proc.kill.assert_not_called();proc.terminate.assert_not_called()
