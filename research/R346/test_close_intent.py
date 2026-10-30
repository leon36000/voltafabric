import json,pathlib,sys,tempfile,threading,types,unittest
from unittest.mock import patch,Mock
sys.modules.setdefault('torch',types.ModuleType('torch'))
from adapters.session_bridge import DurableSession,DurableHookError

class CloseIntent(unittest.TestCase):
 def session(self,d):
  s=DurableSession.__new__(DurableSession);s.directory=pathlib.Path(d);s.lock=threading.Lock();s.terminal=False;s.deadline=2;s.batch=3;s.identity={'run_id':'r','model_revision':'m','plan_id':'p','layer':0,'epoch':1}
  s.proc=types.SimpleNamespace(wait=Mock(return_value=0),stdin=Mock(),stdout=Mock());return s
 def test_intent_fsynced_before_exchange_and_closed(self):
  with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as d:
   s=self.session(d)
   def exchange(msg):
    claim=json.loads((pathlib.Path(d)/'close.claim.json').read_text());self.assertEqual(claim['next_batch'],3);self.assertEqual(claim['identity'],s.identity)
    return {'state':'closed','close':[{'exit_code':0}]}
   s._exchange=exchange;s.close();self.assertTrue((pathlib.Path(d)/'closed.json').exists())
 def test_failed_intent_never_sends_shutdown(self):
  with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as d:
   s=self.session(d);s._exchange=Mock()
   with patch('adapters.session_bridge._exclusive',side_effect=OSError('disk')),self.assertRaises(OSError):s.close()
   s._exchange.assert_not_called();self.assertTrue(s.terminal)
   with self.assertRaises(DurableHookError):s.close()
 def test_unknown_close_has_intent_no_closed_and_no_replay(self):
  with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as d:
   s=self.session(d);s._exchange=Mock(side_effect=DurableHookError('deadline exceeded; outcome UNKNOWN'))
   with self.assertRaises(DurableHookError):s.close()
   self.assertTrue((pathlib.Path(d)/'close.claim.json').exists());self.assertFalse((pathlib.Path(d)/'closed.json').exists())
   with self.assertRaises(DurableHookError):s.close()
   self.assertEqual(s._exchange.call_count,1)
