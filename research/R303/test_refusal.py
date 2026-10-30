import pathlib,subprocess,sys,tempfile,types,unittest
from unittest.mock import patch
sys.modules.setdefault('torch',types.ModuleType('torch'))
from adapters.session_bridge import DurableSession,DurableHookError

class TestRefusal(unittest.TestCase):
 def test_refusal_reports_only_safe_code(self):
  init={'routing':{'stamp':{'run_id':'refusal-test','model_revision':'m','plan_id':'p','layer':0,'epoch':1,'batch':0},'experts':[1]},'inputs':[[0]]}
  with tempfile.TemporaryDirectory() as tmp:
   p=pathlib.Path(tmp);stub=p/'stub';stub.write_text('#!/usr/bin/python3\nimport sys,json\nsys.stdin.readline()\nprint(json.dumps({"ok":False,"error":"WORKER_IO: private-secret"}),flush=True)\n');stub.chmod(0o700)
   real_popen=subprocess.Popen; launched=[]
   def spawn(*args,**kwargs):
    process=real_popen(*args,**kwargs);launched.append(process);return process
   with patch('subprocess.Popen', side_effect=spawn):
    with self.assertRaises(DurableHookError) as caught:DurableSession(stub,init,p/'ledger')
    process=launched[0];process.wait(timeout=5);process.stdin.close();process.stdout.close()
   self.assertIn('WORKER_IO',str(caught.exception));self.assertNotIn('private-secret',str(caught.exception))
if __name__=='__main__':unittest.main()
