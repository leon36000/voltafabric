import pathlib,sys,tempfile,types,unittest
sys.modules.setdefault('torch', types.ModuleType('torch'))
from adapters.resource_admission import AdmissionError
from adapters.guarded_fleet import GuardedFleet

WORKER='''#!/usr/bin/python3
import json,sys
for line in sys.stdin:
 request=json.loads(line)
 if request.get('op')=='shutdown':
  print(json.dumps({'ok':True,'state':'closed','close':[{'exit_code':0}]}),flush=True);break
 print(json.dumps({'ok':True,'state':'ready'}),flush=True)
'''
def init(layer):
 return {'routing':{'stamp':{'run_id':'fresh-r301','model_revision':'model','plan_id':'plan','layer':layer,'epoch':1,'batch':0},'experts':[1]},'inputs':[[0.0]]}

class Tests(unittest.TestCase):
 def test_refusal_closes_only_admitted_process_without_claiming_next(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);binary=root/'stub';binary.write_text(WORKER);binary.chmod(0o700)
   calls=[]
   def admission():
    calls.append(1)
    if len(calls)==2: raise AdmissionError('resource floor')
   fleet=GuardedFleet(admission)
   first=fleet.open(binary,init(0),root/'ledger0')
   with self.assertRaises(AdmissionError):fleet.open(binary,init(1),root/'ledger1')
   self.assertEqual(first.proc.poll(),0);self.assertTrue((first.directory/'closed.json').exists())
   self.assertFalse((root/'ledger1').exists());self.assertEqual(fleet.cleanup_errors,[])
   self.assertTrue(fleet.close_known());self.assertEqual(fleet.cleanup_errors,[])
   with self.assertRaises(RuntimeError):fleet.open(binary,init(2),root/'ledger2')
 def test_unknown_is_not_killed_or_closed(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);binary=root/'stub';binary.write_text(WORKER);binary.chmod(0o700)
   fleet=GuardedFleet(lambda:None); first=fleet.open(binary,init(0),root/'ledger0')
   first.terminal=True
   fleet.close_known()
   self.assertIsNone(first.proc.poll());self.assertEqual(len(fleet.cleanup_errors),1)
   first.terminal=False;first.close()
 def test_duplicate_layer_does_not_claim_identity(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);binary=root/'stub';binary.write_text(WORKER);binary.chmod(0o700)
   fleet=GuardedFleet(lambda:None);fleet.open(binary,init(0),root/'ledger0')
   with self.assertRaises(ValueError):fleet.open(binary,init(0),root/'ledger1')
   self.assertFalse((root/'ledger1').exists());fleet.close_known()
if __name__=='__main__':unittest.main()
