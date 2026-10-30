import copy,pathlib,tempfile,unittest
from adapters.guarded_fleet import GuardedFleet
from adapters.resource_admission import AdmissionError
from research.R318.test_correlated import INIT

STUB='''#!/usr/bin/python3
import json,sys
first=json.loads(sys.stdin.readline())
print(json.dumps({'ok':True,'state':'ready','request_id':first['request_id']}),flush=True)
request=json.loads(sys.stdin.readline())
print(json.dumps({'ok':True,'state':'closed','close':[],'request_id':request['request_id']}),flush=True)
'''
class FleetTest(unittest.TestCase):
 def test_correlated_fleet_closes_sessions_on_admission_refusal(self):
  calls=[]
  def admit():
   calls.append(1)
   if len(calls)==3:raise AdmissionError('refuse')
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td);stub=root/'stub';stub.write_text(STUB);stub.chmod(0o700)
   fleet=GuardedFleet(admit,correlated=True)
   for i in (0,1):
    p=copy.deepcopy(INIT);p['routing']['stamp']['layer']=i
    fleet.open(stub,p,root/('ledger'+str(i)))
   p=copy.deepcopy(INIT);p['routing']['stamp']['layer']=2
   with self.assertRaises(AdmissionError):fleet.open(stub,p,root/'ledger2')
   self.assertFalse((root/'ledger2').exists());self.assertFalse(fleet.cleanup_errors)
   self.assertTrue(all(s.proc.poll()==0 and (s.directory/'closed.json').exists() for s in fleet.sessions.values()))
 def test_reject_invalid_mode_before_admission(self):
  with self.assertRaises(ValueError):GuardedFleet(lambda:None,correlated='yes')
