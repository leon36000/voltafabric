import pathlib,tempfile,unittest
import torch
from adapters.session_bridge import DurableSession,DurableHookError

INIT={'routing':{'stamp':{'run_id':'test','model_revision':'m','plan_id':'p','layer':0,'epoch':1,'batch':0},'experts':[1]},'inputs':[[0.0]]}
STUB='''#!/usr/bin/python3
import json,sys
first=json.loads(sys.stdin.readline())
print(json.dumps({'ok':True,'state':'ready','request_id':first['request_id']}),flush=True)
request=json.loads(sys.stdin.readline())
rid=request['request_id'] if MODE == 'good' else first['request_id']
print(json.dumps({'ok':True,'batch':request['batch'],'result':{'ok':True,'output':[[2.0]],'attempt':0},'request_id':rid}),flush=True)
if MODE == 'good':
 request=json.loads(sys.stdin.readline())
 print(json.dumps({'ok':True,'state':'closed','close':[],'request_id':request['request_id']}),flush=True)
'''
class CorrelationTest(unittest.TestCase):
 def case(self,mode):
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td); stub=root/'stub';stub.write_text(STUB.replace("MODE ==",repr(mode)+" =="));stub.chmod(0o700)
   s=DurableSession(stub,INIT,root/'ledger',correlated=True)
   x=torch.tensor([[1.0]],dtype=torch.float16);w=torch.tensor([[1.0]]);ids=torch.tensor([[1]])
   try:
    if mode=='good':
     self.assertEqual(s(None,x,w,ids,None,None).tolist(),[[2.0]])
     self.assertEqual(s.close()['state'],'closed')
    else:
     with self.assertRaises(DurableHookError):s(None,x,w,ids,None,None)
     self.assertTrue(s.terminal)
     self.assertFalse((s.directory/'batch-00000000000000000000.done.json').exists())
   finally:
    s.proc.wait(timeout=5)
    if not s.proc.stdin.closed:s.proc.stdin.close()
    if not s.proc.stdout.closed:s.proc.stdout.close()
 def test_stale_response_rejected(self):self.case('bad')
 def test_round_trip(self):self.case('good')
