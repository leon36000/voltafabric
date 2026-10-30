"""Controlled pre-init failure with a real local child; no model or network."""
import json,os,pathlib
from unittest.mock import patch
from adapters import session_bridge as bridge
root=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r347-start-unknown-r1');root.mkdir(mode=0o700)
init={'routing':{'stamp':{'run_id':'r347-local-controlled','model_revision':'test','plan_id':'test','layer':0,'epoch':1,'batch':0},'experts':[0]},'inputs':[[0.0]]}
original=bridge._exclusive
def fault(path,data):
 if path.name=='process.json':raise OSError('injected process receipt failure')
 return original(path,data)
with patch.object(bridge,'_exclusive',side_effect=fault):
 try:bridge.DurableSession('/usr/bin/cat',init,root/'ledger')
 except bridge.SessionStartUnknown as e:
  assert e.process.poll() is None and e.pid==e.process.pid and (e.directory/'identity.json').exists()
  # This controlled cat has received no request and has no network or model work.
  # Closing its unused input reconciles this specific child without replay or kill.
  e.process.stdin.close();code=e.process.wait(timeout=5);e.process.stdout.close();assert code==0
  result={'status':'PASS','pid':e.pid,'identity':e.identity,'directory':str(e.directory),'child_exit_code':code,'no_init_sent':True,'gpu':False}
 else:raise AssertionError('failure not surfaced')
(root/'receipt.json').write_text(json.dumps(result,sort_keys=True)+'\n');print(json.dumps(result))
