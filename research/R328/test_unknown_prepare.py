import contextlib,io,json,pathlib,subprocess,tempfile,unittest
from unittest.mock import patch
from research.R327 import crosshost

class UnknownPrepareTest(unittest.TestCase):
 def test_timeout_exposes_identity_and_does_not_retry(self):
  with tempfile.TemporaryDirectory() as td:
   base=pathlib.Path(td);(base/'.runs').mkdir();repo=base/'repo';art=repo/'research/R190/artifacts';art.mkdir(parents=True)
   (art/'rank2.safetensors').write_bytes(b'a');(art/'rank9.safetensors').write_bytes(b'b')
   input_dir=repo/'research/R159';input_dir.mkdir(parents=True)
   (input_dir/'input.json').write_bytes((crosshost.R/'research/R159/input.json').read_bytes())
   def digest(p):
    p=pathlib.Path(p)
    return crosshost.BIN_SHA if p.name=='binary' else (crosshost.LOCAL_SHA if p.name=='rank2.safetensors' else crosshost.REMOTE_SHA)
   stream=io.StringIO()
   with patch.object(crosshost,'B',base),patch.object(crosshost,'R',repo),patch.object(crosshost,'BIN',base/'binary'),patch.object(crosshost,'digest',side_effect=digest),patch.object(crosshost,'ssh_python',side_effect=subprocess.TimeoutExpired('ssh',30)) as call,contextlib.redirect_stdout(stream):
    with self.assertRaises(subprocess.TimeoutExpired):crosshost.execute({'local':1},{'remote':1})
   call.assert_called_once();roots=list((base/'.runs').glob('r327-*'));self.assertEqual(len(roots),1);root=roots[0]
   intent=json.loads((root/'intent.json').read_text());receipt=json.loads((root/'receipt.json').read_text());event=json.loads(stream.getvalue())
   self.assertEqual(event['run_id'],intent['run_id']);self.assertEqual(event['remote_root'],intent['remote_root']);self.assertEqual(receipt['status'],'FAILED_OR_UNKNOWN');self.assertFalse((root/'ledger').exists())
