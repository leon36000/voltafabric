import base64,hashlib,json,pathlib,struct,sys,unittest
import torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));from nvfp4_oracle import expert_forward
class OracleTest(unittest.TestCase):
 def test_actual_checkpoint_expert_against_independent_qpn_capture(self):
  torch.set_num_threads(1);self.assertIsNone(torch.version.cuda);path=R/'fixtures/checkpoint_qwen36_expert0_full.safetensors';self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),'7ede48f5bb49caa171eaf03c691b8d78d11e8644c465e4d008f8bcedf52c1c3a');x=torch.tensor([((i%31)-15)/16 for i in range(2048)],dtype=torch.float16);out=expert_forward(path,0,0,x)
  captured=(B/'.runs/r158-real-resident-owner-cpu/remote.stdout').read_bytes();self.assertEqual(hashlib.sha256(captured).hexdigest(),'2667891667f5d05d2b7a9eb28c3f0a8f88c201a6c464a3583038d438687958f6');ref=torch.tensor(struct.unpack('<2048e',base64.b64decode(json.loads(captured)['results'][0]['row'],validate=True)),dtype=torch.float16);err=(out.float()-ref.float()).abs();self.assertLessEqual(float(err.max()),2e-5);self.assertGreater(float(err.max()),0);self.assertGreater(int((out==ref).sum()),0);print(json.dumps({'max_abs_error':float(err.max()),'exact_values':int((out==ref).sum()),'tolerance':2e-5}))
 def test_nonfinite_input_rejected(self):
  x=torch.tensor([float('nan')],dtype=torch.float16)
  with self.assertRaises(ValueError):expert_forward('unused',0,0,x)
if __name__=='__main__':unittest.main()
