import hashlib,pathlib,tempfile,unittest
from research.R333.crosshost import verify_pins

class CodePinsTest(unittest.TestCase):
 def test_exact_and_changed_and_missing(self):
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)/'worker.py';p.write_bytes(b'approved')
   expected=hashlib.sha256(b'approved').hexdigest();verify_pins({p:expected})
   p.write_bytes(b'changed')
   with self.assertRaises(ValueError):verify_pins({p:expected})
   with self.assertRaises(ValueError):verify_pins({pathlib.Path(td)/'missing':expected})
