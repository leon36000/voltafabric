import json,unittest
from unittest.mock import Mock
from adapters.resource_admission import AdmissionError
from research.R333 import crosshost as m

class CrossHostTest(unittest.TestCase):
 def test_input_and_artifact_pins(self):
  p=json.loads((m.R/'research/R333/input.json').read_text());m.validate_fixture_input(p)
  for path,expected in ((m.BIN,m.BIN_SHA),(m.B/'.runs/r206-gate-route-20261028/owners/rank2.safetensors',m.LOCAL_SHA),(m.B/'.runs/r206-gate-route-20261028/owners/rank9.safetensors',m.REMOTE_SHA),(m.B/'.runs/r206-gate-route-20261028/selected.safetensors',m.SOURCE_SHA)):
   self.assertEqual(m.digest(path),expected)
  p['inputs'][0][0]=0.0
  with self.assertRaises(ValueError):m.validate_fixture_input(p)
 def test_local_refusal_does_not_contact_remote(self):
  remote=Mock();start=Mock()
  with self.assertRaises(AdmissionError):m.admit_both(Mock(side_effect=AdmissionError('no')),remote,start)
  remote.assert_not_called();start.assert_not_called()
 def test_remote_refusal_does_not_start(self):
  start=Mock()
  with self.assertRaises(AdmissionError):m.admit_both(Mock(return_value={}),Mock(side_effect=AdmissionError('no')),start)
  start.assert_not_called()
