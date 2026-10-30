import unittest
from unittest.mock import Mock
from adapters.resource_admission import AdmissionError
from research.R327.crosshost import admit_both

class AdmissionTest(unittest.TestCase):
 def test_local_refusal_never_contacts_remote_or_starts(self):
  local=Mock(side_effect=AdmissionError('local'));remote=Mock();start=Mock()
  with self.assertRaises(AdmissionError):admit_both(local,remote,start)
  remote.assert_not_called();start.assert_not_called()
 def test_remote_refusal_never_starts(self):
  local=Mock(return_value={'host':'local'});remote=Mock(side_effect=AdmissionError('remote'));start=Mock()
  with self.assertRaises(AdmissionError):admit_both(local,remote,start)
  start.assert_not_called()
 def test_both_admitted_runs_once(self):
  local=Mock(return_value={'host':'local'});remote=Mock(return_value={'host':'remote'});start=Mock(return_value='started')
  self.assertEqual(admit_both(local,remote,start),'started');start.assert_called_once_with({'host':'local'},{'host':'remote'})
