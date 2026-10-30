import unittest
from adapters.resource_admission import AdmissionError, ResourcePolicy

class Details(unittest.TestCase):
 def test_all_violated_thresholds_and_snapshot(self):
  p=ResourcePolicy(100,50,80,30)
  with self.assertRaises(AdmissionError) as caught:p.check_values(129,49,79)
  e=caught.exception
  self.assertEqual(e.snapshot,{'mem_available':129,'swap_free':49,'disk_free':79})
  self.assertEqual(e.limits,{'mem_available':130,'swap_free':50,'disk_free':80})
  self.assertEqual(e.reasons,('mem_available','swap_free','disk_free'))
 def test_boundary_is_inclusive_and_single_reason(self):
  p=ResourcePolicy(100,50,80,30)
  self.assertEqual(p.check_values(130,50,80)['mem_available'],130)
  with self.assertRaises(AdmissionError) as caught:p.check_values(130,49,80)
  self.assertEqual(caught.exception.reasons,('swap_free',))
 def test_invalid_snapshot_carries_no_claimed_measurements(self):
  with self.assertRaises(AdmissionError) as caught:ResourcePolicy(1,1,1).check_values(-1,1,1)
  self.assertIsNone(caught.exception.snapshot)
