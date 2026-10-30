import copy,json,unittest
from research.R327.crosshost import R,validate_fixture_input

class ReferenceTest(unittest.TestCase):
 def test_expected_input_is_admitted(self):
  p=json.loads((R/'research/R159/input.json').read_bytes());validate_fixture_input(p)
 def test_changed_input_is_refused(self):
  p=json.loads((R/'research/R159/input.json').read_bytes());q=copy.deepcopy(p);q['inputs'][0][0]=0.0
  with self.assertRaises(ValueError):validate_fixture_input(q)
 def test_extra_row_is_refused(self):
  p=json.loads((R/'research/R159/input.json').read_bytes());p['inputs'].append(p['inputs'][0])
  with self.assertRaises(ValueError):validate_fixture_input(p)
