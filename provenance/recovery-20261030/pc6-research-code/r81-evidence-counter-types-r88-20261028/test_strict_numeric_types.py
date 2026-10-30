"""Synthetic R81 evidence only. No model, CUDA, worker, or network is run."""
import copy
import json
import unittest

import nominal_contract as N
from test_nominal_contract import synthetic


class StrictNumericTypes(unittest.TestCase):
    def test_summary_float_counters_are_rejected(self):
        rows, golden = synthetic()
        mutations = {
            'execs': 2.0,
            'slots_per_exec': 138.0,
            'upload_plane_copies_per_expert': 6.0,
            'kernel_launches': [736.0, 736, 736],
            'cuda_api_calls': 4845.0,
        }
        for key, value in mutations.items():
            with self.subTest(field=key):
                bad = copy.deepcopy(rows)
                bad[6][key] = value
                with self.assertRaises(AssertionError):
                    N.admit_rows(bad, golden)

    def test_binding_float_and_boolean_identifiers_are_rejected(self):
        rows, golden = synthetic()
        for index, key, value in [(0, 'expert', 17.0), (0, 'local', 4.0),
                                  (1, 'local', True), (2, 'expert', 241.0)]:
            with self.subTest(index=index, field=key, value=value):
                bad = copy.deepcopy(rows)
                lines = bad[5]['close']['stderr'].splitlines()
                evicted = json.loads(lines[1])
                evicted['binding']['experts'][index][key] = value
                lines[1] = json.dumps(evicted)
                bad[5]['close']['stderr'] = '\n'.join(lines)
                with self.assertRaises(AssertionError):
                    N.admit_rows(bad, golden)


if __name__ == '__main__':
    unittest.main()
