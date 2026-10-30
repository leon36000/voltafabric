import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from tools.reconcile_r53_snapshots import reconcile


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a = self.root / 'a'; self.a.mkdir()
        self.b = self.root / 'b'; self.b.mkdir()
        self.base = 'r53-test1234'

    def tearDown(self):
        self.temp.cleanup()

    def record(self, directory, suffix='', status='RESERVED_UNKNOWN_NO_REPLAY'):
        effective = self.base + suffix
        value = {'base_id': self.base, 'effective_run_id': effective, 'status': status}
        if status == 'CONSUMED_RETROSPECTIVE_CPU_NO_REPLAY':
            value.update(recorded_after_execution=True, reserved_before_execution=False, automatic_retry=False, recorded_ns=1)
        else:
            value.update(scope='Namespace consumption only; see linked execution receipt for actual outcome', reserved_ns=1, context={})
        (directory / (effective + '.json')).write_text(json.dumps(value))

    def test_identical_pairs_are_not_live_admission(self):
        for d in (self.a, self.b):
            self.record(d); self.record(d, '-duplicate-slot')
        result = reconcile([self.a, self.b])
        self.assertTrue(result['matching_content'])
        self.assertEqual(result['unique_identities'], 2)
        self.assertFalse(result['ready_for_live_use'])
        self.assertEqual(result['conflicts'], [])
        self.assertEqual(result['unpaired_bases'], [])

    def test_partial_and_changed_records_are_not_silently_merged(self):
        self.record(self.a); self.record(self.a, '-duplicate-slot')
        self.record(self.b)
        result = reconcile([self.a, self.b])
        self.assertEqual(result['missing_from_some'], [self.base + '-duplicate-slot'])
        p = self.b / (self.base + '.json'); value = json.loads(p.read_text()); value['extra'] = 1; p.write_text(json.dumps(value))
        self.assertEqual(reconcile([self.a, self.b])['conflicts'], [self.base])
        self.assertEqual(reconcile([self.b])['unpaired_bases'], [self.base])

    def test_retrospective_is_preserved_as_consumed_not_reserved(self):
        for suffix in ('', '-duplicate-slot'):
            self.record(self.a, suffix, 'CONSUMED_RETROSPECTIVE_CPU_NO_REPLAY')
        self.assertEqual(reconcile([self.a])['retrospective_count'], 2)
        p = self.a / (self.base + '.json'); value = json.loads(p.read_text()); value['reserved_before_execution'] = True; p.write_text(json.dumps(value))
        with self.assertRaises(ValueError): reconcile([self.a])

    def test_refuses_duplicate_json_and_symlink(self):
        p = self.a / (self.base + '.json')
        p.write_text('{"base_id":"x","base_id":"y"}')
        with self.assertRaises(ValueError): reconcile([self.a])
        p.unlink(); os.symlink(self.b, p)
        with self.assertRaises(OSError): reconcile([self.a])

    def test_cli_reports_incomplete_without_mutating_source(self):
        self.record(self.a)
        before = (self.a / (self.base + '.json')).read_bytes()
        tool = Path(__file__).resolve().parents[1] / 'tools/reconcile_r53_snapshots.py'
        result = subprocess.run([sys.executable, str(tool), str(self.a)], capture_output=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)['unpaired_bases'], [self.base])
        self.assertEqual((self.a / (self.base + '.json')).read_bytes(), before)

    def test_unknown_status_is_refused(self):
        self.record(self.a)
        p = self.a / (self.base + '.json'); value = json.loads(p.read_text()); value['status'] = 'MAY_RETRY'; p.write_text(json.dumps(value))
        with self.assertRaises(ValueError): reconcile([self.a])


if __name__ == '__main__':
    unittest.main()
