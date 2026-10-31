"""Synthetic local filesystem checks; never opens the live R53 ledger."""
import fcntl
import importlib.util
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tools.reserve_r53_pinned import reserve


class PinnedReservation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'ledger'
        self.root.mkdir(mode=0o700)
        self.lease_path = Path(self.temp.name) / 'lease'
        self.lease = self.lease_path.open('w+')
        self.addCleanup(self.lease.close)
        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s = self.lease_path.stat()
        self.lease_spec = dict(path=str(self.lease_path), dev=s.st_dev, ino=s.st_ino, uid=s.st_uid)
        s = self.root.stat()
        self.spec = dict(root=str(self.root), dev=s.st_dev, ino=s.st_ino, uid=s.st_uid, base_id='r53-synthetic-one')

    def invoke(self):
        return reserve(self.spec, 'r600-synthetic-plan', self.lease_spec, self.lease.fileno())

    def test_both_records_bind_plan_and_lease_and_repeat_is_refused(self):
        self.assertEqual(self.invoke(), ['r53-synthetic-one', 'r53-synthetic-one-duplicate-slot'])
        for name in ('r53-synthetic-one', 'r53-synthetic-one-duplicate-slot'):
            value=json.loads((self.root / (name + '.json')).read_bytes())
            self.assertEqual(value['status'], 'RESERVED_UNKNOWN_NO_REPLAY')
            self.assertEqual(value['effective_run_id'], name)
            self.assertEqual(value['context'], {'plan_id':'r600-synthetic-plan','shared_lease':self.lease_spec})
        with self.assertRaises(FileExistsError): self.invoke()

    def test_root_replacement_refuses_and_never_writes_replacement(self):
        real_open = os.open
        moved = self.root.with_name('moved')
        switched = False
        def swapping(path, flags, *args, **kwargs):
            nonlocal switched
            fd = real_open(path, flags, *args, **kwargs)
            if str(path) == 'r53-synthetic-one.json' and not switched:
                switched = True
                self.root.rename(moved)
                self.root.mkdir(mode=0o700)
            return fd
        with patch('tools.reserve_r53_pinned.os.open', side_effect=swapping):
            with self.assertRaises((ValueError, OSError)): self.invoke()
        self.assertTrue(switched)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertTrue((moved/'r53-synthetic-one.json').exists())

    def test_legacy_producer_can_split_pair_after_root_replacement(self):
        legacy_path = Path(__file__).resolve().parents[1] / 'provenance/recovery-20261030/r53-tools/namespace_ledger.py'
        module_spec = importlib.util.spec_from_file_location('legacy_namespace_ledger', legacy_path)
        legacy = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(legacy)
        real_open = os.open
        moved = self.root.with_name('legacy-moved')
        switched = False
        def swapping(path, flags, *args, **kwargs):
            nonlocal switched
            fd = real_open(path, flags, *args, **kwargs)
            if Path(path).name == 'r53-synthetic-one.json' and not switched:
                switched = True
                self.root.rename(moved)
                self.root.mkdir(mode=0o700)
            return fd
        with patch.object(legacy.os, 'open', side_effect=swapping):
            legacy.reserve(self.root, self.spec['base_id'], {})
        self.assertTrue(switched)
        self.assertTrue((moved/'r53-synthetic-one.json').exists())
        self.assertTrue((self.root/'r53-synthetic-one-duplicate-slot.json').exists())

    def test_lost_lease_leaves_first_identity_consumed(self):
        real_open = os.open
        def unlock_after_first(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if str(path) == 'r53-synthetic-one.json':
                fcntl.flock(self.lease, fcntl.LOCK_UN)
            return fd
        with patch('tools.reserve_r53_pinned.os.open', side_effect=unlock_after_first):
            with self.assertRaises(ValueError): self.invoke()
        self.assertTrue((self.root/'r53-synthetic-one.json').exists())
        self.assertFalse((self.root/'r53-synthetic-one-duplicate-slot.json').exists())
        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaises(FileExistsError): self.invoke()

    def test_missing_lock_and_bad_directory_identity_fail_before_writes(self):
        fcntl.flock(self.lease, fcntl.LOCK_UN)
        with self.assertRaises((ValueError, OSError)): self.invoke()
        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.spec['ino'] += 1
        with self.assertRaises((ValueError, OSError)): self.invoke()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_two_contenders_cannot_both_succeed(self):
        def attempt(_):
            try: self.invoke(); return True
            except FileExistsError: return False
        with ThreadPoolExecutor(max_workers=2) as executor:
            results=list(executor.map(attempt, range(2)))
        self.assertEqual(sum(results), 1)

    def test_suffix_collision_leaves_first_reservation_consumed(self):
        (self.root/'r53-synthetic-one-duplicate-slot.json').write_text('{}')
        with self.assertRaises(FileExistsError): self.invoke()
        self.assertTrue((self.root/'r53-synthetic-one.json').exists())
        with self.assertRaises(FileExistsError): self.invoke()

if __name__ == '__main__':
    unittest.main()
