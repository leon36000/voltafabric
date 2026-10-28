import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('r114_snapshot', ROOT/'scripts/check_gitlab_snapshot.py')
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.r114-test-', dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'provenance').mkdir()
        (self.root/'data').mkdir()
        (self.root/'data/a').write_bytes(b'first')
        (self.root/'data/b').write_bytes(b'second')
        self.manifest = {'data/a': sha(b'first'), 'data/b': sha(b'second')}
        self.write_manifest()

    def write_manifest(self, raw=None):
        self.raw = raw if raw is not None else json.dumps(self.manifest).encode()
        (self.root/checker.MANIFEST).write_bytes(self.raw)

    def check(self, digest=None, count=2):
        return checker.check_snapshot(str(self.root), digest or sha(self.raw), count)

    def test_valid_snapshot(self):
        result = self.check()
        self.assertEqual(result['files_verified'], 2)
        self.assertEqual(result['status'], 'R59_SNAPSHOT_INTEGRITY_PASS')

    def test_manifest_hash_mismatch_and_duplicate_keys(self):
        with self.assertRaises(checker.InvalidSnapshot): self.check('0'*64)
        raw = ('{"data/a":"'+sha(b'first')+'","data/a":"'+sha(b'first')+'"}').encode()
        self.write_manifest(raw)
        with self.assertRaises(checker.InvalidSnapshot): self.check(count=1)

    def test_missing_and_altered(self):
        (self.root/'data/a').write_bytes(b'altered')
        with self.assertRaises(checker.InvalidSnapshot): self.check()
        (self.root/'data/a').unlink()
        with self.assertRaises(OSError): self.check()

    def test_paths_and_digest_are_strict(self):
        for path in ('/etc/passwd', '../outside', 'data/../a', './data/a', 'data//a', 'data\\a', 'data/a\x00'):
            with self.subTest(path=path):
                self.manifest = {path: sha(b'first')}; self.write_manifest()
                with self.assertRaises(checker.InvalidSnapshot): self.check(count=1)
        for value in (True, 'A'*64, '0'*63):
            with self.subTest(value=value):
                self.manifest = {'data/a': value}; self.write_manifest()
                with self.assertRaises(checker.InvalidSnapshot): self.check(count=1)

    def test_file_and_parent_symlinks_refused(self):
        (self.root/'data/a').unlink(); (self.root/'data/a').symlink_to('b')
        with self.assertRaises(OSError): self.check()
        (self.root/'data/a').unlink(); (self.root/'data/a').write_bytes(b'first')
        (self.root/'alias').symlink_to('data', target_is_directory=True)
        self.manifest = {'alias/a': sha(b'first')}; self.write_manifest()
        with self.assertRaises(OSError): self.check(count=1)

    def test_root_ancestor_symlink_refused(self):
        link = self.root.parent/(self.root.name+'-alias')
        link.symlink_to(self.root, target_is_directory=True)
        self.addCleanup(link.unlink)
        with self.assertRaises(OSError): checker.check_snapshot(str(link), sha(self.raw), 2)

    def test_fifo_and_directory_refused_without_blocking(self):
        (self.root/'data/a').unlink();os.mkfifo(self.root/'data/a')
        child = '''import importlib.util,sys
s=importlib.util.spec_from_file_location('check',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
try:m.check_snapshot(sys.argv[2],sys.argv[3],2)
except m.InvalidSnapshot:sys.exit(0)
sys.exit(1)
'''
        p = subprocess.run([sys.executable, '-B', '-c', child, str(ROOT/'scripts/check_gitlab_snapshot.py'),
                            str(self.root), sha(self.raw)], capture_output=True, timeout=3)
        self.assertEqual(p.returncode, 0)
        (self.root/'data/a').unlink();(self.root/'data/a').mkdir()
        with self.assertRaises(checker.InvalidSnapshot): self.check()

    def test_sparse_file_over_cap_refused(self):
        with (self.root/'data/a').open('wb') as f: f.truncate(checker.FILE_LIMIT + 1)
        with self.assertRaises(checker.InvalidSnapshot): self.check()

    def test_replacement_after_open_is_detected(self):
        original = checker.os.read
        changed = False
        def read_then_replace(fd, size):
            nonlocal changed
            value = original(fd, size)
            if value == b'first' and not changed:
                changed = True
                (self.root/'data/replacement').write_bytes(b'first')
                os.replace(self.root/'data/replacement', self.root/'data/a')
            return value
        with mock.patch.object(checker.os, 'read', side_effect=read_then_replace):
            with self.assertRaises(checker.InvalidSnapshot): self.check()
        self.assertTrue(changed)

    def test_manifest_symlink_refused(self):
        p = self.root/checker.MANIFEST
        target = self.root/'real-manifest';target.write_bytes(self.raw)
        p.unlink();p.symlink_to(target)
        with self.assertRaises(OSError): self.check()


if __name__ == '__main__':
    unittest.main()
