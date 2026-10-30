"""Admission checks for model-derived input; no owners are started."""
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools' / 'real_sc117_routed_check.py'

class ModelInputAdmission(unittest.TestCase):
    def test_wrong_digest_refused_before_router_or_reservation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runs') as temp:
            root = Path(temp)
            data = root / 'input.f32'
            data.write_bytes(bytes(10240))
            args = [sys.executable, str(SCRIPT)]
            for key in ('pack', 'index', 'dense', 'observation', 'bridge', 'worker', 'oracle', 'router'):
                args += ['--' + key, str(root / 'not-executable')]
            args += ['--input', str(data), '--work-root', str(root), '--router-digest', '0' * 64,
                     '--model-input-sha256', hashlib.sha256(b'other').hexdigest()]
            result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('model input digest mismatch', result.stderr)
            self.assertEqual(list(root.rglob('*.claim.json')), [])

if __name__ == '__main__':
    unittest.main()
