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

    def test_layer_one_rejects_wrong_route_before_reservation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runs') as temp:
            root = Path(temp)
            data = root / 'input.f32'
            data.write_bytes(bytes(10240))
            route = root / 'route.json'
            route.write_text('[]', encoding='utf8')
            args = [sys.executable, str(SCRIPT)]
            for key in ('pack', 'index', 'dense', 'observation', 'bridge', 'worker', 'oracle', 'router'):
                args += ['--' + key, str(root / 'not-executable')]
            args += ['--input', str(data), '--work-root', str(root), '--router-digest', '0' * 64,
                     '--model-input-sha256', hashlib.sha256(data.read_bytes()).hexdigest(), '--layer', '1',
                     '--route-json', str(route), '--route-sha256', '0' * 64]
            result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('explicit route digest mismatch', result.stderr)
            self.assertEqual(list(root.rglob('*.claim.json')), [])

    def test_extractor_refuses_unpinned_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runs') as temp:
            root = Path(temp)
            source = root / 'wrong.bin'
            source.write_bytes(bytes(110728))
            cmd = [sys.executable, str(ROOT / 'tools' / 'extract_sc117_layer1_route.py'), str(source), str(root)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('source digest mismatch', result.stderr)
            self.assertFalse((root / 'activation.f32').exists())
            self.assertFalse((root / 'route.json').exists())

if __name__ == '__main__':
    unittest.main()
