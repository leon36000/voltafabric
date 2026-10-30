import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from overlay import prepare_overlay


def sha(b):
    return hashlib.sha256(b).hexdigest()


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wheel = self.root / 'input.whl'
        self.originals = {'modelopt.py': b'original_modelopt = 1\n', 'marlin.py': b'original_marlin = 1\n'}
        with zipfile.ZipFile(self.wheel, 'w') as z:
            for name, data in self.originals.items():
                path = ('vllm/model_executor/layers/quantization/' if name == 'modelopt.py' else 'vllm/model_executor/kernels/linear/nvfp4/') + name
                z.writestr(path, data)
        self.sources = {}
        self.replacements = {}
        for name in ('modelopt.py', 'marlin.py', 'skinny_moe.py'):
            data = (name.replace('.', '_') + ' = 1\n').encode()
            self.replacements[name] = data
            self.sources[name] = self.root / name
            self.sources[name].write_bytes(data)
        self.expected = {'wheel': sha(self.wheel.read_bytes()), 'originals': {k: sha(v) for k, v in self.originals.items()}, 'patches': {k: sha(v) for k, v in self.replacements.items()}}
        self.out = self.root / 'out'

    def call(self):
        return prepare_overlay(self.wheel, self.sources, self.out, self.expected)

    def test_pins_originals_and_three_patches_without_touching_wheel(self):
        before = self.wheel.read_bytes()
        result = self.call()
        self.assertEqual(result['status'], 'PREPARED_NOT_INSTALLED')
        self.assertEqual(self.wheel.read_bytes(), before)
        for n, v in self.originals.items(): self.assertEqual((self.out / 'originals' / n).read_bytes(), v)
        for n, v in self.replacements.items(): self.assertEqual((self.out / 'overlay' / n).read_bytes(), v)

    def test_modified_patch_is_refused_before_destination_creation(self):
        self.sources['skinny_moe.py'].write_bytes(b'changed = 1\n')
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.out.exists())

    def test_modified_wheel_is_refused_before_destination_creation(self):
        with self.wheel.open('ab') as f: f.write(b'x')
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.out.exists())

    def test_existing_destination_is_not_overwritten(self):
        self.out.mkdir(); (self.out / 'keep').write_text('keep')
        with self.assertRaises(FileExistsError): self.call()
        self.assertEqual((self.out / 'keep').read_text(), 'keep')

    def test_wrong_original_digest_is_refused(self):
        self.expected['originals']['marlin.py'] = '0' * 64
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.out.exists())

    def test_symlink_patch_is_refused(self):
        target = self.sources['skinny_moe.py']
        target.unlink()
        target.symlink_to(self.sources['marlin.py'])
        with self.assertRaises(OSError): self.call()
        self.assertFalse(self.out.exists())

    def test_existing_skinny_module_is_refused(self):
        with zipfile.ZipFile(self.wheel, 'a') as z:
            z.writestr('vllm/model_executor/layers/quantization/skinny_moe.py', b'x = 1\n')
        self.expected['wheel'] = sha(self.wheel.read_bytes())
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.out.exists())

    def test_syntax_error_in_patch_is_refused_before_destination(self):
        self.sources['modelopt.py'].write_bytes(b'invalid !!!\n')
        self.expected['patches']['modelopt.py'] = sha(b'invalid !!!\n')
        with self.assertRaises(SyntaxError): self.call()
        self.assertFalse(self.out.exists())


if __name__ == '__main__': unittest.main()
