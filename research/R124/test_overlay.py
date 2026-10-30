import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from overlay import prepare, PATHS


def digest(b): return hashlib.sha256(b).hexdigest()


class FullOverlay(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.addCleanup(self.t.cleanup)
        self.p = Path(self.t.name); self.wheel = self.p / 'wheel.whl'
        self.sources = {}; self.originals = {}; self.patches = {}
        with zipfile.ZipFile(self.wheel, 'w') as z:
            for name, member in PATHS.items():
                data = ('original_' + name.replace('.', '_') + '=1\n').encode(); self.originals[name] = data; z.writestr(member, data)
        for name in list(PATHS) + ['skinny_moe.py']:
            data = ('patch_' + name.replace('.', '_') + '=1\n').encode(); self.patches[name] = data
            p = self.p / name; p.write_bytes(data); self.sources[name] = p
        self.expected = {'wheel': digest(self.wheel.read_bytes()), 'originals': {k:digest(v) for k,v in self.originals.items()}, 'patches': {k:digest(v) for k,v in self.patches.items()}}
        self.dest = self.p/'out'

    def test_seven_originals_eight_patches(self):
        result = prepare(self.wheel,self.sources,self.dest,self.expected)
        self.assertEqual(result['status'],'PREPARED_NOT_INSTALLED')
        for n, v in self.originals.items(): self.assertEqual((self.dest/'originals'/n).read_bytes(),v)
        for n, v in self.patches.items(): self.assertEqual((self.dest/'overlay'/n).read_bytes(),v)

    def test_any_patch_mutation_is_refused_before_writes(self):
        self.sources['gpu_model_runner.py'].write_bytes(b'wrong=1\n')
        with self.assertRaises(ValueError): prepare(self.wheel,self.sources,self.dest,self.expected)
        self.assertFalse(self.dest.exists())

    def test_any_wheel_member_mismatch_is_refused(self):
        self.expected['originals']['attention.py']='0'*64
        with self.assertRaises(ValueError): prepare(self.wheel,self.sources,self.dest,self.expected)
        self.assertFalse(self.dest.exists())

    def test_source_set_must_be_exact(self):
        del self.sources['gdn_attn.py']
        with self.assertRaises(ValueError): prepare(self.wheel,self.sources,self.dest,self.expected)
        self.assertFalse(self.dest.exists())

    def test_modified_wheel_is_refused(self):
        with self.wheel.open('ab') as f: f.write(b'x')
        with self.assertRaises(ValueError): prepare(self.wheel,self.sources,self.dest,self.expected)
        self.assertFalse(self.dest.exists())

    def test_already_present_skinny_module_is_refused(self):
        with zipfile.ZipFile(self.wheel,'a') as z: z.writestr('vllm/skinny_moe.py',b'x=1\n')
        self.expected['wheel']=digest(self.wheel.read_bytes())
        with self.assertRaises(ValueError): prepare(self.wheel,self.sources,self.dest,self.expected)
        self.assertFalse(self.dest.exists())

    def test_syntax_error_is_refused(self):
        data=b'not valid !!!\n';self.sources['torch_utils.py'].write_bytes(data);self.expected['patches']['torch_utils.py']=digest(data)
        with self.assertRaises(SyntaxError): prepare(self.wheel,self.sources,self.dest,self.expected)
        self.assertFalse(self.dest.exists())

    def test_destination_must_not_exist(self):
        self.dest.mkdir(); (self.dest/'keep').write_text('untouched')
        with self.assertRaises(FileExistsError): prepare(self.wheel,self.sources,self.dest,self.expected)
        self.assertEqual((self.dest/'keep').read_text(),'untouched')

    def test_symbolic_link_rejected(self):
        p=self.sources['attention.py'];p.unlink();p.symlink_to(self.sources['marlin.py'])
        with self.assertRaises(OSError): prepare(self.wheel,self.sources,self.dest,self.expected)
        self.assertFalse(self.dest.exists())

if __name__=='__main__': unittest.main()
