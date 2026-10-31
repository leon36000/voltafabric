"""Exercise the linker against isolated copies; never touches real claims."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[2]
FILES = [
    'research/R582/verify_fault_links.py', 'research/R581/AUDIT.json',
    'provenance/recovery-20261030/handoff/PRODUCTIVITY_CURRENT.json',
    *(f'provenance/recovery-20261030/pc1-ledgers/part-{i}.json' for i in range(1,6)),
    'research/R278/remote.py','research/R278/failing_worker.py',
    'research/R287/multiplex.py','research/R287/owner_service.py',
    'research/R287/proxy.py','research/R287/test_real.py',
]

class FaultLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        self.root=Path(self.temp.name)
        for name in FILES:
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
    def tearDown(self): self.temp.cleanup()
    def run_probe(self):
        return subprocess.run([sys.executable,str(self.root/'research/R582/verify_fault_links.py')],capture_output=True,text=True)
    def test_exact_sources_and_fault_claims_link_without_authorizing(self):
        r=self.run_probe();self.assertEqual(r.returncode,0,r.stderr)
        data=json.loads(r.stdout);self.assertFalse(data['ready_for_live_use'])
        self.assertEqual({x['round'] for x in data['links']},{'R278','R287'})
        self.assertTrue(all(x['classification']=='CLAIM_WITHOUT_DONE_NO_REPLAY' and not x['report_content_verified'] for x in data['links']))
    def test_changed_source_is_rejected(self):
        path=self.root/'research/R278/remote.py';path.write_bytes(path.read_bytes()+b'\n# altered\n')
        self.assertNotEqual(self.run_probe().returncode,0)
    def test_changed_handoff_identity_is_rejected(self):
        path=self.root/'provenance/recovery-20261030/handoff/PRODUCTIVITY_CURRENT.json'
        doc=json.loads(path.read_bytes());doc['R287']['run_id']='other';path.write_text(json.dumps(doc))
        self.assertNotEqual(self.run_probe().returncode,0)

if __name__=='__main__': unittest.main()
