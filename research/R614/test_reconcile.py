import hashlib, json, os, tempfile, unittest
from pathlib import Path
import reconcile

def record(base, effective, status='RESERVED_UNKNOWN_NO_REPLAY'):
    return json.dumps({'status':status,'base_id':base,'effective_run_id':effective,
                       'scope':'fixture','reserved_ns':1,'context':{}}).encode()

class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); root=Path(self.tmp.name)
        self.snapshot=root/'snapshot';self.candidate=root/'candidate'
        self.snapshot.mkdir(mode=0o700);self.candidate.mkdir(mode=0o700)
        for name in ('r53-abcdefgh','r53-abcdefgh-duplicate-slot'):
            data=record('r53-abcdefgh',name)
            (self.snapshot/(name+'.json')).write_bytes(data)
            (self.candidate/(name+'.json')).write_bytes(data)
            (self.candidate/(name+'.json')).chmod(0o600)
    def tearDown(self): self.tmp.cleanup()
    def digest(self):
        h=hashlib.sha256()
        for f in sorted(self.snapshot.iterdir()):h.update(f.name.encode()+b'\0'+hashlib.sha256(f.read_bytes()).digest())
        return h.hexdigest()
    def audit(self):return reconcile.audit(self.snapshot,self.candidate,self.digest())
    def test_exact_snapshot_is_coverage_only(self):
        result=self.audit();self.assertTrue(result['snapshot_covered']);self.assertFalse(result['live_ledger_qualified'])
        self.assertEqual(result['missing'],[]);self.assertEqual(result['changed'],[])
    def test_missing_and_changed_are_not_covered(self):
        (self.candidate/'r53-abcdefgh.json').unlink()
        (self.candidate/'r53-abcdefgh-duplicate-slot.json').write_bytes(b'{}')
        result=self.audit();self.assertFalse(result['snapshot_covered']);self.assertEqual(result['missing'],['r53-abcdefgh.json']);self.assertEqual(result['changed'],['r53-abcdefgh-duplicate-slot.json'])
    def test_extra_records_are_reported_without_qualifying(self):
        (self.candidate/'r53-ijklmnop.json').write_bytes(record('r53-ijklmnop','r53-ijklmnop'))
        (self.candidate/'r53-ijklmnop.json').chmod(0o600)
        result=self.audit();self.assertTrue(result['snapshot_covered']);self.assertEqual(result['additional'],['r53-ijklmnop.json']);self.assertFalse(result['live_ledger_qualified'])
    def test_symlink_candidate_file_is_refused(self):
        f=self.candidate/'r53-abcdefgh.json';f.unlink();f.symlink_to(self.snapshot/'r53-abcdefgh.json')
        with self.assertRaises(ValueError):self.audit()
    def test_invalid_snapshot_and_duplicate_json_keys_are_refused(self):
        (self.snapshot/'r53-abcdefgh.json').write_bytes(b'{"status":"ok","status":"bad"}')
        with self.assertRaises(ValueError):self.audit()
    def test_symlink_candidate_directory_is_refused(self):
        link=self.snapshot.parent/'link';link.symlink_to(self.candidate, target_is_directory=True)
        with self.assertRaises(ValueError):reconcile.audit(self.snapshot,link,self.digest())
    def test_hard_link_candidate_file_is_refused(self):
        f=self.candidate/'r53-abcdefgh.json';f.unlink();os.link(self.snapshot/'r53-abcdefgh.json',f)
        with self.assertRaises(ValueError):self.audit()
    def test_insecure_candidate_file_is_refused(self):
        os.chmod(self.candidate/'r53-abcdefgh.json',0o644)
        with self.assertRaises(ValueError):self.audit()
    def test_untrusted_or_incomplete_snapshot_digest_is_refused(self):
        with self.assertRaises(ValueError):reconcile.audit(self.snapshot,self.candidate,"0"*64)
    def test_insecure_candidate_permissions_are_refused(self):
        os.chmod(self.candidate,0o755)
        with self.assertRaises(ValueError):self.audit()

if __name__=='__main__':unittest.main()
