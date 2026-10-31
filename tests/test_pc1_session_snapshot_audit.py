import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('audit_pc1', Path(__file__).parents[1] / 'tools' / 'audit_pc1_session_snapshots.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

class SnapshotAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        self.root = Path(self.temp.name)
        self.ident = {'run_id':'unit','model_revision':'m','plan_id':'p','layer':2,'epoch':1}
        self.digest = hashlib.sha256(json.dumps(self.ident,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.parent = 'fixture/ledger/' + self.digest
        self.claim = {**self.ident, 'batch':0, 'attempt':'assigned-by-rust'}
        self.done = {**self.claim, 'attempt':3, 'sha256':'a'*64}
    def tearDown(self):
        self.temp.cleanup()
    def part(self, records, name='part.json'):
        p=self.root/name
        p.write_text(json.dumps({'original_root':'/observed/.runs','files':{self.parent+'/'+k:json.dumps(v) for k,v in records.items()}}))
        return p
    def base(self):
        return {'identity.json':self.ident,'process.json':{'pid':1},'batch-00000000000000000000.claim.json':self.claim}
    def test_missing_done_is_unknown_and_never_live(self):
        summary, sessions = M.audit([self.part(self.base())])
        self.assertEqual(len(summary['claims_without_done']),1)
        self.assertFalse(summary['ready_for_live_use'])
        self.assertFalse(summary['done_payloads_verified'])
        self.assertIn(self.digest,sessions)
    def test_done_does_not_verify_payload(self):
        records=self.base();records['batch-00000000000000000000.done.json']=self.done;records['closed.json']={'exit_code':0}
        summary,_=M.audit([self.part(records)])
        self.assertEqual(summary['claims_without_done'],[])
        self.assertFalse(summary['done_payloads_verified'])
        self.assertEqual(summary['sessions_without_closed_marker'],0)
    def test_missing_claim_and_nonzero_close_reported(self):
        records={'identity.json':self.ident,'closed.json':{'exit_code':2},'batch-00000000000000000000.done.json':self.done}
        summary,_=M.audit([self.part(records)])
        self.assertEqual(len(summary['done_without_claim']),1)
        self.assertEqual(len(summary['nonzero_close_markers']),1)
    def test_wrong_identity_and_filename_refused(self):
        records=self.base();records['batch-00000000000000000000.claim.json']={**self.claim,'layer':True}
        with self.assertRaises(ValueError): M.audit([self.part(records)])
        records=self.base();records['batch-00000000000000000001.claim.json']=records.pop('batch-00000000000000000000.claim.json')
        with self.assertRaises(ValueError): M.audit([self.part(records)])
    def test_duplicate_paths_mixed_root_and_symlink_refused(self):
        a=self.part(self.base(),'a.json'); b=self.part(self.base(),'b.json')
        with self.assertRaises(ValueError): M.audit([a,b])
        payload=json.loads(b.read_text());payload['original_root']='/other';payload['files']={};b.write_text(json.dumps(payload))
        with self.assertRaises(ValueError): M.audit([a,b])
        link=self.root/'link';link.symlink_to(a)
        with self.assertRaises(OSError): M.audit([link])
    def test_candidate_reuse_is_refused_and_absence_is_not_authorization(self):
        archive=self.part(self.base()); candidate=self.root/'candidate.json'
        script=Path(__file__).parents[1]/'tools'/'audit_pc1_session_snapshots.py'
        for identity, expected in ((self.ident, 'CONSUMED_NO_REPLAY'), ({**self.ident,'run_id':'different'}, 'NOT_OBSERVED_NOT_AUTHORIZED')):
            candidate.write_text(json.dumps(identity))
            result=subprocess.run([sys.executable,str(script),'--identity-json',str(candidate),str(archive)],capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            value=json.loads(result.stdout)
            self.assertEqual(value['candidate']['status'],expected)
            self.assertFalse(value['ready_for_live_use'])

    def test_duplicate_json_key_and_unsafe_path_refused(self):
        p=self.root/'invalid';p.write_text('{"original_root":"/x","files":{},"files":{}}')
        with self.assertRaises(ValueError): M.audit([p])
        p.write_text(json.dumps({'original_root':'/x','files':{'../escape/identity.json':json.dumps(self.ident)}}))
        with self.assertRaises(ValueError): M.audit([p])

if __name__ == '__main__': unittest.main()
