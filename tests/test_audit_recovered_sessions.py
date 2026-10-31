import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from tools.audit_recovered_sessions import audit_pair


class RecoveredSessions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.a = {'run_id':'pc1-run','model_revision':'model','plan_id':'plan','layer':1,'epoch':1}
        self.b = {**self.a, 'run_id':'pc6-run'}
        self.p1, self.h1 = self.write('pc1.json', '/pc1', self.a)
        self.p6, self.h6 = self.write('pc6.json', '/pc6', self.b)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, root, identity):
        digest = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        prefix = 'run/ledger/'+digest+'/'
        files = {prefix+'identity.json':json.dumps(identity), prefix+'process.json':json.dumps({'pid':1}),
                 prefix+'batch-00000000000000000000.claim.json':json.dumps({**identity,'batch':0,'attempt':'assigned-by-rust'})}
        raw = json.dumps({'original_root':root,'files':files}).encode()
        path = self.root/name; path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def check(self, candidate):
        return audit_pair([self.p1],[self.p6],[self.h1],[self.h6],candidate)

    def test_each_host_is_consumed_and_absent_never_authorizes(self):
        for identity, host in ((self.a,'pc1'),(self.b,'pc6')):
            r=self.check(identity)
            self.assertEqual(r['candidate']['observed_hosts'],[host])
            self.assertEqual(r['candidate']['status'],'CONSUMED_NO_REPLAY')
            self.assertFalse(r['ready_for_live_use'])
            self.assertEqual(len(r[host]['claims_without_done']),1)
        self.assertEqual(self.check({**self.a,'run_id':'unseen'})['candidate']['status'],'NOT_OBSERVED_NOT_AUTHORIZED')

    def test_same_identity_on_both_hosts_is_still_consumed(self):
        self.p6,self.h6=self.write('pc6-same.json','/pc6',self.a)
        r=self.check(self.a)
        self.assertEqual(r['shared_session_identity_count'],1)
        self.assertEqual(r['candidate']['observed_hosts'],['pc1','pc6'])

    def test_missing_altered_or_malformed_archive_refuses(self):
        with self.assertRaises(ValueError):
            audit_pair([self.p1],[self.p6],['0'*64],[self.h6],self.a)
        self.p6.write_text('{"original_root":"/pc6","files":{},"files":{}}')
        with self.assertRaises(ValueError): self.check(self.a)
        self.p6.unlink()
        with self.assertRaises(OSError): self.check(self.a)

    def test_malformed_candidate_refuses(self):
        with self.assertRaises(ValueError): self.check({**self.a,'epoch':True})


if __name__ == '__main__':
    unittest.main()
