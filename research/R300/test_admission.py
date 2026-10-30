import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault('torch', types.ModuleType('torch'))
from adapters.resource_admission import AdmissionError, ResourcePolicy, parse_meminfo
from adapters.session_bridge import DurableSession

class AdmissionTest(unittest.TestCase):
    def test_meminfo_requires_unambiguous_kib(self):
        self.assertEqual(parse_meminfo('MemAvailable: 300 kB\nSwapFree: 40 kB\n'), (307200, 40960))
        for text in ('MemAvailable: 3 kB\n', 'MemAvailable: 3 MB\nSwapFree: 4 kB\n', 'MemAvailable: 3 kB\nMemAvailable: 4 kB\nSwapFree: 4 kB\n'):
            with self.assertRaises(AdmissionError): parse_meminfo(text)

    def test_floor_and_projected_load(self):
        policy = ResourcePolicy(100, 50, 80, 30)
        policy.check_values(130, 50, 80)
        for values in ((129, 50, 80), (130, 49, 80), (130, 50, 79)):
            with self.assertRaises(AdmissionError): policy.check_values(*values)
        with self.assertRaises(ValueError): ResourcePolicy(-1, 1, 1, 1)

    def test_refusal_happens_before_identity_or_process(self):
        init={'routing':{'stamp':{'run_id':'r','model_revision':'m','plan_id':'p','layer':0,'epoch':1,'batch':0}}}
        def refuse(): raise AdmissionError('insufficient')
        with tempfile.TemporaryDirectory() as tmp, patch('subprocess.Popen') as spawn:
            ledger=pathlib.Path(tmp)/'ledger'
            with self.assertRaises(AdmissionError): DurableSession('/missing',init,ledger,admission=refuse)
            self.assertFalse(ledger.exists()); spawn.assert_not_called()

    def test_os_snapshot(self):
        policy=ResourcePolicy(0,0,0,0)
        with tempfile.TemporaryDirectory() as tmp:
            snap=policy.check(tmp)
            self.assertGreater(snap['mem_available'],0)
            self.assertGreaterEqual(snap['swap_free'],0)
            self.assertGreater(snap['disk_free'],0)

if __name__=='__main__': unittest.main()
