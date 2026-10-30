import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from adapters.session_bridge import DurableSession, DurableHookError

class FramingTest(unittest.TestCase):
    def test_sequential_ready_and_close(self):
        init = {'routing': {'stamp': {'run_id':'normal-test','model_revision':'m','plan_id':'p','layer':0,'epoch':1,'batch':0}, 'experts':[1]}, 'inputs':[[0]]}
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)
            stub=path/'stub'
            stub.write_text('#!/usr/bin/python3\nimport sys\nsys.stdin.readline()\nprint(\'{"ok":true,"state":"ready"}\',flush=True)\nsys.stdin.readline()\nprint(\'{"ok":true,"state":"closed","close":[]}\',flush=True)\n')
            stub.chmod(0o700)
            session=DurableSession(stub,init,path/'ledger')
            self.assertEqual(session.close()['state'],'closed')

    def test_unsolicited_frame_in_same_read_is_refused(self):
        init = {'routing': {'stamp': {'run_id':'frame-test','model_revision':'m','plan_id':'p','layer':0,'epoch':1,'batch':0}, 'experts':[1]}, 'inputs':[[0]]}
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)
            stub=path/'stub'
            stub.write_text('#!/usr/bin/python3\nimport os,sys\nsys.stdin.readline()\nos.write(1,b\'{"ok":true,"state":"ready"}\\n{"ok":true,"batch":0}\\n\')\n')
            stub.chmod(0o700)
            launched=[]
            original=subprocess.Popen
            def capture(*args,**kwargs):
                p=original(*args,**kwargs); launched.append(p); return p
            with patch('subprocess.Popen',side_effect=capture):
                try:
                    with self.assertRaises(DurableHookError):
                        DurableSession(stub,init,path/'ledger')
                finally:
                    p=launched[0]; p.wait(timeout=5); p.stdin.close(); p.stdout.close()
