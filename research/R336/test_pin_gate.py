"""Reject changed pinned code before creating any run or contacting the peer."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from research.R333 import crosshost

class GateTest(unittest.TestCase):
    def test_mutated_code_refuses_before_peer_or_run(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'code.py'
            path.write_bytes(b'approved')
            approved = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_bytes(b'changed')
            with patch.object(crosshost, 'local_code_pins', return_value={path: approved}), \
                 patch.object(crosshost, 'ssh_python') as remote, \
                 patch.object(crosshost.uuid, 'uuid4') as identity:
                with self.assertRaisesRegex(ValueError, 'hash changed'):
                    crosshost.execute({}, {})
                remote.assert_not_called()
                identity.assert_not_called()

if __name__ == '__main__':
    unittest.main()
