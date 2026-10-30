import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/observe_sc117_experts.py"
INDEX = ROOT / "fixtures/strata_sc117_native_experts_v3.txt"
SIZE = 50_292_326_400

class ObservationTest(unittest.TestCase):
    def test_bounded_read_and_refusals(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".runs") as directory:
            pack = Path(directory) / "pack"
            with open(pack, "wb") as f:
                f.truncate(SIZE)
                f.seek(0)
                f.write(b"abcdefgh")
            def invoke(*extras):
                return subprocess.run(["python3", str(TOOL), "--index", str(INDEX),
                                       "--pack", str(pack), *extras], capture_output=True, text=True)
            result = invoke("--expert", "0:0", "--expert", "0:1")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(result.stdout)
            self.assertEqual(manifest["kind"], "sc117-observation-not-authentication")
            self.assertEqual(manifest["experts"][0]["sha256"],
                             hashlib.sha256(b"abcdefgh" + bytes(2_176_000 - 8)).hexdigest())
            self.assertEqual(manifest["experts"][1]["sha256"], hashlib.sha256(bytes(2_176_000)).hexdigest())
            for arguments in [("--expert", "0:512"), ("--expert", "48:0"),
                              ("--expert", "0:0", "--expert", "0:0")]:
                failed = invoke(*arguments)
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(failed.stdout, "")
            os.unlink(pack)
            os.mkfifo(pack)
            failed = invoke("--expert", "0:0")
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(failed.stdout, "")

if __name__ == "__main__":
    unittest.main()
