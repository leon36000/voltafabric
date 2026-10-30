"""Discriminant for the exact parsed-input versus approved-hash group boundary."""

import hashlib
import json
import subprocess
import sys
from test_guards import Guards, CAND


class ConfigRace(Guards):
    def test_one_raw_config_is_hashed_and_consumed(self):
        (self.d / "fixture-mode").write_text("hold")
        rank = self.d / "rank-0"
        rank.mkdir()
        self.c["receipt"] = str(rank / "guard.json")
        self.c["run_id"] = "r60-cpu-unapproved-A000"
        a = dict(
            lease_path=str(self.nic),
            nic_lease=self.c["nic_lease"],
            guard_script=str(CAND / "rping_guard.py"),
            guard_hashes={
                str(CAND / n): hashlib.sha256((CAND / n).read_bytes()).hexdigest()
                for n in ["rping_guard.py", "server_group.py", "lease_contract.py"]
            },
            guards=[dict(rank=0, config=self.c)],
        )
        b = json.loads(json.dumps(a))
        b["guards"][0]["config"]["run_id"] = "r60-cpu-approved-B000"
        config = self.d / "group.json"
        config.write_text(json.dumps(a))
        approved = self.d / "approved.json"
        approved.write_text(json.dumps(b))
        wrapper = self.d / "race.py"
        wrapper.write_text("""import runpy,sys
from pathlib import Path
config=Path(sys.argv[1]);replacement=Path(sys.argv[2]).read_bytes()
old=Path.read_text
def swap(self,*args,**kwargs):
 text=old(self,*args,**kwargs)
 if self==config:config.write_bytes(replacement)
 return text
Path.read_text=swap
script=sys.argv[3];sha=sys.argv[4];sys.argv=[script,str(config),sha]
runpy.run_path(script,run_name='__main__')
""")
        # runpy does not automatically add candidate/ for its sibling import.
        q = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(wrapper),
                str(config),
                str(approved),
                str(CAND / "server_group.py"),
                hashlib.sha256(approved.read_bytes()).hexdigest(),
            ],
            env=dict(self.env, PYTHONPATH=str(CAND)),
            pass_fds=(self.held.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.children.append(q)
        out, err = q.communicate(timeout=8)
        (self.d / "race-result.json").write_text(
            json.dumps(
                dict(
                    exit_code=q.returncode,
                    stdout=out.decode(),
                    stderr=err.decode(),
                    sha_a=hashlib.sha256(json.dumps(a).encode()).hexdigest(),
                    sha_b=hashlib.sha256(approved.read_bytes()).hexdigest(),
                )
            )
        )
        self.assertNotEqual(q.returncode, 0)
        self.assertFalse(
            (self.d / "spawn.json").exists(), "unapproved A was executed under SHA of B"
        )
        self.assertIn(b"external group config SHA mismatch", err)
