"""Real group/guard/private-lock topology. All workers and inventory are CPU fixtures."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from test_guards import Guards, CAND, busy


class GroupContracts(Guards):
    # Only run explicitly selected group methods to avoid inherited repetitions.
    def group_start(self):
        rank = self.d / "rank-0"
        rank.mkdir()
        self.c["receipt"] = str(rank / "guard.json")
        group = dict(
            lease_path=str(self.nic),
            nic_lease=self.c["nic_lease"],
            guard_script=str(CAND / "rping_guard.py"),
            guard_hashes={
                str(CAND / n): hashlib.sha256((CAND / n).read_bytes()).hexdigest()
                for n in ["rping_guard.py", "server_group.py", "lease_contract.py"]
            },
            guards=[dict(rank=0, config=self.c)],
        )
        path = self.d / "group.json"
        path.write_text(json.dumps(group))
        q = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(CAND / "server_group.py"),
                str(path),
                getattr(
                    self,
                    "group_sha_override",
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                ),
            ],
            env=self.env,
            pass_fds=(self.held.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.children.append(q)
        return q

    def group_ready(self, q):
        self.assertTrue(
            select.select([q.stdout], [], [], 4)[0], "group readiness timeout"
        )
        line = q.stdout.readline()
        self.assertTrue(line, line)
        self.assertEqual(json.loads(line)["status"], "GROUP_READY")

    def test_group_nominal_stop_holds_both_locks(self):
        (self.d / "fixture-mode").write_text("hold")
        q = self.group_start()
        self.group_ready(q)
        self.assertTrue(busy(self.nic))
        self.assertTrue(busy(self.gpu))
        q.stdin.write(b"STOP\n")
        q.stdin.flush()
        out, err = q.communicate(timeout=8)
        self.assertEqual(q.returncode, 0, (out, err))
        j = json.loads((self.d / "group-operation.json").read_text())
        self.assertTrue(j["stop_requested"])
        self.assertEqual(j["reader_errors"], [])
        r = json.loads(Path(self.c["receipt"]).read_text())
        self.assertFalse(Path("/proc", str(r["worker_pid"])).exists())
        self.assertFalse(busy(self.gpu))
        self.assertTrue(busy(self.nic))

    def test_group_stale_nic_no_child(self):
        self.c["nic_lease"]["ino"] += 1
        q = self.group_start()
        out, err = q.communicate(timeout=5)
        self.assertNotEqual(q.returncode, 0, (out, err))
        self.assertFalse((self.d / "spawn.json").exists())
        self.assertFalse((self.d / "group-operation.json").exists())

    def test_group_external_digest_mismatch(self):
        self.group_sha_override = "0" * 64
        q = self.group_start()
        out, err = q.communicate(timeout=5)
        self.assertNotEqual(q.returncode, 0, (out, err))
        self.assertIn(b"external group config SHA mismatch", err)
        self.assertFalse((self.d / "spawn.json").exists())

    def test_group_timeout_real_twelve_seconds(self):
        (self.d / "fixture-mode").write_text("hold")
        q = self.group_start()
        self.group_ready(q)
        q.wait(timeout=17)
        out, err = q.communicate(timeout=2)
        self.assertEqual(q.returncode, 2, (out, err))
        j = json.loads((self.d / "group-operation.json").read_text())
        self.assertIn("group runtime cap", j["error"])
        self.assertFalse(j["stop_requested"])
        self.assertEqual(j["reader_errors"], [])
        r = json.loads(Path(self.c["receipt"]).read_text())
        self.assertFalse(Path("/proc", str(r["worker_pid"])).exists())
        self.assertFalse(busy(self.gpu))
        self.assertTrue(busy(self.nic))

    def test_group_abrupt_death_reaps_guard_and_bridge(self):
        libc = ctypes.CDLL(None, use_errno=True)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        (self.d / "fixture-mode").write_text("hold")
        q = self.group_start()
        self.group_ready(q)
        j = json.loads((self.d / "group-operation.json").read_text())
        guard = j["controllers"][0]["pid"]
        bridge = json.loads((self.d / "spawn.json").read_text())["pid"]
        q.kill()
        q.communicate(timeout=3)
        wanted = {guard, bridge}
        statuses = {}
        until = time.monotonic() + 3
        while wanted and time.monotonic() < until:
            for pid in list(wanted):
                try:
                    got, status = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    continue
                if got == pid:
                    statuses[pid] = status
                    wanted.remove(pid)
            if wanted:
                time.sleep(0.005)
        self.assertFalse(wanted, statuses)
        for pid, status in statuses.items():
            self.assertTrue(os.WIFSIGNALED(status))
            self.assertEqual(os.WTERMSIG(status), signal.SIGKILL)
            self.assertFalse(Path("/proc", str(pid)).exists())
        self.assertFalse(busy(self.gpu))
        self.assertTrue(busy(self.nic))
        self.held.close()
        self.assertFalse(busy(self.nic))
        (self.d / "orphan-reaping.json").write_text(
            json.dumps(dict(statuses=statuses, guard_pid=guard, bridge_pid=bridge))
        )
        self.assertEqual(libc.prctl(36, 0, 0, 0, 0), 0)

    def test_group_guard_package_hash_mismatch(self):
        # Keep config digest valid while changing the approved script digest.
        # Build directly because group_start pins the real package by design.
        rank = self.d / "rank-0"
        rank.mkdir()
        self.c["receipt"] = str(rank / "guard.json")
        c = dict(
            lease_path=str(self.nic),
            nic_lease=self.c["nic_lease"],
            guard_script=str(CAND / "rping_guard.py"),
            guard_hashes={
                str(CAND / n): hashlib.sha256((CAND / n).read_bytes()).hexdigest()
                for n in ["rping_guard.py", "server_group.py", "lease_contract.py"]
            },
            guards=[dict(rank=0, config=self.c)],
        )
        c["guard_hashes"][str(CAND / "lease_contract.py")] = "0" * 64
        p = self.d / "group.json"
        p.write_text(json.dumps(c))
        q = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(CAND / "server_group.py"),
                str(p),
                hashlib.sha256(p.read_bytes()).hexdigest(),
            ],
            env=self.env,
            pass_fds=(self.held.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.children.append(q)
        out, err = q.communicate(timeout=5)
        self.assertNotEqual(q.returncode, 0, (out, err))
        self.assertIn(b"guard package hash mismatch", err)
        self.assertFalse((self.d / "spawn.json").exists())

    def test_group_cpu_unchanged(self):
        for k in ["uuid", "gpu_lease", "nic_lease", "run_id"]:
            self.c.pop(k)
        # Group has its outer lease identity even when inner mode remains CPU.
        self.c["mode"] = "cpu"
        (self.d / "fixture-mode").write_text("hold")
        from test_guards import identity

        self.c["nic_lease"] = identity(self.nic)
        # group_start supplies outer metadata, then strip only the inner field.
        rank = self.d / "rank-0"
        rank.mkdir()
        self.c["receipt"] = str(rank / "guard.json")
        nic = self.c.pop("nic_lease")
        c = dict(
            lease_path=str(self.nic),
            nic_lease=nic,
            guard_script=str(CAND / "rping_guard.py"),
            guard_hashes={
                str(CAND / n): hashlib.sha256((CAND / n).read_bytes()).hexdigest()
                for n in ["rping_guard.py", "server_group.py", "lease_contract.py"]
            },
            guards=[dict(rank=0, config=self.c)],
        )
        p = self.d / "group.json"
        p.write_text(json.dumps(c))
        q = subprocess.Popen(
            [sys.executable, "-B", str(CAND / "server_group.py"), str(p)],
            env=self.env,
            pass_fds=(self.held.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.children.append(q)
        self.group_ready(q)
        self.assertEqual(json.loads((self.d / "spawn.json").read_text())["cuda"], "")
        self.assertFalse(busy(self.gpu))
        q.stdin.write(b"STOP\n")
        q.stdin.flush()
        q.communicate(timeout=8)
        self.assertEqual(q.returncode, 0)
        self.assertTrue(busy(self.nic))

    def test_guard_timeout_real_fifteen_seconds(self):
        (self.d / "fixture-mode").write_text("hold")
        q = self.start()
        self.wait_for("spawn.json")
        r = self.finish(q, 2)
        self.assertIn("TimeoutExpired", r["error"])
        self.assertFalse(Path("/proc", str(r["worker_pid"])).exists())
        self.assertFalse(busy(self.gpu))
        self.assertTrue(busy(self.nic))
        self.assertGreaterEqual(r["seconds"], 15)
        self.assertLess(r["seconds"], 19)

    def test_guard_parent_death_kills_fixture(self):
        # Become a subreaper so the test really waits for its orphan, not a /proc snapshot alone.
        libc = ctypes.CDLL(None, use_errno=True)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        (self.d / "fixture-mode").write_text("hold")
        q = self.start()
        self.wait_for("spawn.json")
        pid = json.loads((self.d / "spawn.json").read_text())["pid"]
        q.kill()
        q.communicate(timeout=3)
        until = time.monotonic() + 3
        status = None
        while time.monotonic() < until:
            got, s = os.waitpid(pid, os.WNOHANG)
            if got == pid:
                status = s
                break
            time.sleep(0.005)
        self.assertIsNotNone(status)
        self.assertTrue(os.WIFSIGNALED(status))
        self.assertEqual(os.WTERMSIG(status), signal.SIGKILL)
        self.assertFalse(busy(self.gpu))
        self.assertTrue(busy(self.nic))
        self.assertFalse(Path("/proc", str(pid)).exists())
        # Guard died before final receipt: keep explicit UNKNOWN, no fabricated exit.
        r = json.loads(Path(self.c["receipt"]).read_text())
        self.assertNotEqual(r["status"], "WORKER_EXITED")
        self.assertFalse(r["automatic_retry"])
        self.assertEqual(libc.prctl(36, 0, 0, 0, 0), 0)
