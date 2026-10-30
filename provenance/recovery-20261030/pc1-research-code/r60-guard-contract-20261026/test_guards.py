"""Real CPU processes/private locks; ONLY nvidia-smi is a declared fake boundary."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent
CAND = Path(os.environ.get("R60_GUARD_DIR", str(ROOT / "candidate")))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def identity(p):
    st = p.stat()
    return dict(path=str(p), dev=st.st_dev, ino=st.st_ino, uid=st.st_uid)


def busy(p):
    with p.open("r+b") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


class Guards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.d = Path(self.temp.name)
        self.nic = self.d / "nic.lease"
        self.nic.touch()
        self.gpu = self.d / "gpu.lease"
        self.gpu.touch()
        self.held = self.nic.open("r+b")
        self.addCleanup(self.held.close)
        fcntl.flock(self.held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.worker = self.d / "fixture.py"
        self.worker.write_text("""#!/usr/bin/python3
import json,os,sys,time
from pathlib import Path
p=Path(__file__).parent; mode=(p/'fixture-mode').read_text()
fds=[]
for n in Path('/proc/self/fd').iterdir():
 try:
  s=n.stat();fds.append([int(n.name),s.st_dev,s.st_ino])
 except FileNotFoundError:pass
(p/'spawn.json').write_text(json.dumps(dict(pid=os.getpid(),parent=os.getppid(),cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),fds=fds)))
print('R24_LISTENING',file=sys.stderr,flush=True)
if mode=='hold':
 while not (p/'release').exists():time.sleep(.005)
raise SystemExit(7 if mode=='fail' else 0)
""")
        self.worker.chmod(0o700)
        (self.d / "fixture-mode").write_text("exit")
        self.cookie = self.d / "cookie"
        self.cookie.write_bytes(b"x" * 64)
        self.cookie.chmod(0o600)
        self.worker_target = self.d / "worker-target"
        self.worker_target.write_text("CPU fixture never executed")
        # Fake executable is CPU-only, uses no NVIDIA library/NVML/GPU call.
        self.smi = self.d / "nvidia-smi"
        self.smi.write_text("""#!/usr/bin/python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['R60_TEST_DIR']);(p/'inventory-seen').touch()
if (p/'swap-gpu').exists():
 q=p/'gpu.lease';q.rename(p/'old-gpu');q.touch()
if (p/'swap-nic').exists():
 q=p/'nic.lease';q.rename(p/'old-nic');q.touch()
if (p/'inventory-fail').exists():raise SystemExit(9)
u='GPU-CPU-FIXTURE' if not (p/'wrong-uuid').exists() else 'GPU-WRONG'
print('<nvidia_smi_log><driver_version>CPU_FAKE</driver_version><gpu><uuid>'+u+'</uuid><product_name>CPU_FIXTURE</product_name><processes/></gpu></nvidia_smi_log>')
""")
        self.smi.chmod(0o700)
        self.env = dict(
            os.environ,
            PATH=str(self.d) + ":" + os.environ["PATH"],
            R60_TEST_DIR=str(self.d),
            PYTHONDONTWRITEBYTECODE="1",
        )
        self.c = dict(
            mode="gpu_rdma",
            run_id="r60-cpu-" + hashlib.sha256(self.d.name.encode()).hexdigest()[:16],
            receipt=str(self.d / "guard.json"),
            argv=[
                str(self.worker),
                "server",
                "10.40.13.1",
                "18915",
                str(self.cookie),
                str(self.worker_target),
            ],
            hashes={
                str(self.worker): digest(self.worker),
                str(self.cookie): digest(self.cookie),
                str(self.worker_target): digest(self.worker_target),
            },
            uuid="GPU-CPU-FIXTURE",
            nic_lease=identity(self.nic),
            gpu_lease=identity(self.gpu),
        )
        self.children = []
        self.addCleanup(self.archive)
        self.addCleanup(self.reap)

    def archive(self):
        root = Path(os.environ.get("R60_TEST_ARCHIVE", str(ROOT / "evidence/cases-r2")))
        d = root / self.id().split(".")[-1]
        d.mkdir(parents=True, exist_ok=False)
        for p in self.d.rglob("*.json"):
            rel = p.relative_to(self.d)
            q = d / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(p.read_bytes())
        (d / "processes.json").write_text(
            json.dumps(
                [
                    dict(
                        pid=q.pid,
                        returncode=q.poll(),
                        present=Path("/proc", str(q.pid)).exists(),
                    )
                    for q in self.children
                ]
            )
        )
        (d / "markers.json").write_text(
            json.dumps(
                dict(
                    inventory_seen=(self.d / "inventory-seen").exists(),
                    gpu_replaced=(self.d / "old-gpu").exists(),
                    nic_replaced=(self.d / "old-nic").exists(),
                )
            )
        )

    def reap(self):
        for q in self.children:
            if q.poll() is None:
                os.killpg(q.pid, signal.SIGTERM)
                try:
                    q.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(q.pid, signal.SIGKILL)
                    q.communicate(timeout=3)

    def start(self, config=None, pass_fds=None):
        payload = json.dumps(config or self.c)
        q = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(CAND / "rping_guard.py"),
                "--config",
                payload,
                "--config-sha",
                getattr(
                    self, "sha_override", hashlib.sha256(payload.encode()).hexdigest()
                ),
            ],
            env=self.env,
            pass_fds=(self.held.fileno(),) if pass_fds is None else pass_fds,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.children.append(q)
        return q

    def finish(self, q, expected):
        out, err = q.communicate(timeout=20)
        self.assertEqual(q.returncode, expected, (out, err))
        return json.loads(Path(self.c["receipt"]).read_text())

    def wait_for(self, name):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            if (self.d / name).exists():
                return
            time.sleep(0.005)
        self.fail("missing " + name)

    def refused(self, config=None, pass_fds=None):
        q = self.start(config, pass_fds)
        r = self.finish(q, 2)
        self.assertFalse((self.d / "spawn.json").exists(), r)
        self.assertFalse(busy(self.gpu))
        self.assertTrue(busy(self.nic))
        self.assertIsNone(r["worker_exit_code"])
        return r

    def test_nominal_gpu_keeps_nic_and_gpu_until_exit(self):
        (self.d / "fixture-mode").write_text("hold")
        q = self.start()
        self.wait_for("spawn.json")
        spawned = json.loads((self.d / "spawn.json").read_text())
        self.assertEqual(spawned["cuda"], "GPU-CPU-FIXTURE")
        for spec in [self.c["nic_lease"], self.c["gpu_lease"]]:
            self.assertEqual(
                sum(
                    (dev, ino) == (spec["dev"], spec["ino"])
                    for fd, dev, ino in spawned["fds"]
                ),
                1,
            )
        self.assertTrue(busy(self.nic))
        self.assertTrue(busy(self.gpu))
        (self.d / "release").touch()
        r = self.finish(q, 0)
        self.assertEqual(r["exit_code"], 0)
        self.assertFalse(busy(self.gpu))
        self.assertTrue(busy(self.nic), "inner must not unlock shared NIC OFD")
        self.held.close()
        self.assertFalse(busy(self.nic))

    def test_cpu_nominal_is_unchanged_and_hides_cuda(self):
        c = {k: self.c[k] for k in ["receipt", "argv", "hashes"]}
        c["mode"] = "cpu"
        self.finish(self.start(c), 0)
        self.assertEqual(json.loads((self.d / "spawn.json").read_text())["cuda"], "")
        self.assertFalse((self.d / "inventory-seen").exists())

    def test_cpu_masquerade_gpu_fields_refused(self):
        self.c["mode"] = "cpu"
        self.refused()

    def test_missing_run_identity(self):
        self.c.pop("run_id")
        self.refused()

    def test_invalid_run_identity_underscore(self):
        self.c["run_id"] = "r60-cpu-invalid_identity"
        r = self.refused()
        self.assertIn("run identity", r["error"])

    def test_missing_uuid(self):
        self.c.pop("uuid")
        self.refused()

    def test_empty_hashes(self):
        self.c["hashes"] = {}
        self.refused()

    def test_relative_executable(self):
        self.c["argv"][0] = "python3"
        self.refused()

    def test_private_cpu_lease_rejected_in_gpu_mode(self):
        self.c["cpu_lease"] = str(self.d / "cpu.lease")
        self.refused()
        self.assertFalse((self.d / "cpu.lease").exists())

    def test_external_config_digest_mismatch(self):
        self.sha_override = "0" * 64
        r = self.refused()
        self.assertIn("external launch config SHA mismatch", r["error"])
        self.assertFalse((self.d / "inventory-seen").exists())

    def test_wrong_server_role(self):
        self.c["argv"][1] = "client"
        self.refused()

    def test_wrong_data_ip(self):
        self.c["argv"][2] = "127.0.0.1"
        self.refused()

    def test_out_of_range_port(self):
        self.c["argv"][3] = "18920"
        self.refused()

    def test_missing_cookie_hash(self):
        self.c["hashes"].pop(str(self.cookie))
        self.refused()

    def test_missing_executable_hash(self):
        self.c["hashes"].pop(str(self.worker))
        self.refused()

    def test_shell_argv_refused(self):
        self.c["argv"] = ["/bin/sh", "-c", "touch " + str(self.d / "spawn.json")]
        self.c["hashes"]["/bin/sh"] = digest(Path("/bin/sh"))
        self.refused()

    def test_unknown_mode_refused(self):
        self.c["mode"] = "gpu"
        self.refused()

    def test_nic_missing_fd(self):
        self.refused(pass_fds=())

    def test_nic_duplicate_fd(self):
        dup = os.dup(self.held.fileno())
        try:
            self.refused(pass_fds=(self.held.fileno(), dup))
        finally:
            os.close(dup)

    def test_nic_unlocked_fd(self):
        fcntl.flock(self.held, fcntl.LOCK_UN)
        q = self.start()
        r = self.finish(q, 2)
        self.assertFalse((self.d / "spawn.json").exists(), r)

    def test_nic_shared_flock_refused(self):
        fcntl.flock(self.held, fcntl.LOCK_UN)
        fcntl.flock(self.held, fcntl.LOCK_SH | fcntl.LOCK_NB)
        q = self.start()
        r = self.finish(q, 2)
        self.assertFalse((self.d / "spawn.json").exists(), r)
        self.assertIn("exclusive inherited NIC flock required", r["error"])

    def test_nic_posix_lock_refused(self):
        fcntl.flock(self.held, fcntl.LOCK_UN)
        fcntl.lockf(self.held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q = self.start()
        r = self.finish(q, 2)
        self.assertFalse((self.d / "spawn.json").exists(), r)
        self.assertIn("exclusive inherited NIC flock required", r["error"])

    def test_nic_stale_pin(self):
        self.c["nic_lease"]["ino"] += 1
        self.refused()

    def test_nic_symlink(self):
        p = self.d / "nic-alias"
        p.symlink_to(self.nic)
        self.c["nic_lease"]["path"] = str(p)
        self.refused()

    def test_gpu_symlink(self):
        p = self.d / "gpu-alias"
        p.symlink_to(self.gpu)
        self.c["gpu_lease"]["path"] = str(p)
        self.refused()

    def test_gpu_uid_pin(self):
        self.c["gpu_lease"]["uid"] += 1
        self.refused()

    def test_gpu_stale_pin(self):
        self.c["gpu_lease"]["ino"] += 1
        self.refused()

    def test_gpu_nic_inode_alias(self):
        self.c["gpu_lease"] = identity(self.nic)
        self.refused()

    def test_gpu_contention(self):
        with self.gpu.open("r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            q = self.start()
            r = self.finish(q, 2)
            self.assertFalse((self.d / "spawn.json").exists(), r)
            self.assertTrue(busy(self.gpu))
            self.assertTrue(busy(self.nic))

    def test_gpu_missing_is_not_created(self):
        self.gpu.unlink()
        q = self.start()
        r = self.finish(q, 2)
        self.assertFalse(self.gpu.exists())
        self.assertFalse((self.d / "spawn.json").exists(), r)

    def test_gpu_fifo_is_not_opened_blocking(self):
        self.gpu.unlink()
        os.mkfifo(self.gpu)
        self.c["gpu_lease"] = identity(self.gpu)
        q = self.start()
        r = self.finish(q, 2)
        self.assertFalse((self.d / "spawn.json").exists(), r)

    def test_gpu_replaced_by_inventory_before_spawn(self):
        (self.d / "swap-gpu").touch()
        self.refused()

    def test_nic_replaced_by_inventory_before_spawn(self):
        (self.d / "swap-nic").touch()
        q = self.start()
        r = self.finish(q, 2)
        self.assertFalse((self.d / "spawn.json").exists(), r)
        self.assertFalse(busy(self.gpu))
        self.assertFalse(busy(self.nic))
        self.assertTrue(busy(self.d / "old-nic"))

    def test_inventory_exit_failure(self):
        (self.d / "inventory-fail").touch()
        self.refused()

    def test_inventory_wrong_uuid(self):
        (self.d / "wrong-uuid").touch()
        self.refused()

    def test_hash_mismatch_before_inventory(self):
        self.c["hashes"][str(self.worker)] = "0" * 64
        self.refused()
        self.assertFalse((self.d / "inventory-seen").exists())

    def test_child_error_propagates(self):
        (self.d / "fixture-mode").write_text("fail")
        r = self.finish(self.start(), 2)
        self.assertEqual(r["exit_code"], 7)
        self.assertEqual(r["worker_exit_code"], 7)
        self.assertFalse(busy(self.gpu))
        self.assertTrue(busy(self.nic))

    def test_guard_sigterm_reaps_child(self):
        (self.d / "fixture-mode").write_text("hold")
        q = self.start()
        self.wait_for("spawn.json")
        pid = json.loads((self.d / "spawn.json").read_text())["pid"]
        q.terminate()
        r = self.finish(q, 2)
        self.assertFalse(Path("/proc", str(pid)).exists(), r)
        self.assertFalse(busy(self.gpu))
        self.assertTrue(busy(self.nic))


if __name__ == "__main__":
    unittest.main()
