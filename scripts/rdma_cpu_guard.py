"""One bounded, leased worker launch; receipts distinguish exit from result validity."""

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time
import xml.etree.ElementTree as ET


def kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def checked_inventory(raw, expected_uuid):
    if not raw or len(raw) > 1024 * 1024 or b"<!ENTITY" in raw:
        raise ValueError("invalid inventory bounds")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError("malformed inventory") from exc

    def one(parent, name):
        nodes = parent.findall(name)
        if len(nodes) != 1 or not (nodes[0].text or "").strip():
            raise ValueError("missing or ambiguous " + name)
        text = nodes[0].text.strip()
        if text == "N/A":
            raise ValueError("unavailable " + name)
        return text

    if root.tag != "nvidia_smi_log" or len(root.findall("gpu")) != 1:
        raise ValueError("ambiguous GPU inventory")
    gpu = root.find("gpu")
    uuid = one(gpu, "uuid")
    if uuid != expected_uuid:
        raise ValueError("GPU identity mismatch")
    name, driver = one(gpu, "product_name"), one(root, "driver_version")
    lists = gpu.findall("processes")
    if len(lists) != 1 or (lists[0].text or "").strip():
        raise ValueError("unavailable process list")
    for process in lists[0]:
        if process.tag != "process_info":
            raise ValueError("unknown process entry")
        kind, pid = one(process, "type"), one(process, "pid")
        if not pid.isdecimal() or int(pid) <= 0:
            raise ValueError("invalid process identity")
        # Existing policy checks compute applications; graphics-only is allowed.
        if kind != "G":
            raise ValueError("compute GPU occupied or unknown process type")
    return uuid + ", " + name + ", " + driver


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    c = json.loads(a.config)
    receipt = Path(c["receipt"])
    receipt.parent.mkdir(parents=True, exist_ok=True)
    r = dict(
        status="UNKNOWN_NO_REPLAY",
        automatic_retry=False,
        pid=os.getpid(),
        host=socket.gethostname(),
        config=c,
    )
    with receipt.open("x") as f:
        json.dump(r, f, indent=2)
    q = None
    lease = None
    start = time.monotonic()
    r["phase_ns"] = {"guard_start": time.monotonic_ns()}

    def save():
        tmp = receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, indent=2) + "\n")
        tmp.replace(receipt)

    def stop(signum, frame):
        r["termination_signal"] = signum
        if q is not None and q.poll() is None:
            kill_group(q.pid)
        raise InterruptedError("worker guard interrupted")

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        if not __debug__:
            raise ValueError("optimized Python is unsupported by the CPU RDMA guard")
        if c.get("mode") != "cpu":
            raise ValueError(
                "CPU RDMA guard only; GPU launches require their separate gate"
            )
        for name, sha in c["hashes"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, (
                "binary/library identity mismatch"
            )
        r["phase_ns"]["hashes_done"] = time.monotonic_ns()
        env = dict(os.environ)
        if c["mode"] == "gpu":
            lease = open(c["lease"], "a+b")
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            r["phase_ns"]["lease_acquired"] = time.monotonic_ns()
            uuid = c["uuid"]
            inventory = subprocess.run(
                ["nvidia-smi", "-q", "-x", "-i", uuid],
                capture_output=True,
                timeout=10,
            )
            r["phase_ns"]["single_inventory_done"] = time.monotonic_ns()
            assert inventory.returncode == 0, "GPU inventory query failed"
            r["gpu_inventory"] = checked_inventory(inventory.stdout, uuid)
            env["CUDA_VISIBLE_DEVICES"] = uuid
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            if c.get("cpu_lease"):
                lease = open(c["cpu_lease"], "a+b")
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Parent death kills the worker even if SSH itself was abruptly lost.
        parent = os.getpid()

        def child_setup():
            libc = ctypes.CDLL(None)
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
                os._exit(125)

        r["phase_ns"]["spawn_begin"] = time.monotonic_ns()
        q = subprocess.Popen(
            c["argv"],
            env=env,
            start_new_session=True,
            pass_fds=(() if lease is None else (lease.fileno(),)),
            preexec_fn=child_setup,
        )
        r["phase_ns"]["spawn_end"] = time.monotonic_ns()
        r.update(worker_pid=q.pid, status="WORKER_STARTED_UNKNOWN_UNTIL_TERMINAL")
        save()
        code = q.wait(timeout=15)
        r["phase_ns"]["worker_exit"] = time.monotonic_ns()
        r.update(status="WORKER_EXITED", exit_code=code)
    except Exception as e:
        r["error"] = type(e).__name__ + ": " + str(e)
    finally:
        if q is not None and q.poll() is None:
            kill_group(q.pid)
            try:
                q.wait(timeout=2)
            except subprocess.TimeoutExpired:
                r["status"] = "UNKNOWN_UNREAPED_WORKER_NO_REPLAY"
        r.update(
            seconds=time.monotonic() - start,
            worker_exit_code=None if q is None else q.returncode,
            maxrss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        r["phase_ns"]["guard_end"] = time.monotonic_ns()
        save()
        if lease is not None:
            lease.close()
    return 0 if r.get("exit_code") == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
