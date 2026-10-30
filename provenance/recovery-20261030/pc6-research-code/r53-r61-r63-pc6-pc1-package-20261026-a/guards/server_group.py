"""One bounded SSH group, one NIC lease inherited from the existing outer guard."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time
from lease_contract import validate_group, inherited_nic, check_fd, require_nic_lock


def write(path, j):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(j, indent=2) + "\n")
    tmp.replace(path)


def main():
    config = Path(sys.argv[1])
    raw = config.read_bytes()
    c = json.loads(raw)
    root = config.parent
    receipt = root / "group-operation.json"
    gpu_group = any(item["config"].get("mode") == "gpu_rdma" for item in c["guards"])
    if gpu_group:
        expected = hashlib.sha256(raw).hexdigest()
        if len(sys.argv) != 3 or sys.argv[2] != expected:
            raise ValueError("external group config SHA mismatch")
        validate_group(c)
        lease_fd = inherited_nic(c["nic_lease"])
    else:
        st = Path(c["lease_path"]).stat()
        fds = []
        for p in Path("/proc/self/fd").iterdir():
            try:
                s = p.stat()
                if (s.st_dev, s.st_ino) == (st.st_dev, st.st_ino):
                    fds.append(int(p.name))
            except FileNotFoundError:
                pass
        assert len(fds) == 1, "exact inherited group lease required"
        lease_fd = fds[0]
        fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    j = dict(
        status="GROUP_RUNNING",
        pid=os.getpid(),
        lease_fd=lease_fd,
        controllers=[],
        automatic_retry=False,
    )
    with receipt.open("x") as f:
        json.dump(j, f)
    children = []
    readers = []
    readiness = []
    errors = []
    stop_requested = False

    def interrupt(sig, frame):
        raise InterruptedError("group stopped; model outcome may be UNKNOWN")

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupt)
    try:
        for item in c["guards"]:
            rank = item["rank"]
            g = item["config"]
            if gpu_group:
                validate_group(c)
                check_fd(lease_fd, c["nic_lease"])
                require_nic_lock(lease_fd)
            else:
                assert not g.get("cpu_lease") and g["mode"] == "cpu"
            cmd = [
                "/usr/bin/setpriv",
                "--pdeathsig",
                "KILL",
                "/usr/bin/python3",
                "-B",
                c["guard_script"],
                "--config",
                json.dumps(g),
            ]
            if gpu_group:
                cmd.extend(
                    ["--config-sha", hashlib.sha256(json.dumps(g).encode()).hexdigest()]
                )
            q = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
                pass_fds=(lease_fd,),
            )
            children.append(q)
            j["controllers"].append(dict(rank=rank, pid=q.pid, argv=cmd))
            write(receipt, j)
            ready = threading.Event()
            readiness.append(ready)

            def drain(process=q, worker=rank, event=ready):
                seen = bytearray()
                try:
                    with (root / f"rank-{worker}" / "server-transport.stderr").open(
                        "xb", buffering=0
                    ) as f:
                        while True:
                            b = os.read(process.stderr.fileno(), 4096)
                            if not b:
                                return
                            if len(seen) + len(b) > 65536:
                                raise ValueError("stderr cap")
                            seen.extend(b)
                            f.write(b)
                            if b"R24_LISTENING\n" in seen:
                                event.set()
                except Exception as exc:
                    errors.append(repr(exc))

            t = threading.Thread(target=drain)
            t.start()
            readers.append(t)
        for e in readiness:
            assert e.wait(4), "a server failed readiness"
        j["ready_ns"] = time.monotonic_ns()
        write(receipt, j)
        print(
            json.dumps(
                dict(
                    status="GROUP_READY",
                    ranks=[x["rank"] for x in c["guards"]],
                    pid=os.getpid(),
                )
            ),
            flush=True,
        )
        until = time.monotonic() + 12
        while any(q.poll() is None for q in children):
            if select.select([0], [], [], 0.03)[0]:
                data = os.read(0, 1024)
                if data == b"STOP\n":
                    stop_requested = True
                    break
                if data == b"":
                    raise InterruptedError("SSH control input closed; no replay")
                raise ValueError("unknown group control")
            if time.monotonic() > until:
                raise TimeoutError("group runtime cap")
    except Exception as exc:
        j["error"] = repr(exc)
    finally:
        for q in children:
            if q.poll() is None:
                q.terminate()
        for q in children:
            try:
                q.wait(timeout=2)
            except subprocess.TimeoutExpired:
                q.kill()
                q.wait(timeout=2)
        for t in readers:
            t.join(timeout=2)
        assert not any(t.is_alive() for t in readers)
        j.update(
            status="GROUP_TERMINAL",
            stop_requested=stop_requested,
            exits=[q.returncode for q in children],
            reader_errors=errors,
            end_ns=time.monotonic_ns(),
        )
        write(receipt, j)
        print(json.dumps(dict(status=j["status"], exits=j["exits"])), flush=True)
    return (
        0
        if not j.get("error")
        and not errors
        and (stop_requested or all(q.returncode == 0 for q in children))
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
