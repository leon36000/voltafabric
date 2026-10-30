"""Private Linux/cooperative lease admission. No GPU/NIC calls and no inner NIC unlock."""

import os
import hashlib
from pathlib import Path
import re
import stat


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"path", "dev", "ino", "uid"}:
        raise ValueError("exact lease specification required")
    if not isinstance(spec["path"], str) or not Path(spec["path"]).is_absolute():
        raise ValueError("absolute lease path required")
    if any(type(spec[n]) is not int or spec[n] < 0 for n in ["dev", "ino", "uid"]):
        raise ValueError("invalid lease identity")
    if spec["uid"] != os.getuid():
        raise ValueError("lease UID mismatch")


def path_info(spec):
    validate_spec(spec)
    st = os.stat(spec["path"], follow_symlinks=False)
    if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino, st.st_uid) != (
        spec["dev"],
        spec["ino"],
        spec["uid"],
    ):
        raise ValueError("lease path identity/type changed")
    return st


def check_fd(fd, spec):
    path_info(spec)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino, st.st_uid) != (
        spec["dev"],
        spec["ino"],
        spec["uid"],
    ):
        raise ValueError("held lease identity/type changed")
    return dict(dev=st.st_dev, ino=st.st_ino, uid=st.st_uid, fd=fd)


def require_nic_lock(fd):
    # Linux fdinfo identifies the lock associated with THIS open file description.
    # No second open/flock and, critically, no LOCK_UN on the shared description.
    lines = Path("/proc/self/fdinfo", str(fd)).read_text().splitlines()
    locks = [line.split() for line in lines if line.startswith("lock:")]
    if (
        len(locks) != 1
        or len(locks[0]) != 9
        or locks[0][2:5] != ["FLOCK", "ADVISORY", "WRITE"]
        or locks[0][-2:] != ["0", "EOF"]
    ):
        raise ValueError("exclusive inherited NIC flock required")
    st = os.fstat(fd)
    device_inode = locks[0][6].split(":")
    if len(device_inode) != 3 or (
        int(device_inode[0], 16),
        int(device_inode[1], 16),
        int(device_inode[2]),
    ) != (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino):
        raise ValueError("NIC lock identity mismatch")


def inherited_nic(spec):
    path_info(spec)
    fds = []
    for p in Path("/proc/self/fd").iterdir():
        try:
            st = os.fstat(int(p.name))
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == (spec["dev"], spec["ino"]):
            fds.append(int(p.name))
    if len(fds) != 1 or fds[0] < 3:
        raise ValueError("one inherited NIC descriptor required")
    fd = fds[0]
    check_fd(fd, spec)
    require_nic_lock(fd)
    return fd


def open_gpu(spec, nic):
    path_info(spec)
    if (spec["dev"], spec["ino"]) == (nic["dev"], nic["ino"]):
        raise ValueError("GPU/NIC inode alias forbidden")
    # NONBLOCK prevents a raced FIFO open from hanging; fstat rechecks its type.
    fd = os.open(spec["path"], os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        check_fd(fd, spec)
        return os.fdopen(fd, "r+b")
    except BaseException:
        os.close(fd)
        raise


def validate_mode(c):
    if c.get("mode") == "cpu":
        if any(
            k in c
            for k in ["gpu_lease", "nic_lease", "uuid", "lease", "inventory_executable"]
        ):
            raise ValueError("CPU mode cannot carry GPU admission")
        return
    if c.get("mode") != "gpu_rdma" or c.get("cpu_lease"):
        raise ValueError("explicit gpu_rdma admission required")
    if not isinstance(c.get("run_id"), str) or not re.fullmatch(
        r"r[0-9]+-[a-zA-Z0-9-]{8,100}", c["run_id"]
    ):
        raise ValueError("fresh explicit run identity required")
    if not isinstance(c.get("uuid"), str) or not re.fullmatch(
        r"GPU-[a-zA-Z0-9-]{8,100}", c["uuid"]
    ):
        raise ValueError("explicit GPU UUID required")
    for n in ["nic_lease", "gpu_lease"]:
        validate_spec(c[n])
    argv = c.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) != 6
        or any(not isinstance(a, str) or "\0" in a for a in argv)
    ):
        raise ValueError("exact direct RDMA server argv required")
    if (
        argv[1] != "server"
        or argv[2] not in ["10.40.13.1", "10.40.13.2"]
        or argv[3] not in [str(n) for n in range(18900, 18920)]
    ):
        raise ValueError("RDMA server role/IP/port contract mismatch")
    if any(not Path(argv[n]).is_absolute() for n in [0, 4, 5]):
        raise ValueError("absolute bridge/cookie/worker required")
    hashes = c.get("hashes")
    if (
        not isinstance(hashes, dict)
        or not hashes
        or any(
            not isinstance(p, str)
            or not Path(p).is_absolute()
            or not isinstance(h, str)
            or not re.fullmatch("[0-9a-f]{64}", h)
            for p, h in hashes.items()
        )
    ):
        raise ValueError("absolute artifact paths and SHA256 required")
    if any(argv[n] not in hashes for n in [0, 4, 5]):
        raise ValueError("bridge/cookie/worker hashes required")


def validate_group(c):
    if len(c["guards"]) != 1:
        raise ValueError("initial gpu_rdma group supports exactly one guard")
    g = c["guards"][0]["config"]
    validate_mode(g)
    if (
        g["mode"] != "gpu_rdma"
        or c["lease_path"] != c["nic_lease"]["path"]
        or g["nic_lease"] != c["nic_lease"]
    ):
        raise ValueError("group/guard NIC contract mismatch")
    scripts = {
        str(Path(__file__).with_name(n))
        for n in ["rping_guard.py", "server_group.py", "lease_contract.py"]
    }
    if (
        c.get("guard_script") != str(Path(__file__).with_name("rping_guard.py"))
        or set(c.get("guard_hashes", {})) != scripts
    ):
        raise ValueError("exact sibling guard package required")
    for name, value in c["guard_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != value:
            raise ValueError("guard package hash mismatch")
    path_info(c["nic_lease"])
