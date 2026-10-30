"""Single-case R61 adapter contracts; trusted private binaries, not a plugin sandbox."""

import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
from guards.lease_contract import check_fd, path_info

LIBSHA = "f856c9bdf5c4d021203b2b7ce5b1f393058370e3f5d2d0f05072fb067e739213"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def no_duplicates(pairs):
    d = {}
    for k, v in pairs:
        require(k not in d, "duplicate plan field")
        d[k] = v
    return d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_plan(root, digest, namespace):
    require(
        re.fullmatch("[0-9a-f]{64}", digest or "") is not None,
        "external plan SHA missing",
    )
    p = Path(root) / "CASE_PLAN.json"
    # Read once; the buffer hashed is the only JSON input consumed.
    with p.open("rb") as f:
        raw = f.read(32769)
    require(
        len(raw) <= 32768 and hashlib.sha256(raw).hexdigest() == digest,
        "case plan hash/bound mismatch",
    )
    plan = json.loads(raw, object_pairs_hook=no_duplicates)
    require(
        set(plan)
        == {
            "namespace",
            "logical_requests",
            "network",
            "gpu_uuid",
            "producer_library_sha256",
            "compressed_artifact_bytes",
            "no_automatic_retry",
            "client",
            "server",
        },
        "exact case plan schema",
    )
    require(
        isinstance(namespace, str)
        and re.fullmatch(r"r53-r61-[A-Za-z0-9_-]{8,55}", namespace) is not None
        and len(namespace) <= 64
        and plan["namespace"] == namespace,
        "namespace mismatch",
    )
    require(
        plan["no_automatic_retry"] is True
        and plan["producer_library_sha256"] == LIBSHA
        and type(plan["compressed_artifact_bytes"]) is int
        and plan["compressed_artifact_bytes"] == 6688,
        "fixed resident producer contract",
    )
    require(
        plan["logical_requests"]
        == [
            dict(
                stamp=dict(
                    run_id=namespace,
                    model_revision="r13-tiny-independent-golden",
                    plan_id="tiny-45-independent-cases",
                    layer=0,
                    batch=b,
                    epoch=20,
                ),
                rank=7,
                attempt_id=b + 1,
            )
            for b in range(2)
        ],
        "logical Stamp/attempt mismatch",
    )
    for request in plan["logical_requests"]:
        require(
            all(type(request[k]) is int for k in ["rank", "attempt_id"])
            and all(
                type(request["stamp"][k]) is int for k in ["layer", "batch", "epoch"]
            ),
            "integer identity fields required",
        )
    n = plan["network"]
    require(
        set(n) == {"client_ip", "server_ip", "port", "native_seconds", "guard_seconds"}
        and n["client_ip"] == "10.40.13.2"
        and n["server_ip"] == "10.40.13.1"
        and type(n["port"]) is int
        and 18900 <= n["port"] <= 18919
        and type(n["native_seconds"]) is int
        and n["native_seconds"] == 12
        and type(n["guard_seconds"]) is int
        and n["guard_seconds"] == 15,
        "network role/limit mismatch",
    )
    require(
        isinstance(plan["gpu_uuid"], str)
        and re.fullmatch(r"GPU-[a-zA-Z0-9-]{8,100}", plan["gpu_uuid"]) is not None,
        "GPU UUID missing",
    )
    for name in ["client", "server"]:
        side = plan[name]
        required = {"root", "bridge", "cookie", "nic_lease", "hashes"} | (
            {"worker", "library", "wrapper", "gpu_lease", "inventory"}
            if name == "server"
            else set()
        )
        require(set(side) == required, "exact side schema")
        root_path = Path(side["root"])
        require(root_path.is_absolute(), "absolute case root")
        for key in required - {"root", "hashes", "nic_lease", "gpu_lease", "inventory"}:
            path = Path(side[key])
            require(
                path.is_absolute()
                and path.is_relative_to(root_path)
                and ".." not in path.parts,
                "artifact outside private case root",
            )
        hashes = side["hashes"]
        require(
            isinstance(hashes, dict) and 1 <= len(hashes) <= 64, "artifact hash count"
        )
        require(
            all(
                isinstance(p, str)
                and Path(p).is_absolute()
                and Path(p).is_relative_to(root_path)
                and ".." not in Path(p).parts
                and isinstance(h, str)
                and re.fullmatch("[0-9a-f]{64}", h)
                for p, h in hashes.items()
            ),
            "artifact hash schema",
        )
        require(
            all(
                side[k] in hashes
                for k in required
                - {"root", "hashes", "nic_lease", "gpu_lease", "inventory"}
            ),
            "required artifact hash missing",
        )
    inventory = plan["server"]["inventory"]
    require(
        isinstance(inventory, dict)
        and set(inventory) == {"path", "sha256"}
        and isinstance(inventory["path"], str)
        and Path(inventory["path"]).is_absolute()
        and isinstance(inventory["sha256"], str)
        and re.fullmatch("[0-9a-f]{64}", inventory["sha256"]) is not None,
        "pinned inventory specification required",
    )
    require(
        plan["server"]["hashes"][plan["server"]["library"]] == LIBSHA,
        "compiled library identity mismatch",
    )
    require(
        Path(plan["server"]["library"]).parent == Path(plan["server"]["worker"]).parent
        and Path(plan["server"]["library"]).name == "libvf_r13.so",
        "worker ORIGIN library contract",
    )
    return plan


def check_side(plan, side, root):
    c = plan[side]
    require(c["root"] == str(root), "wrong case root")
    for name, h in c["hashes"].items():
        require(sha(name) == h, "artifact changed: " + name)
    return c


def acquire_nic(spec):
    path_info(spec)
    fd = os.open(spec["path"], os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        check_fd(fd, spec)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_fd(fd, spec)
        os.set_inheritable(fd, True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def parent_death():
    parent = os.getppid()
    require(parent > 1, "parent already absent")
    libc = ctypes.CDLL(None, use_errno=True)
    require(
        libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0 and os.getppid() == parent,
        "parent-death setup",
    )


def inherited_parent_death():
    value = ctypes.c_int()
    libc = ctypes.CDLL(None, use_errno=True)
    require(
        libc.prctl(2, ctypes.byref(value), 0, 0, 0) == 0
        and value.value == signal.SIGKILL
        and os.getppid() > 1,
        "C worker parent-death contract missing",
    )
