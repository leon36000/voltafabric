#!/usr/bin/python3
"""Single server group entry: exact R60 GPU guard, pinned NIC and direct bridge; no retries."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from case_contract import read_plan, check_side, acquire_nic, parent_death, require

ROOT = Path(__file__).resolve().parent


def main():
    require(__debug__, "optimized adapter refused")
    p = argparse.ArgumentParser()
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--namespace", required=True)
    args = p.parse_args()
    plan = read_plan(ROOT, args.plan_sha256, args.namespace)
    c = check_side(plan, "server", ROOT)
    # A consumed launch marker is not a model outcome; loss after here is never replayed.
    with (ROOT / "LAUNCH_ONCE.json").open("x") as f:
        json.dump(
            dict(
                status="PLAN_VALIDATED_NO_LEASE_OR_GROUP_STARTED_NO_REPLAY",
                pid=os.getpid(),
                plan_sha256=args.plan_sha256,
                namespace=args.namespace,
                automatic_retry=False,
            ),
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    (ROOT / "rank-7").mkdir(exist_ok=False)
    n = plan["network"]
    hashes = {
        **c["hashes"],
        str(ROOT / "CASE_PLAN.json"): args.plan_sha256,
        c["inventory"]["path"]: c["inventory"]["sha256"],
    }
    guard = dict(
        mode="gpu_rdma",
        run_id=args.namespace,
        receipt=str(ROOT / "rank-7/guard.json"),
        argv=[
            c["bridge"],
            "server",
            n["server_ip"],
            str(n["port"]),
            c["cookie"],
            c["wrapper"],
        ],
        hashes=hashes,
        uuid=plan["gpu_uuid"],
        inventory_executable=c["inventory"]["path"],
        nic_lease=c["nic_lease"],
        gpu_lease=c["gpu_lease"],
    )
    scripts = {
        str(ROOT / "guards" / name): c["hashes"][str(ROOT / "guards" / name)]
        for name in ["rping_guard.py", "server_group.py", "lease_contract.py"]
    }
    group = dict(
        lease_path=c["nic_lease"]["path"],
        nic_lease=c["nic_lease"],
        guard_script=str(ROOT / "guards/rping_guard.py"),
        guard_hashes=scripts,
        guards=[dict(rank=7, config=guard)],
    )
    raw = (json.dumps(group, indent=2) + "\n").encode()
    config = ROOT / "GROUP_CONFIG.json"
    with config.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    env = dict(
        os.environ,
        VF_R61_PLAN_SHA256=args.plan_sha256,
        VF_GPU_TEST_RUN_ID=args.namespace,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
    )
    for k in ["LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"]:
        env.pop(k, None)
    fd = acquire_nic(c["nic_lease"])
    state = dict(
        status="NIC_HELD_GPU_GATE_PENDING_NO_MODEL_RESULT",
        pid=os.getpid(),
        plan_sha256=args.plan_sha256,
        namespace=args.namespace,
        automatic_retry=False,
    )
    temp = ROOT / "LAUNCH_ONCE.lease-state.tmp"
    with temp.open("x") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, ROOT / "LAUNCH_ONCE.json")
    try:
        parent_death()
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        argv = [
            sys.executable,
            "-B",
            str(ROOT / "guards/server_group.py"),
            str(config),
            hashlib.sha256(raw).hexdigest(),
        ]
        os.execve(sys.executable, argv, env)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
