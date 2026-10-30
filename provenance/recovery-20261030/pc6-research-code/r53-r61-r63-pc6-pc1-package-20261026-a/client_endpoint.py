#!/usr/bin/python3
"""R55 Endpoint argv seam, direct R59 ARM bridge client; protocol stdout untouched."""

import os
from pathlib import Path
import sys
from case_contract import read_plan, check_side, parent_death, require
from guards.lease_contract import inherited_nic

ROOT = Path(__file__).resolve().parent


def main():
    require(
        __debug__ and sys.argv[1:] == ["--audit-counters"], "foreign endpoint arguments"
    )
    plan = read_plan(
        ROOT, os.environ.get("VF_R61_PLAN_SHA256"), os.environ.get("VF_GPU_TEST_RUN_ID")
    )
    c = check_side(plan, "client", ROOT)
    fd = inherited_nic(c["nic_lease"])
    parent_death()
    os.set_inheritable(fd, True)
    n = plan["network"]
    argv = [c["bridge"], "client", n["client_ip"], str(n["port"]), c["cookie"], "-"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    for k in ["LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"]:
        env.pop(k, None)
    os.execve(argv[0], argv, env)


if __name__ == "__main__":
    main()
