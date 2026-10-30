#!/usr/bin/python3
"""C bridge exec-only worker seam: same PID/PDEATHSIG and exact qualified native owner."""

import os
from pathlib import Path
import sys
from case_contract import read_plan, check_side, inherited_parent_death, require

ROOT = Path(__file__).resolve().parent


def main():
    require(__debug__ and sys.argv[1:] == [], "foreign worker wrapper arguments")
    plan = read_plan(
        ROOT, os.environ.get("VF_R61_PLAN_SHA256"), os.environ.get("VF_GPU_TEST_RUN_ID")
    )
    c = check_side(plan, "server", ROOT)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu_uuid"],
        "GPU guard UUID not inherited",
    )
    inherited_parent_death()
    env = dict(os.environ, CUDA_CACHE_DISABLE="1", PYTHONDONTWRITEBYTECODE="1")
    for k in ["LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"]:
        env.pop(k, None)
    os.execve(c["worker"], [c["worker"], "--audit-counters"], env)


if __name__ == "__main__":
    main()
