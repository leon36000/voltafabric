"""Execute the bounded, complete R60 PRIVATE Python suite; no CUDA/RDMA runtime."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
GROUP_TESTS = [
    "test_group_nominal_stop_holds_both_locks",
    "test_group_stale_nic_no_child",
    "test_group_cpu_unchanged",
    "test_group_external_digest_mismatch",
    "test_group_guard_package_hash_mismatch",
    "test_group_timeout_real_twelve_seconds",
    "test_group_abrupt_death_reaps_guard_and_bridge",
    "test_guard_timeout_real_fifteen_seconds",
    "test_guard_parent_death_kills_fixture",
]
NAMES = (
    ["test_guards.Guards"]
    + ["test_group.GroupContracts." + n for n in GROUP_TESTS]
    + ["test_config_race.ConfigRace.test_one_raw_config_is_hashed_and_consumed"]
)


def manifest():
    paths = [*sorted((ROOT / "candidate").glob("*.py"))] + [
        ROOT / n
        for n in [
            "test_guards.py",
            "test_group.py",
            "test_config_race.py",
            "qualify_cpu.py",
        ]
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def main():
    before = manifest()
    argv = [sys.executable, "-B", "-m", "unittest", "-v", *NAMES]
    evidence = ROOT / "evidence/cpu-qualification"
    evidence.mkdir(exist_ok=False)
    record = dict(
        argv=argv,
        source_before=before,
        status="UNKNOWN_NO_REPLAY",
        automatic_retry=False,
        gpu_executed=False,
        rdma_data_executed=False,
        inventory="CPU_FAKE_EXECUTABLE_ONLY",
        start_ns=time.monotonic_ns(),
    )
    path = evidence / "operation.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    env = dict(
        os.environ, TMPDIR=str(ROOT / "tmp"), R60_TEST_ARCHIVE=str(evidence / "cases")
    )
    p = subprocess.Popen(
        argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    record["pid"] = p.pid
    path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        out, err = p.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate(timeout=3)
        record["timed_out"] = True
    (evidence / "stdout").write_bytes(out)
    (evidence / "stderr").write_bytes(err)
    after = manifest()
    record.update(
        status="PROCESS_TERMINAL",
        exit_code=p.returncode,
        source_after=after,
        source_stable=after == before,
        process_reconciled=True,
        end_ns=time.monotonic_ns(),
    )
    path.write_text(json.dumps(record, indent=2) + "\n")
    if p.returncode != 0 or before != after:
        raise SystemExit(2)
    (ROOT / "private_source_final.json").write_text(
        json.dumps(before, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            dict(
                status="R60_PRIVATE_CPU_SUITE_PASS",
                source_stable=True,
                methods=48,
                scope="Real private files/processes and fake NVIDIA inventory; no GPU/RDMA runtime",
                operation=str(path),
            )
        )
    )


if __name__ == "__main__":
    main()
