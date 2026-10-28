#!/usr/bin/env python3
"""Oracle sensitivity matrix: run every CPU oracle against deliberately broken binaries.

A mutation campaign asks whether tests notice a removed guard. This asks the
complementary question about the oracles: given a binary broken on one path,
which oracles notice? An oracle that never reacts to any mutant on its own path
asserts nothing. A control row with the canonical binary must pass every oracle,
otherwise no cell of the matrix is valid. Canonical sources are never modified.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rust_mutations  # noqa: E402

# The same oracle invocations verify.py uses, keyed by its step names. crosscheck
# uses fewer seeds than the gate so the matrix stays affordable; a detection with
# fewer seeds is a lower bound on the gate's own sensitivity.
ORACLES = {
    "audit_regression_oracle": ["scripts/a13_crosscheck.py", "--out", "{out}"],
    "independent_oracle": ["scripts/crosscheck.py", "--seeds", "16", "--out", "{out}"],
    "independent_planner": ["scripts/planner_crosscheck.py", "--out", "{out}"],
    "resident_buffers_oracle": ["scripts/resident_crosscheck.py", "--out", "{out}"],
    "real_checkpoint_codec": ["scripts/checkpoint_crosscheck.py", "--out", "{out}"],
    "full_expert_oracle": ["scripts/full_expert_crosscheck.py", "--out", "{out}"],
    "completion_output_oracle": ["scripts/completion_crosscheck.py", "--out", "{out}"],
    "launch_metadata_oracle": [
        "scripts/launch_crosscheck.py",
        "--validator",
        "{validator}",
        "--out",
        "{out}",
    ],
    "sm70_scale_domain_oracle": ["scripts/a14_crosscheck.py"],
    "inspection_output_oracle": ["scripts/inspection_crosscheck.py"],
    "a16_weighted_score_oracle": [
        "scripts/a16_planner_crosscheck.py",
        "--out",
        "{out}",
    ],
}


def run_oracles(python, source, binary, validator, scratch):
    row = {}
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    env.pop("TMPDIR", None)
    for step, template in ORACLES.items():
        out = scratch / f"{step}.json"
        argv = [str(python), template[0], "--binary", str(binary)] + [
            a.format(out=out, validator=validator) for a in template[1:]
        ]
        run = subprocess.run(
            argv,
            cwd=source,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
        )
        (scratch / f"{step}.log").write_text(run.stdout)
        crashed = "Traceback" in run.stdout and "AssertionError" not in run.stdout
        row[step] = {
            "exit_code": run.returncode,
            "detected": run.returncode != 0 and not crashed,
            "crashed_without_assertion": run.returncode != 0 and crashed,
        }
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--canonical-binary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-dir", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated mutant ids")
    args = ap.parse_args()
    source = args.source.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    target_base = args.target_dir.resolve()
    target_base.mkdir(parents=True, exist_ok=True)
    validator = out / "libvf_validate.so"
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-shared",
            "-fPIC",
            str(source / "cuda/qpn2_moe_validate.c"),
            "-o",
            str(validator),
        ],
        check=True,
    )

    control_dir = out / "CONTROL"
    control_dir.mkdir()
    control = run_oracles(
        args.python,
        source,
        args.canonical_binary.resolve(strict=True),
        validator,
        control_dir,
    )
    failing = [k for k, v in control.items() if v["exit_code"] != 0]
    print(f"CONTROL canonical binary: oracles failing={failing}", flush=True)
    if failing:
        (out / "MATRIX.json").write_text(
            json.dumps(dict(status="CONTROL_FAILED", control=control), indent=2) + "\n"
        )
        return 1

    wanted = {x for x in args.only.split(",") if x}
    matrix = {}
    for case in rust_mutations.CASES:
        name, rel, clause = case[0], case[1], case[2]
        if wanted and name not in wanted:
            continue
        if not rel.startswith("src/"):
            continue
        replacement = case[5] if len(case) > 5 else ""
        base = (source / rel).read_text()
        assert base.count(clause) == 1, (name, "anchor count")
        tree = out / name / "tree"
        rust_mutations.copy_source(source, tree)
        (tree / rel).write_text(base.replace(clause, replacement))
        target = target_base / name
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(target)
        build = subprocess.run(
            ["cargo", "build", "--release", "--locked", "--bin", "vf"],
            cwd=tree,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
        )
        (out / name / "BUILD.log").write_text(build.stdout)
        if build.returncode != 0:
            matrix[name] = {"file": rel, "built": False}
            print(f"{name} {rel} built=False", flush=True)
            shutil.rmtree(target, ignore_errors=True)
            continue
        row = run_oracles(
            args.python, source, target / "release/vf", validator, out / name
        )
        detected = sorted(k for k, v in row.items() if v["detected"])
        matrix[name] = {
            "file": rel,
            "built": True,
            "oracles": row,
            "detected_by": detected,
        }
        print(f"{name} {rel} detected_by={detected}", flush=True)
        shutil.rmtree(target, ignore_errors=True)

    never = sorted(
        step
        for step in ORACLES
        if not any(
            row.get("built") and row["oracles"][step]["detected"]
            for row in matrix.values()
        )
    )
    report = dict(
        schema_version=1,
        status="MATRIX_COMPLETE",
        control="every oracle passes on the canonical binary",
        mutants=matrix,
        oracles_that_detected_nothing=never,
        cuda_executed=False,
    )
    (out / "MATRIX.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], silent_oracles=never)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
