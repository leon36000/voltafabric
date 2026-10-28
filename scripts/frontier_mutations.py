#!/usr/bin/env python3
"""Four bounded, compilable frontier mutants; never modifies canonical sources."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    base = (source / "adapters/skinny_moe.py").read_text()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "SUMMARY.json").exists():
        raise FileExistsError("inspect existing mutation campaign before repeating")
    cases = [
        (
            "D01",
            '    if bool((~torch.isfinite(compact_x)).any()):\n        raise ValueError("FP16 compact input overflow")\n',
            "tests.adapter.test_a16_numerical.NumericalProperties.test_cast_signs_neighbors_and_nonfinite_stop_before_provider",
        ),
        (
            "D02",
            '    if bool((~torch.isfinite(act)).any()):\n        raise ValueError("nonfinite SwiGLU activation")\n',
            "tests.adapter.test_a16_numerical.NumericalProperties.test_swiglu_signed_overflow_negative_and_cancellation_controls",
        ),
        (
            "D03",
            '    if topk_ids.shape[1] > 1:\n        ordered = topk_ids.sort(dim=1).values\n        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):\n            raise ValueError("duplicate global expert ID within token top-k")\n',
            "tests.adapter.test_a16_numerical.NumericalProperties.test_duplicates_unknown_ids_and_masks_reject_before_selection",
        ),
        (
            "D04",
            "        _validate_sm70_scale_domain(scales, gscales, k)\n",
            "tests.adapter.test_a14_boundaries.A14BoundaryTests.test_runtime_half_scale_domain_is_checked_before_any_provider",
        ),
    ]
    results = []
    for name, guard, test in cases:
        assert base.count(guard) == 1, (name, "guard anchor differs")
        mutant = base.replace(guard, "")
        ast.parse(mutant)
        directory = out / name
        (directory / "adapters").mkdir(parents=True, exist_ok=False)
        (directory / "adapters/__init__.py").write_text("")
        path = directory / "adapters/skinny_moe.py"
        path.write_text(mutant)
        driver = (
            "import hashlib,json,sys,unittest; import adapters.skinny_moe as a; "
            "from pathlib import Path; "
            "print('MODULE='+json.dumps({'path':a.__file__,'sha256':hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()}),flush=True); "
            "s=unittest.defaultTestLoader.loadTestsFromName(sys.argv[1]); "
            "r=unittest.TextTestRunner(verbosity=2).run(s); "
            "print('MUTATION_RESULT='+json.dumps({'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors)}),flush=True); "
            "sys.exit(0 if r.wasSuccessful() else 1)"
        )
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(source),
            PYTHONDONTWRITEBYTECODE="1",
            VF_BINARY=str(args.binary.resolve()),
        )
        argv = [sys.executable, "-c", driver, test]
        run = subprocess.run(
            argv,
            cwd=directory,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=45,
        )
        (directory / "LOG.txt").write_text(run.stdout)
        module = json.loads(
            next(
                line[7:]
                for line in run.stdout.splitlines()
                if line.startswith("MODULE=")
            )
        )
        result = json.loads(
            next(
                line[16:]
                for line in run.stdout.splitlines()
                if line.startswith("MUTATION_RESULT=")
            )
        )
        assert Path(module["path"]).resolve() == path.resolve(), module
        assert module["sha256"] == hashlib.sha256(mutant.encode()).hexdigest()
        killed = (
            run.returncode == 1
            and result["tests"] == 1
            and result["failures"] > 0
            and result["errors"] == 0
        )
        results.append(
            dict(
                mutant=name,
                source_sha256=module["sha256"],
                compiled=True,
                argv=argv,
                cwd=str(directory),
                exit_code=run.returncode,
                expected="assertion failure, no setup errors",
                observed=result,
                killed=killed,
            )
        )
        assert killed, (name, run.stdout)
    report = dict(
        status="PASS",
        domain="four removed Python frontier guards",
        canonical_source_sha256=hashlib.sha256(base.encode()).hexdigest(),
        mutants=results,
        cuda_executed=False,
    )
    (out / "SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(status="PASS", mutants_killed=len(results), cuda_executed=False)
        )
    )


if __name__ == "__main__":
    main()
