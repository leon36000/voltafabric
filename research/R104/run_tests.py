"""Persist exact argv, output, and exit status for the R104 CPU suite."""
import json
import pathlib
import subprocess
import sys
root = pathlib.Path(__file__).resolve().parent
(root / ".private-tmp").mkdir(exist_ok=True)
logs = root / "logs"
logs.mkdir(exist_ok=True)
argv = [sys.executable, "-m", "unittest", "-v", "test_workload_bound_compare"]
completed = subprocess.run(argv, cwd=root, capture_output=True, text=True)
log = ("ARGV=" + json.dumps(argv) + "\nCWD=" + str(root) + "\nSTDOUT:\n" + completed.stdout +
       "\nSTDERR:\n" + completed.stderr + "\nEXIT_CODE=" + str(completed.returncode) + "\n")
(logs / "test-run.log").write_text(log, encoding="utf-8")
print(log, end="")
raise SystemExit(completed.returncode)
