# R104 private adapter update

## Change

`workload_bound_compare.py` now reads each R97 quality evidence path itself before invoking the frozen comparator. The read uses `O_NONBLOCK`, checks `fstat()` for a regular file, reads at most `LIMIT + 1` bytes, and hashes that exact byte snapshot. A FIFO is rejected promptly. The declared quality SHA must match before proceeding.

For the private R97 comparison only, the adapter writes the already verified quality bytes into a `TemporaryDirectory` (mode 0700) and changes `quality.evidence_path` in the two temporary run copies to those private files. It preserves each declared SHA and the quality bytes. The original run snapshots and their SHA-256 values remain recorded as read; the original run files and quality files are not rewritten. This path rewrite exists only to prevent the frozen R97 reader from blocking on a special file. R97 still performs its own parsing, schema, digest, and pairwise checks against the snapshots.

The prior adapter source is preserved at `before/workload_bound_compare.py`; its SHA-256 is recorded in `before/SHA256SUMS`. The frozen R97 and R103 snapshots were not changed.

## Verification

Command: `python3 run_tests.py`

Result: exit code 0; 9 persistent CPU-only unittest cases passed. The runner creates `.private-tmp` if absent, then stores exact argv, stdout/stderr, and exit code in `logs/test-run.log`. The final runner-only mkdir change was made after the passing suite; the 9 cases were not replayed for that change. The FIFO subprocess argv, elapsed time, output, and exit code are stored in `logs/fifo-check.log`; it returned `NON_COMPARABLE` in about 0.02 s, exit 0, with reason `quality evidence is not a regular file`.

Cases: positive synthetic pair and workload linkage; right-run workload mismatch; corpus substitution after snapshot/hash; quality substitution after hash; duplicate JSON key; boolean token count; incorrect quality hash; FIFO timeout-bound rejection; and R97 incomparable package pair.

`python3 -m py_compile workload_bound_compare.py test_workload_bound_compare.py run_tests.py` also succeeded. All fixtures are synthetic and make no execution, tokenizer, hardware, or performance claim.

## SHA-256

- Updated `workload_bound_compare.py`: `51f10d3745314acdf49c37794c9cb7ae30b00632b9b88c669a49f4c127b5184d`
- Frozen R97 `r97_compare_snapshot.py`: `e398a0cc0d36da8b25fb971488b73870f045507326c42f19fccf33d812eeb4d2`
- Frozen R103 `workload_admission.py`: `04ca282e61ac18d16aa63e3ec9626937e613b1c50ff5da1c719754f7d75fa3c9`
- Prior adapter: `11726f3b49eb44b5c1869347cbb613e9f7f16da90e28ec53ee73868dc9397f41` (see exact value in `before/SHA256SUMS`)
