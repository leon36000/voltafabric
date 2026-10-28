# R116 — enforce declared output-token ceiling

## Finding and change

The R104 workload comparator accepted a synthetic one-request record with three generated tokens when both run records declared `max_new_tokens=2`. The data-file hash, request identity, and total token count were internally consistent, but the declared per-request ceiling was violated. The regression test failed before the fix with that exact unexpected verdict. `RED_REPRODUCER.json` preserves the test command, exit, and source hashes for the pre-fix reproduction.

R116 carries `max_new_tokens` into workload admission and checks each entry in the already-read and already-hashed output-count bytes before accepting the workload. It is a per-request limit: two requests with two output tokens each are accepted under a maximum of two, though the total is four. The R97 comparator remains pinned at its original SHA; source copies and hashes are listed in `MANIFEST.sha256`.

## Verification

`python3 -B run_tests.py` — exit 0, 11 CPU-only tests passed, including the violating case, the exact boundary, and an aggregate total above the per-request limit. The existing duplicate-key, boolean, hash-substitution, FIFO, workload-linkage, and package-comparability checks also pass. Python `compile()` parsed the four executable Python sources successfully. No model, engine, GPU, or benchmark was run.

The first test invocation exposed a setup issue: the copied test suite expects `.private-tmp` to exist. The R116 runner now creates this private mode-0700 directory before invoking tests; the later full run completed successfully. No product verdict was inferred from the setup failure.

## Scope

This is a private research comparator correction. The assumption that `max_new_tokens` means a per-request generated-token ceiling is stated in `CONTRACT.md`; the check does not authenticate producers or prove tokenizer correctness. No canonical VoltaFabric source or `main` branch changed.
