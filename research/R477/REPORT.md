# R477 — bounded classification of engine refusals

Parent: R476, commit 170d8dbff8cc7db131c92b1b3dfcdc467106712c. The SC117 tokenizer prompt runner previously discarded all stderr when the engine failed. It now maps four exact, complete known engine stderr lines to fixed category names: rank payload capacity, global payload capacity, eviction target, or eviction limit. All other diagnostics remain `unclassified`. It never includes arbitrary engine stdout or stderr in the exception, performs no retry, and continues to demand reconciliation of UNKNOWN ledger entries. Matching is deliberately exact; nested or changed error messages remain unclassified. No claim is made that R476 failed for any of the named reasons: its stderr is unavailable.

The existing Strata-derived tokenizer adapter and llama.cpp-derived tokenizer code retain their earlier provenance and licenses; this change is new VoltaFabric glue, not code extracted from another engine.

Validation: focused regression initially failed against the old runner on PC1; the default PC6 Python lacks `regex`, which was an environment failure, not a passing test. With the existing PC1 Strata virtualenv, `PYTHONDONTWRITEBYTECODE=1 python -m unittest adapters.strata_tokenizer.test_prompt_runner adapters.strata_tokenizer.test_adapter` ran 13 tests, 12 passed and 1 observed-pack test was skipped. Tests cover all four exact categories, refusal of a near-match or arbitrary stderr, no stdout leakage, no retry, and a real short-lived CPU subprocess. No generation, workers, GPU or network experiment was run, and R476 claims were not replayed.

SHA256: prompt_runner.py 49faf2397ff6c49afa783db50a58caddc9a28fb41f8817629f3516c8cf4f3ee3; test_prompt_runner.py d950fdb5e7f38020135cd38f956ab2c1bd164eb3a1dda0302402362b13340b98.

Next: design and independently test rank-aware headroom before trying a fresh admitted generation. Global eviction does not guarantee headroom on an individual rank; this is a demonstrated design limitation, not an established explanation of the R476 failure. Physical host memory and cross-machine operation remain unqualified.
