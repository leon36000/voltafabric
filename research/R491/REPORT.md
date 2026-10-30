# R491: bind a guarded remote worker digest and transport policy

Parent: GitLab R490 dbd6343926e571064b60bc30354626ae17ad2cb0. This is VoltaFabric code, not a donor import. R490's native GGML CPU provider remains the existing llama.cpp/Strata-derived integration; this change does not establish additional engine compatibility.

A remote rank now requires VF_RANK{2,3}_SSH_HOST, VF_RANK{2,3}_SSH_WORKER and VF_RANK{2,3}_SSH_WORKER_SHA256 together. The digest must be 64 lowercase hex characters. The SSH argv includes the explicit guarded mode and expected digest. Before HELLO, the worker first checks the established memory/swap thresholds and then hashes the running executable through an opened /proc/self/exe, bounded to 128 MiB; it rejects unreadable, changed, nonregular, empty or mismatched files. The plan v3 hashes the endpoints including digest and the raw-F32-if-advertised/JSON-otherwise policy. The actual negotiated transport remains observable separately in R490 counters.

This is an accidental-mismatch check by a cooperating binary, not remote attestation. A different or malicious program can ignore the arguments; the host, libraries, weights, lease and aggregated resources remain subject to external verification and admissions. No hardware or cross-host worker execution is authorized by this change. PC6 swap was below 1 GiB; the real local guarded binary refused before HELLO, exit 2, zero stdout bytes. Local mode remains unchanged.

Validation: two worker unit tests and three prefill unit tests passed; Clippy passed with -D warnings; independent Python JSON/SHA256 reproduced plan r491-0e6ef92e19e5b47c7fa82fbe3ce52f03f279ebee5b8480572f1d2c3f9bb04c73. The isolated CPU synthetic native-GGML test executed two actual local workers successfully. Its nine claims are preserved verbatim in CLAIMS.json, all RESERVED_UNKNOWN_NO_REPLAY. No GPU or inter-PC test was run. This does not qualify full generation with remote workers or measure performance.

Source SHA256: vf-rank-session.rs 65b4c62a10527a52d8e5d9c40b3b83b3a6ae13cd7747ca6b41302c4e07fb44c2; vf-sc117-prefill-greedy.rs d393e8b581aedb7e8cd5d511d535d89b35f2effee3f82766c35c79698b8534be.
