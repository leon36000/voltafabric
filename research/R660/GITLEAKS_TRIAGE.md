# R660 — classification Gitleaks R659

Pipeline 2250's Gitleaks job 4760 reported 53 unique `generic-api-key` matches in the inherited history. The report redacts all matched values. Cross-checking the exact field names and source records classified them as digest data, not authentication material:

- 42 `prompt_tokens_sha256` entries;
- 7 SHA-256 file digests recorded by `strata_tokenizer.py`;
- one each for `tokenizer_manifest_sha256`, `hostkeys_sha256`, `token_type.json`, and `tokenizer.json`.

The allowlist in the repository root contains only those 53 exact Gitleaks fingerprints from job 4760. It does not disable or broaden the `generic-api-key` detector. No matched value is copied into this note. This is a narrow CI false-positive classification for the inherited synthetic/digest records; new findings remain subject to scanning.
