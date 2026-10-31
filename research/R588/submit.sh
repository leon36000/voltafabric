#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r588-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker=$work/target/release/vf-rank-session
bin="$work/target/release/vf-sc117-prefill-greedy"
manifest=/home/pc1/NAOS/vf-r443-work/manifest.json
cd "$work"
test ! -e SUBMITTED
test ! -e RESULT.json
test -z "$(find ledger -mindepth 1 -print -quit)"
test "$(sha256sum PLAN.json | cut -d' ' -f1)" = 65a53cc13eb6b6fb1e29f921781328a8d81e26e342269181f1d7886f30f5cee2
printf '%s  %s\n' eea9612b670f7766accf7dd2314e81fe74dc5ae7aae2e124e203371ddf6effb2 "$bin" aa562e626440742e55b2b09f8b4daa51db736c8f0a902082624885aa93d831ae "$worker" 436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824 "$manifest" 110b1ae3e7b6103dbf92d6b607f9975fb4fe060f94c0fe890799e35936056f60 PROMPT.json f87c08f36db743e4828c1b566c22174ce3da3507ba8e88e18b48440a0e8ac361 src/sc117_expert_source.rs 64d81c53ade8290d54febcc8418a947cd39a32f846fc326d41af5b7b77929f10 src/session.rs c18b90eee0d6569a323d90563d2bc61d4dbc5e223618e2012362760a2ea132db src/sc117_layer_chain.rs 005ce14d9b84f225bea534ced0dd4bbf5b27945b18f67e9239ef3ac9f43104a3 src/bin/vf-sc117-prefill-greedy.rs 99c8d6c56c6f6e4022897aeb28d0e74a8f08ed859bff6f64d811e070dd1f7d3f adapters/strata_tokenizer/prompt_runner.py | sha256sum --check --status
test "$(df --output=avail -k "$work" | tail -1)" -ge 10485760
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 16777216
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
expected=r576-1617cc85766f02ee3c40e4b9fadbd77b2a5aa9ea4eb8a21333247e628d79a048
/usr/bin/python3 -c 'import json; x=json.load(open("PREPARED.json")); assert x["plan_id"]=="r576-1617cc85766f02ee3c40e4b9fadbd77b2a5aa9ea4eb8a21333247e628d79a048" and x["prompt_tokens_sha256"]=="58906779ec0ce915a130efda3a987ad6aa45e6425109e0f67b72775e7a34f0f6"'
/usr/bin/python3 -c 'import os; f=os.open("SUBMITTED",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.write(f,b"65a53cc13eb6b6fb1e29f921781328a8d81e26e342269181f1d7886f30f5cee2\n"); os.fsync(f); os.close(f); d=os.open(".",os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)'
args=(--messages-json --max-new-tokens 2 --tokenizer-dir "$pack/tokenizer" --tokenizer-manifest "$work/adapters/strata_tokenizer/observed-sc117.json" --binary "$bin" --binary-sha256 eea9612b670f7766accf7dd2314e81fe74dc5ae7aae2e124e203371ddf6effb2 --dense "$pack/dense.bin" --shard "$model/$base-00001-of-00002.gguf" --pack "$pack/experts.bin" --index "$pack/native_experts.txt" --session "$worker" --ledger "$work/ledger" --ple-shard "$model/$base-00002-of-00002.gguf" --ple-manifest "$manifest")
set +e
systemd-run --user --scope --quiet -p MemoryHigh=11G -p MemoryMax=13G -p MemorySwapMax=0 -p TasksMax=320 -p CPUQuota=200% /usr/bin/env -i HOME=/home/pc1 PATH=/usr/bin:/bin TMPDIR="$work/tmp" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VF_LOCAL_WORKER_SHA256=aa562e626440742e55b2b09f8b4daa51db736c8f0a902082624885aa93d831ae VF_PREFILL_GROUP_SIZE=2 VF_EXPERT_RANK_AFFINITY=sticky VF_EXPECTED_PLAN_ID="$expected" /usr/bin/ionice -c2 -n7 /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner "${args[@]}" < PROMPT.json > RESULT.json 2> error.log
status=$?
printf '%s\n' "$status" > EXIT_STATUS
exit "$status"
