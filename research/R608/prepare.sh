#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r608-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker=$work/target/release/vf-rank-session
bin=$work/target/release/vf-sc117-prefill-greedy
manifest=/home/pc1/NAOS/vf-r443-work/manifest.json
cd "$work"
test ! -e PREPARED.json
test ! -e SUBMITTED
test -z "$(find ledger -mindepth 1 -print -quit)"
printf '%s  %s\n' 436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824 "$manifest" 110b1ae3e7b6103dbf92d6b607f9975fb4fe060f94c0fe890799e35936056f60 PROMPT.json | sha256sum --check --status
bsha=$(sha256sum "$bin" | cut -d' ' -f1)
wsha=$(sha256sum "$worker" | cut -d' ' -f1)
args=(--prepare-plan --messages-json --max-new-tokens 2 --tokenizer-dir "$pack/tokenizer" --tokenizer-manifest "$work/adapters/strata_tokenizer/observed-sc117.json" --binary "$bin" --binary-sha256 "$bsha" --dense "$pack/dense.bin" --shard "$model/$base-00001-of-00002.gguf" --pack "$pack/experts.bin" --index "$pack/native_experts.txt" --session "$worker" --ledger "$work/ledger" --ple-shard "$model/$base-00002-of-00002.gguf" --ple-manifest "$manifest")
systemd-run --user --scope --quiet -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=16 -p CPUQuota=100% /usr/bin/env -i HOME=/home/pc1 PATH=/usr/bin:/bin TMPDIR="$work/tmp" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VF_SC117_NATIVE_BACKEND=scalar VF_LOCAL_WORKER_SHA256="$wsha" VF_PREFILL_GROUP_SIZE=2 VF_EXPERT_RANK_AFFINITY=sticky /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner "${args[@]}" < PROMPT.json > PREPARED.json 2> prepare.err
test ! -s prepare.err
test -z "$(find ledger -mindepth 1 -print -quit)"
