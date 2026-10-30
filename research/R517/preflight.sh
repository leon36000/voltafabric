#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r517-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker=/home/pc1/NAOS/vf-r495-work/target/release/vf-rank-session
test -z "$(find "$work/ledger" -maxdepth 1 -type f -print -quit)"
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
cd "$work"
printf Hi | systemd-run --user --scope --quiet -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=16 -p CPUQuota=100% /usr/bin/env -u VF_EXPECTED_PLAN_ID -u VF_RANK2_SSH_HOST -u VF_RANK2_SSH_WORKER -u VF_RANK2_SSH_WORKER_SHA256 PYTHONDONTWRITEBYTECODE=1 VF_CPU_EXPERT_BUDGET=3221225472 VF_RANK2_BUDGET=1073741824 VF_RANK3_BUDGET=2147483648 VF_RANK3_SSH_HOST=pc1-lan VF_RANK3_SSH_WORKER="$worker" VF_RANK3_SSH_WORKER_SHA256=9dd3b08169e3ddce9228241b3c72a18f885cd8433db4d2806fa10aebc7b386c9 /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner --prepare-plan --chat --max-new-tokens 1 --tokenizer-dir "$pack/tokenizer" --tokenizer-manifest "$work/adapters/strata_tokenizer/observed-sc117.json" --binary /home/pc1/NAOS/vf-r516-work/target/release/vf-sc117-prefill-greedy --binary-sha256 72e52652742b50ff3d1ffd20ed50480c3e48aed7396900bd4dc931ac12bd78f3 --dense "$pack/dense.bin" --shard "$model/$base-00001-of-00002.gguf" --pack "$pack/experts.bin" --index "$pack/native_experts.txt" --session "$worker" --ledger "$work/ledger" --ple-shard "$model/$base-00002-of-00002.gguf" --ple-manifest /home/pc1/NAOS/vf-r443-work/manifest.json > "$work/plan.json" 2> "$work/error.log"
test ! -s "$work/error.log"
test -z "$(find "$work/ledger" -maxdepth 1 -type f -print -quit)"
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
printf 'PASS: tokenizer and plan-only engine completed without workers or claims\n'
