#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r523-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker=/home/pc1/NAOS/vf-r495-work/target/release/vf-rank-session
cd "$work"
mkdir -p ledger tmp
[ -z "$(find ledger -mindepth 1 -print -quit)" ]
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
cp /home/pc1/NAOS/vf-r443-work/manifest.json manifest-a.json
cp manifest-a.json manifest-b.json
printf ' ' >> manifest-b.json
bin="$work/target/release/vf-sc117-prefill-greedy"
digest=$(sha256sum "$bin" | cut -d' ' -f1)
args=(--chat --max-new-tokens 1 --tokenizer-dir "$pack/tokenizer" --tokenizer-manifest "$work/adapters/strata_tokenizer/observed-sc117.json" --binary "$bin" --binary-sha256 "$digest" --dense "$pack/dense.bin" --shard "$model/$base-00001-of-00002.gguf" --pack "$pack/experts.bin" --index "$pack/native_experts.txt" --session "$worker" --ledger "$work/ledger" --ple-shard "$model/$base-00002-of-00002.gguf")
common=(/usr/bin/env -u VF_EXPECTED_PLAN_ID -u VF_RANK2_SSH_HOST -u VF_RANK2_SSH_WORKER -u VF_RANK2_SSH_WORKER_SHA256 -u VF_RANK3_SSH_HOST -u VF_RANK3_SSH_WORKER -u VF_RANK3_SSH_WORKER_SHA256 PYTHONDONTWRITEBYTECODE=1 VF_CPU_EXPERT_BUDGET=3221225472 VF_RANK2_BUDGET=1073741824 VF_RANK3_BUDGET=2147483648 OMP_NUM_THREADS=1 TMPDIR="$work/tmp")
scope=(systemd-run --user --scope --quiet -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=16 -p CPUQuota=100%)
for suffix in a b; do
 printf Hi | "${scope[@]}" "${common[@]}" /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner --prepare-plan "${args[@]}" --ple-manifest "$work/manifest-$suffix.json" > "plan-$suffix.json" 2> "plan-$suffix.err"
 test ! -s "plan-$suffix.err"
done
expected=$(/usr/bin/python3 -c 'import json; print(json.load(open("plan-a.json"))["plan_id"])')
if printf Hi | "${scope[@]}" "${common[@]}" VF_EXPECTED_PLAN_ID="$expected" /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner "${args[@]}" --ple-manifest "$work/manifest-b.json" > mismatch.out 2> mismatch.err; then exit 1; fi
test ! -s mismatch.out
grep -q 'engine failed' mismatch.err
[ -z "$(find ledger -mindepth 1 -print -quit)" ]
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
sha256sum "$bin" manifest-a.json manifest-b.json
printf '%s\n' 'PASS: changed manifest rejected before workers; fresh ledger empty'
