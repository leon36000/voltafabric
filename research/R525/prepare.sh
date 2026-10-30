#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r525-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker="$work/target/release/vf-rank-session"
bin="$work/target/release/vf-sc117-prefill-greedy"
manifest=/home/pc1/NAOS/vf-r443-work/manifest.json
cd "$work"
mkdir -p ledger tmp
[ -z "$(find ledger -mindepth 1 -print -quit)" ]
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
wd=$(sha256sum "$worker" | cut -d' ' -f1)
bd=$(sha256sum "$bin" | cut -d' ' -f1)
test "$wd" = 49216f6ce76a637be323dd0aa88c69e7788b4fa8ff13f5663ee103c5e4a22409
printf '%s  %s\n' 436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824 "$manifest" | sha256sum --check --status
args=(--chat --max-new-tokens 1 --tokenizer-dir "$pack/tokenizer" --tokenizer-manifest "$work/adapters/strata_tokenizer/observed-sc117.json" --binary "$bin" --binary-sha256 "$bd" --dense "$pack/dense.bin" --shard "$model/$base-00001-of-00002.gguf" --pack "$pack/experts.bin" --index "$pack/native_experts.txt" --session "$worker" --ledger "$work/ledger" --ple-shard "$model/$base-00002-of-00002.gguf" --ple-manifest "$manifest")
common=(/usr/bin/env -u VF_LOCAL_WORKER_SHA256 -u VF_EXPECTED_PLAN_ID -u VF_RANK2_SSH_HOST -u VF_RANK2_SSH_WORKER -u VF_RANK2_SSH_WORKER_SHA256 -u VF_RANK3_SSH_HOST -u VF_RANK3_SSH_WORKER -u VF_RANK3_SSH_WORKER_SHA256 PYTHONDONTWRITEBYTECODE=1 VF_CPU_EXPERT_BUDGET=3221225472 VF_RANK2_BUDGET=1073741824 VF_RANK3_BUDGET=2147483648 OMP_NUM_THREADS=1 TMPDIR="$work/tmp")
scope=(systemd-run --user --scope --quiet -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=16 -p CPUQuota=100%)
for n in a b; do
 if [[ $n == a ]]; then hash="$wd"; else hash=0000000000000000000000000000000000000000000000000000000000000000; fi
 printf Hi | "${scope[@]}" "${common[@]}" VF_LOCAL_WORKER_SHA256="$hash" /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner --prepare-plan "${args[@]}" > "plan-$n.json" 2> "plan-$n.err"
 test ! -s "plan-$n.err"
done
if printf Hi | "${scope[@]}" "${common[@]}" /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner --prepare-plan "${args[@]}" > missing.out 2> missing.err; then exit 1; fi
test ! -s missing.out
test -s missing.err
[ -z "$(find ledger -mindepth 1 -print -quit)" ]
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
sha256sum "$bin" "$worker"
printf '%s\n' 'PASS: two plans and missing digest refusal; fresh ledger empty'
