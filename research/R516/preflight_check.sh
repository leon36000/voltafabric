#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r516-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker=/home/pc1/NAOS/vf-r495-work/target/release/vf-rank-session
bin=$work/target/release/vf-sc117-prefill-greedy
args=("$pack/dense.bin" "$model/$base-00001-of-00002.gguf" "$pack/experts.bin" "$pack/native_experts.txt" "$worker" "$work/ledger" "$model/$base-00002-of-00002.gguf" /home/pc1/NAOS/vf-r443-work/manifest.json 48 248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271 1 248046 chat)
common=(/usr/bin/env -u VF_EXPECTED_PLAN_ID -u VF_RANK2_SSH_HOST -u VF_RANK2_SSH_WORKER -u VF_RANK2_SSH_WORKER_SHA256 VF_CPU_EXPERT_BUDGET=3221225472 VF_RANK2_BUDGET=1073741824 VF_RANK3_BUDGET=2147483648 VF_RANK3_SSH_HOST=pc1-lan VF_RANK3_SSH_WORKER="$worker" VF_RANK3_SSH_WORKER_SHA256=9dd3b08169e3ddce9228241b3c72a18f885cd8433db4d2806fa10aebc7b386c9)
test -z "$(find "$work/ledger" -maxdepth 1 -type f -print -quit)"
if ps -eo comm | grep -Eq '^(vf-rank-session|vf-sc117-prefi)$'; then exit 1; fi
"${common[@]}" "$bin" --plan-only "${args[@]}" > "$work/plan.json"
if "${common[@]}" "$bin" "${args[@]}" > "$work/missing.out" 2> "$work/missing.err"; then exit 1; fi
if "${common[@]}" VF_EXPECTED_PLAN_ID=r512-0000000000000000000000000000000000000000000000000000000000000000 "$bin" "${args[@]}" > "$work/mismatch.out" 2> "$work/mismatch.err"; then exit 1; fi
grep -Fx 'prefill-greedy qualification refused: expected worker plan missing or mismatched' "$work/missing.err" >/dev/null
grep -Fx 'prefill-greedy qualification refused: expected worker plan missing or mismatched' "$work/mismatch.err" >/dev/null
test ! -s "$work/missing.out" && test ! -s "$work/mismatch.out"
test -z "$(find "$work/ledger" -maxdepth 1 -type f -print -quit)"
if ps -eo comm | grep -Eq '^(vf-rank-session|vf-sc117-prefi)$'; then exit 1; fi
printf 'PASS: plan-only and both refusals completed without workers or claims\n'
