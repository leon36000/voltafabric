#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r524-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker=/home/pc1/NAOS/vf-r495-work/target/release/vf-rank-session
bin=/home/pc1/NAOS/vf-r523-work/target/release/vf-sc117-prefill-greedy
manifest=/home/pc1/NAOS/vf-r443-work/manifest.json
cd "$work"
printf '%s  %s\n' 6d7f398523c521fb9fd40757ccab38e7d635d59a4f36a9934fff64bb69ac79ef "$bin" 9dd3b08169e3ddce9228241b3c72a18f885cd8433db4d2806fa10aebc7b386c9 "$worker" 436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824 "$manifest" 3639efcd08abb273b1619e82e78c29a7df02c1051b1820e99fc395dcaa3326b8 prompt.txt | sha256sum --check --status
[ -z "$(find ledger -mindepth 1 -print -quit)" ]
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
args=(--chat --max-new-tokens 16 --tokenizer-dir "$pack/tokenizer" --tokenizer-manifest "$work/adapters/strata_tokenizer/observed-sc117.json" --binary "$bin" --binary-sha256 6d7f398523c521fb9fd40757ccab38e7d635d59a4f36a9934fff64bb69ac79ef --dense "$pack/dense.bin" --shard "$model/$base-00001-of-00002.gguf" --pack "$pack/experts.bin" --index "$pack/native_experts.txt" --session "$worker" --ledger "$work/ledger" --ple-shard "$model/$base-00002-of-00002.gguf" --ple-manifest "$manifest")
common=(/usr/bin/env -u VF_EXPECTED_PLAN_ID -u VF_RANK2_SSH_HOST -u VF_RANK2_SSH_WORKER -u VF_RANK2_SSH_WORKER_SHA256 -u VF_RANK3_SSH_HOST -u VF_RANK3_SSH_WORKER -u VF_RANK3_SSH_WORKER_SHA256 PYTHONDONTWRITEBYTECODE=1 VF_CPU_EXPERT_BUDGET=3221225472 VF_RANK2_BUDGET=1073741824 VF_RANK3_BUDGET=2147483648 OMP_NUM_THREADS=1 TMPDIR="$work/tmp")
if [[ ${1:-} == prepare ]]; then
 test ! -e plan.json
 systemd-run --user --scope --quiet -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=16 -p CPUQuota=100% "${common[@]}" /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner --prepare-plan "${args[@]}" < prompt.txt > plan.json 2> prepare.err
 test ! -s prepare.err
 [ -z "$(find ledger -mindepth 1 -print -quit)" ]
elif [[ ${1:-} == submit ]]; then
 test "${2:-}" = "$(sha256sum plan.json | cut -d' ' -f1)"
 expected=$(/usr/bin/python3 -c 'import json; print(json.load(open("plan.json"))["plan_id"])')
 test "${3:-}" = "$expected"
 EXPECTED_PLAN="$expected" /usr/bin/python3 -c 'import os; fd=os.open("SUBMITTED",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.write(fd,(os.environ["EXPECTED_PLAN"]+"\n").encode()); os.fsync(fd); os.close(fd); d=os.open(".",os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)'
 systemd-run --user --scope --quiet -p MemoryHigh=11G -p MemoryMax=13G -p MemorySwapMax=0 -p TasksMax=320 -p CPUQuota=200% "${common[@]}" VF_EXPECTED_PLAN_ID="$expected" /usr/bin/ionice -c2 -n7 /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner "${args[@]}" < prompt.txt > result.json 2> error.log
else exit 2
fi
