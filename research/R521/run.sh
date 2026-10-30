#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r521-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker=/home/pc1/NAOS/vf-r495-work/target/release/vf-rank-session
cd "$work"
args=(--chat --max-new-tokens 1 --tokenizer-dir "$pack/tokenizer" --tokenizer-manifest "$work/adapters/strata_tokenizer/observed-sc117.json" --binary "$work/target/release/vf-sc117-prefill-greedy" --binary-sha256 81b60fbfd0044ecc8d985b7d5f67b0d6abd6f5bbb45bb3e89a0581fabc209f36 --dense "$pack/dense.bin" --shard "$model/$base-00001-of-00002.gguf" --pack "$pack/experts.bin" --index "$pack/native_experts.txt" --session "$worker" --ledger "$work/ledger" --ple-shard "$model/$base-00002-of-00002.gguf" --ple-manifest /home/pc1/NAOS/vf-r443-work/manifest.json)
common=(/usr/bin/env -u VF_EXPECTED_PLAN_ID -u VF_RANK2_SSH_HOST -u VF_RANK2_SSH_WORKER -u VF_RANK2_SSH_WORKER_SHA256 -u VF_RANK3_SSH_HOST -u VF_RANK3_SSH_WORKER -u VF_RANK3_SSH_WORKER_SHA256 PYTHONDONTWRITEBYTECODE=1 VF_CPU_EXPERT_BUDGET=3221225472 VF_RANK2_BUDGET=1073741824 VF_RANK3_BUDGET=2147483648 OMP_NUM_THREADS=1 TMPDIR="$work/tmp")
test -z "$(find "$work/ledger" -maxdepth 1 -type f -print -quit)"
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
if [[ ${1:-} == prepare ]]; then
 test ! -e "$work/plan.json"
 systemd-run --user --scope --quiet -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=16 -p CPUQuota=100% "${common[@]}" /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner --prepare-plan "${args[@]}" < prompt.txt > plan.json 2> prepare.err
 test ! -s prepare.err
 test -z "$(find "$work/ledger" -maxdepth 1 -type f -print -quit)"
elif [[ ${1:-} == submit ]]; then
 printf '%s  %s\n' 92ba481ce77d00a16f1a8b4c66e892423501fc51463bdc67ca563f6e8891eddb plan.json 3639efcd08abb273b1619e82e78c29a7df02c1051b1820e99fc395dcaa3326b8 prompt.txt | sha256sum --check --status
 expected=$(/usr/bin/python3 -c 'import json; print(json.load(open("plan.json"))["plan_id"])')
 test "$expected" = r520-4dbd46cb334b490948419595b28c8203474b08c8b38c7c2da15bb219dc749945
 EXPECTED_PLAN="$expected" /usr/bin/python3 -c 'import os; fd=os.open("SUBMITTED",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.write(fd,(os.environ["EXPECTED_PLAN"]+"\n").encode()); os.fsync(fd); os.close(fd); d=os.open(".",os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)'
 systemd-run --user --scope --quiet -p MemoryHigh=11G -p MemoryMax=13G -p MemorySwapMax=0 -p TasksMax=320 -p CPUQuota=200% "${common[@]}" VF_EXPECTED_PLAN_ID="$expected" /usr/bin/ionice -c2 -n7 /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner "${args[@]}" < prompt.txt > result.json 2> error.log
else
 exit 2
fi
