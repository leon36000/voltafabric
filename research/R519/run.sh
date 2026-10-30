#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r519-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker=/home/pc1/NAOS/vf-r495-work/target/release/vf-rank-session
cd "$work"
args=(--messages-json --max-new-tokens 1 --tokenizer-dir "$pack/tokenizer" --tokenizer-manifest "$work/adapters/strata_tokenizer/observed-sc117.json" --binary "$work/target/release/vf-sc117-prefill-greedy" --binary-sha256 afaeb85d16634b7a6a1ab623cb2f14df2257604cfea0abe578364f4599b368e2 --dense "$pack/dense.bin" --shard "$model/$base-00001-of-00002.gguf" --pack "$pack/experts.bin" --index "$pack/native_experts.txt" --session "$worker" --ledger "$work/ledger" --ple-shard "$model/$base-00002-of-00002.gguf" --ple-manifest /home/pc1/NAOS/vf-r443-work/manifest.json)
common=(/usr/bin/env -u VF_EXPECTED_PLAN_ID -u VF_RANK2_SSH_HOST -u VF_RANK2_SSH_WORKER -u VF_RANK2_SSH_WORKER_SHA256 -u VF_RANK3_SSH_HOST -u VF_RANK3_SSH_WORKER -u VF_RANK3_SSH_WORKER_SHA256 PYTHONDONTWRITEBYTECODE=1 VF_CPU_EXPERT_BUDGET=3221225472 VF_RANK2_BUDGET=1073741824 VF_RANK3_BUDGET=2147483648 OMP_NUM_THREADS=1 TMPDIR="$work/tmp")
test -z "$(find "$work/ledger" -maxdepth 1 -type f -print -quit)"
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
if [[ ${1:-} == prepare ]]; then
 test ! -e "$work/plan.json"
 systemd-run --user --scope --quiet -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=16 -p CPUQuota=100% "${common[@]}" /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner --prepare-plan "${args[@]}" < prompt.json > plan.json 2> prepare.err
 test ! -s prepare.err
 test -z "$(find "$work/ledger" -maxdepth 1 -type f -print -quit)"
elif [[ ${1:-} == submit ]]; then
 printf '%s  %s\n' de0d3fc80a1ab6b2d2c258510bc9c9e61c6806596fe1f9913839ef34a3598b0a plan.json d28485b0c0ae8343dae6d965a0bf6a9d365b9a6db0c78f8995ddc7d993ee60e7 prompt.json | sha256sum --check --status
 expected=$(/usr/bin/python3 -c 'import json; print(json.load(open("plan.json"))["plan_id"])')
 test "$expected" = r518-5a14526a95cf27069e030dabe0e17b4d73b70feec670ea9b23afd2e31157ed67
 EXPECTED_PLAN="$expected" /usr/bin/python3 -c 'import os; fd=os.open("SUBMITTED",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.write(fd,(os.environ["EXPECTED_PLAN"]+"\n").encode()); os.fsync(fd); os.close(fd); d=os.open(".",os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)'
 systemd-run --user --scope --quiet -p MemoryHigh=11G -p MemoryMax=13G -p MemorySwapMax=0 -p TasksMax=320 -p CPUQuota=200% "${common[@]}" VF_EXPECTED_PLAN_ID="$expected" /usr/bin/ionice -c2 -n7 /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner "${args[@]}" < prompt.json > result.json 2> error.log
else
 exit 2
fi
