#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r527-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker=/home/pc1/NAOS/vf-r525-work/target/release/vf-rank-session
bin="$work/target/release/vf-sc117-prefill-greedy"
manifest=/home/pc1/NAOS/vf-r443-work/manifest.json
cd "$work"
mkdir -p ledger tmp
[ -z "$(find ledger -mindepth 1 -print -quit)" ]
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
python3 - <<'PY'
import base64,struct,pathlib
for name,byte in [('a',0),('b',1)]:
 key=struct.pack('>I',11)+b'ssh-ed25519'+struct.pack('>I',32)+bytes([byte])*32
 pathlib.Path('known-'+name+'.txt').write_text('pc1-lan ssh-ed25519 '+base64.b64encode(key).decode()+'\n')
PY
bd=$(sha256sum "$bin" | cut -d' ' -f1)
args=(--chat --max-new-tokens 1 --tokenizer-dir "$pack/tokenizer" --tokenizer-manifest "$work/adapters/strata_tokenizer/observed-sc117.json" --binary "$bin" --binary-sha256 "$bd" --dense "$pack/dense.bin" --shard "$model/$base-00001-of-00002.gguf" --pack "$pack/experts.bin" --index "$pack/native_experts.txt" --session "$worker" --ledger "$work/ledger" --ple-shard "$model/$base-00002-of-00002.gguf" --ple-manifest "$manifest")
common=(/usr/bin/env -u VF_EXPECTED_PLAN_ID -u VF_RANK2_SSH_HOST -u VF_RANK2_SSH_WORKER -u VF_RANK2_SSH_WORKER_SHA256 -u VF_RANK2_SSH_KNOWN_HOSTS -u VF_RANK2_SSH_KNOWN_HOSTS_SHA256 PYTHONDONTWRITEBYTECODE=1 VF_CPU_EXPERT_BUDGET=3221225472 VF_RANK2_BUDGET=1073741824 VF_RANK3_BUDGET=2147483648 VF_LOCAL_WORKER_SHA256=49216f6ce76a637be323dd0aa88c69e7788b4fa8ff13f5663ee103c5e4a22409 VF_RANK3_SSH_HOST=pc1-lan VF_RANK3_SSH_WORKER=/opt/vf-worker VF_RANK3_SSH_WORKER_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa VF_RANK3_SSH_KNOWN_HOSTS="$work/known-hosts" OMP_NUM_THREADS=1 TMPDIR="$work/tmp")
scope=(systemd-run --user --scope --quiet -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=16 -p CPUQuota=100%)
for n in a b; do
 cp "known-$n.txt" known-hosts
 hash=$(sha256sum known-hosts | cut -d' ' -f1)
 printf Hi | "${scope[@]}" "${common[@]}" VF_RANK3_SSH_KNOWN_HOSTS_SHA256="$hash" /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner --prepare-plan "${args[@]}" > "plan-$n.json" 2> "plan-$n.err"
 test ! -s "plan-$n.err"
done
if printf Hi | "${scope[@]}" "${common[@]}" VF_RANK3_SSH_KNOWN_HOSTS_SHA256=0000000000000000000000000000000000000000000000000000000000000000 /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner --prepare-plan "${args[@]}" > mismatch.out 2> mismatch.err; then exit 1; fi
test ! -s mismatch.out; test -s mismatch.err
[ -z "$(find ledger -mindepth 1 -print -quit)" ]
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
sha256sum "$bin" "$worker" known-a.txt known-b.txt
printf '%s\n' 'PASS: two pinned-key plans, wrong digest refused; no workers or claims'
