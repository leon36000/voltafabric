#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r580-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
model=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S
base=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S
worker=/home/pc1/NAOS/vf-r552-work/target/release/vf-rank-session
bin="$work/target/release/vf-sc117-prefill-greedy"
manifest=/home/pc1/NAOS/vf-r443-work/manifest.json
cd "$work"
test ! -e SUBMITTED
test ! -e RESULT.json
test -z "$(find ledger -mindepth 1 -print -quit)"
test "$(sha256sum PLAN.json | cut -d' ' -f1)" = 126c407abeb1f277cf42bed6385647b8fbd3e081e10449f07bdfa76ee2d05a6f
printf '%s  %s\n' 8afd8697e42eea244311db2beddcd87a12c509f7d0e0d8c505586876db521db5 "$bin" ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32 "$worker" 436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824 "$manifest" 110b1ae3e7b6103dbf92d6b607f9975fb4fe060f94c0fe890799e35936056f60 PROMPT.json 38aaa4dce307b8056d4f9547e24c4bedfda080075992c443ddcbec3e53f8f29a src/sc117_expert_source.rs c18b90eee0d6569a323d90563d2bc61d4dbc5e223618e2012362760a2ea132db src/sc117_layer_chain.rs 248c075d86203f993ef3a89455b020d332a99306cb18ddf1e086c64e67adb622 src/bin/vf-sc117-prefill-greedy.rs 99c8d6c56c6f6e4022897aeb28d0e74a8f08ed859bff6f64d811e070dd1f7d3f adapters/strata_tokenizer/prompt_runner.py | sha256sum --check --status
test "$(df --output=avail -k "$work" | tail -1)" -ge 10485760
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 16777216
if ps -eo comm | grep -Fx vf-rank-session >/dev/null; then exit 1; fi
expected=r576-af93e2331126d1cd09aeaabbf39191ebe7929de0283335d55fa57162ce35d7e0
/usr/bin/python3 -c 'import json; x=json.load(open("PREPARED.json")); assert x["plan_id"]=="r576-af93e2331126d1cd09aeaabbf39191ebe7929de0283335d55fa57162ce35d7e0" and x["prompt_tokens_sha256"]=="58906779ec0ce915a130efda3a987ad6aa45e6425109e0f67b72775e7a34f0f6"'
/usr/bin/python3 -c 'import os; f=os.open("SUBMITTED",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.write(f,b"126c407abeb1f277cf42bed6385647b8fbd3e081e10449f07bdfa76ee2d05a6f\n"); os.fsync(f); os.close(f); d=os.open(".",os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)'
args=(--messages-json --max-new-tokens 2 --tokenizer-dir "$pack/tokenizer" --tokenizer-manifest "$work/adapters/strata_tokenizer/observed-sc117.json" --binary "$bin" --binary-sha256 8afd8697e42eea244311db2beddcd87a12c509f7d0e0d8c505586876db521db5 --dense "$pack/dense.bin" --shard "$model/$base-00001-of-00002.gguf" --pack "$pack/experts.bin" --index "$pack/native_experts.txt" --session "$worker" --ledger "$work/ledger" --ple-shard "$model/$base-00002-of-00002.gguf" --ple-manifest "$manifest")
set +e
systemd-run --user --scope --quiet -p MemoryHigh=11G -p MemoryMax=13G -p MemorySwapMax=0 -p TasksMax=320 -p CPUQuota=200% /usr/bin/env -i HOME=/home/pc1 PATH=/usr/bin:/bin TMPDIR="$work/tmp" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VF_LOCAL_WORKER_SHA256=ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32 VF_PREFILL_GROUP_SIZE=2 VF_EXPERT_RANK_AFFINITY=sticky VF_EXPECTED_PLAN_ID="$expected" /usr/bin/ionice -c2 -n7 /usr/bin/nice -n 10 /home/pc1/Strata/.venv/bin/python -m adapters.strata_tokenizer.prompt_runner "${args[@]}" < PROMPT.json > RESULT.json 2> error.log
status=$?
printf '%s\n' "$status" > EXIT_STATUS
exit "$status"
