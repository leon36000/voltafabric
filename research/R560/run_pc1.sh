#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r560-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
shard=/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf
bin=/home/pc1/NAOS/vf-r556-work/target/release/vf-sc117-qsa-eviction-fault-check
worker=/home/pc1/NAOS/vf-r552-work/target/release/vf-rank-session
cd "$work"
test ! -e SUBMITTED
test ! -e suppressed.json
test -z "$(find ledger-lost ledger-idle -mindepth 1 -print -quit)"
test -z "$(ps -eo comm | grep -Fx vf-rank-session || :)"
printf '%s  %s\n' 89556f8dbe33176e77415802fba3e8701d7428852273b7f545436ff65e838dd1 "$bin" ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32 "$worker" 98b8236a7094a853208298a8247710f096b757885e79ff974aa3d4fdfbaa06cd "$work/drop_eviction_ack.py" 3c20888b55920f5b108115ca4d24208de578755bc286c7c8ae8301959ecfea20 "$work/spec.json" | sha256sum --check --status
test "$(df --output=avail -k "$work" | tail -1)" -ge 10485760
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 4194304
plan=$(sha256sum PLAN.json | cut -d' ' -f1)
PLAN="$plan" /usr/bin/python3 -c 'import os; f=os.open("SUBMITTED",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.write(f,(os.environ["PLAN"]+"\n").encode()); os.fsync(f); os.close(f); d=os.open(".",os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)'
systemd-run --user --scope --quiet -p MemoryMax=3G -p MemorySwapMax=0 -p TasksMax=64 -p CPUQuota=100% /usr/bin/env -u VF_SHARED_LEASE_SPEC -u VF_SHARED_LEASE_FD -u VF_R53_NAMESPACE_SPEC -u VF_EXPECTED_PLAN_ID OMP_NUM_THREADS=1 TMPDIR="$work/tmp" /usr/bin/ionice -c2 -n7 /usr/bin/nice -n 10 "$bin" /home/pc1/NAOS/vf-r552-work/trace-v2.f32 /home/pc1/NAOS/vf-r552-work/layer3.json "$pack/dense.bin" "$shard" "$pack/experts.bin" "$work/ledger-lost" "$work/ledger-idle" "$worker" "$plan" "$pack/native_experts.txt" "$work/spec.json" "$work/drop_eviction_ack.py" "$work/suppressed.json" > result.json 2> error.log
