#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r553-work
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
worker=/home/pc1/NAOS/vf-r552-work/target/release/vf-rank-session
bin=$work/target/release/vf-sc117-real-grouped-batch
cd "$work"
test ! -e SUBMITTED
test -z "$(find ledger -mindepth 1 -print -quit)"
test -z "$(ps -eo comm | grep -Fx vf-rank-session || :)"
printf '%s  %s\n' eb888e40827d08e94b945ac12a6958e49a5d8b0c6d0b0a6a530d5a249beec5b3 "$bin" ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32 "$worker" | sha256sum --check --status
test "$(df --output=avail -k "$work" | tail -1)" -ge 10485760
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 4194304
plan=$(sha256sum PLAN.json | cut -d' ' -f1)
PLAN="$plan" /usr/bin/python3 -c 'import os; f=os.open("SUBMITTED",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.write(f,(os.environ["PLAN"]+"\n").encode()); os.fsync(f); os.close(f); d=os.open(".",os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)'
systemd-run --user --scope --quiet -p MemoryMax=3G -p MemorySwapMax=0 -p TasksMax=64 -p CPUQuota=100% /usr/bin/env -u VF_SHARED_LEASE_SPEC -u VF_SHARED_LEASE_FD -u VF_R53_NAMESPACE_SPEC -u VF_EXPECTED_PLAN_ID OMP_NUM_THREADS=1 TMPDIR="$work/tmp" /usr/bin/ionice -c2 -n7 /usr/bin/nice -n 10 "$bin" "$pack/experts.bin" "$pack/native_experts.txt" /home/pc1/NAOS/vf-r428-work/activation0.f32 /home/pc1/NAOS/vf-r428-work/activation1.f32 "$work/ledger" "$worker" "$plan" > result.json 2> error.log
