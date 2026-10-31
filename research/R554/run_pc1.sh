#!/bin/bash
set -euo pipefail
work=/home/pc1/NAOS/vf-r554-attempt2
pack=/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s
bin=/home/pc1/NAOS/vf-r553-work/target/release/vf-sc117-observed-prefill-batch
worker=/home/pc1/NAOS/vf-r552-work/target/release/vf-rank-session
cd "$work"
test ! -e SUBMITTED
test -z "$(find ledger -mindepth 1 -print -quit)"
test -z "$(ps -eo comm | grep -Fx vf-rank-session || :)"
printf '%s  %s\n' 7cca4714f0ff46dc283a98ccabfcd3cbcacdd185b796b0b103b59f2e2caf6a3b "$bin" ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32 "$worker" 3c20888b55920f5b108115ca4d24208de578755bc286c7c8ae8301959ecfea20 spec.json | sha256sum --check --status
test "$(df --output=avail -k "$work" | tail -1)" -ge 10485760
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 4194304
plan=$(sha256sum PLAN.json | cut -d' ' -f1)
PLAN="$plan" /usr/bin/python3 -c 'import os; f=os.open("SUBMITTED",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.write(f,(os.environ["PLAN"]+"\n").encode()); os.fsync(f); os.close(f); d=os.open(".",os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)'
systemd-run --user --scope --quiet -p MemoryMax=3G -p MemorySwapMax=0 -p TasksMax=64 -p CPUQuota=100% /usr/bin/env -u VF_SHARED_LEASE_SPEC -u VF_SHARED_LEASE_FD -u VF_R53_NAMESPACE_SPEC -u VF_EXPECTED_PLAN_ID OMP_NUM_THREADS=1 TMPDIR="$work/tmp" /usr/bin/ionice -c2 -n7 /usr/bin/nice -n 10 "$bin" spec.json "$pack/experts.bin" "$pack/native_experts.txt" "$work/ledger" "$worker" "$plan" > result.json 2> error.log
