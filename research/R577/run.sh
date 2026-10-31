#!/bin/bash
set -euo pipefail
root=/home/pc6/NAOS/60_travaux/voltafabric-gitlab-r365
work="$root/.runs/vf-r577-work"
cd "$root"
test ! -e "$work/SUBMITTED"
test -z "$(find "$work/ledger" -mindepth 1 -maxdepth 1 -print -quit)"
test "$(sha256sum research/R577/PLAN.json | cut -d' ' -f1)" = 1531399d002dd4f900b678a41fb7e0ee0c9349597e9a18a2aa40907b9493072e
printf '%s  %s\n' a5ae117538c3db19a8933360b459c429717126c010054c8b3d025a0db823da04 target/debug/deps/voltafabric-4035fa6cc8238506 05df66b3140414af842d24f1ad5a33874d5ba90fe8b5b8b342b46c1442781246 target/release/vf-rank-session f9cb3047930cddd411078de868aced811a363fe6df7357c0e66dabb05ba847e9 src/sc117_expert_source.rs | sha256sum --check --status
test "$(df --output=avail -k "$work" | tail -1)" -ge 10485760
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 4194304
PLAN=1531399d002dd4f900b678a41fb7e0ee0c9349597e9a18a2aa40907b9493072e WORK="$work" python3 -c 'import os; p=os.path.join(os.environ["WORK"],"SUBMITTED"); f=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.write(f,(os.environ["PLAN"]+"\n").encode()); os.fsync(f); os.close(f); d=os.open(os.environ["WORK"],os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)'
set +e
systemd-run --user --scope --quiet -p MemoryMax=1G -p MemorySwapMax=0 -p TasksMax=32 -p CPUQuota=100% /usr/bin/env TMPDIR="$work/tmp" VF_R577_WORKER="$root/target/release/vf-rank-session" VF_R577_WORKER_SHA256=05df66b3140414af842d24f1ad5a33874d5ba90fe8b5b8b342b46c1442781246 VF_R577_PLAN=1531399d002dd4f900b678a41fb7e0ee0c9349597e9a18a2aa40907b9493072e VF_R577_RUN=r577-sticky-local-1793412208275715695 VF_R577_LEDGER="$work/ledger" /usr/bin/ionice -c2 -n7 /usr/bin/nice -n 10 "$root/target/debug/deps/voltafabric-4035fa6cc8238506" --ignored --exact sc117_expert_source::tests::sticky_affinity_survives_acknowledged_local_worker_eviction --nocapture > "$work/stdout" 2> "$work/stderr"
status=$?
printf '%s\n' "$status" > "$work/exit-status"
exit "$status"
