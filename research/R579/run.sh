#!/bin/bash
set -euo pipefail
root=/home/pc6/NAOS/60_travaux/voltafabric-gitlab-r365
work="$root/.runs/vf-r579-work"
cd "$root"
test ! -e "$work/SUBMITTED"
test -z "$(find "$work/ledger" -mindepth 1 -maxdepth 1 -print -quit)"
test "$(sha256sum research/R579/PLAN.json | cut -d' ' -f1)" = ff430352e4240a779ecf3b2ac39945c12d9be34be21b15b5650d697d00afe2ab
printf '%s  %s\n' 55e86d3aaedc77b996e87edbc2809dc99328f4a81847fc473e9c39de79abc4a8 target/debug/deps/voltafabric-4035fa6cc8238506 05df66b3140414af842d24f1ad5a33874d5ba90fe8b5b8b342b46c1442781246 target/release/vf-rank-session 38aaa4dce307b8056d4f9547e24c4bedfda080075992c443ddcbec3e53f8f29a src/sc117_expert_source.rs | sha256sum --check --status
test "$(df --output=avail -k "$work" | tail -1)" -ge 10485760
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 4194304
PLAN=ff430352e4240a779ecf3b2ac39945c12d9be34be21b15b5650d697d00afe2ab WORK="$work" python3 -c 'import os; p=os.path.join(os.environ["WORK"],"SUBMITTED"); f=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.write(f,(os.environ["PLAN"]+"\n").encode()); os.fsync(f); os.close(f); d=os.open(os.environ["WORK"],os.O_RDONLY|os.O_DIRECTORY); os.fsync(d); os.close(d)'
set +e
systemd-run --user --scope --quiet -p MemoryMax=1G -p MemorySwapMax=0 -p TasksMax=32 -p CPUQuota=100% /usr/bin/env TMPDIR="$work/tmp" VF_R579_WORKER="$root/target/release/vf-rank-session" VF_R579_WORKER_SHA256=05df66b3140414af842d24f1ad5a33874d5ba90fe8b5b8b342b46c1442781246 VF_R579_PLAN=ff430352e4240a779ecf3b2ac39945c12d9be34be21b15b5650d697d00afe2ab VF_R579_RUN=r579-preflight-local-1793412766849027672 VF_R579_LEDGER="$work/ledger" /usr/bin/ionice -c2 -n7 /usr/bin/nice -n 10 "$root/target/debug/deps/voltafabric-4035fa6cc8238506" --ignored --exact sc117_expert_source::tests::sticky_preflight_survives_acknowledged_local_worker_eviction --nocapture > "$work/stdout" 2> "$work/stderr"
status=$?
printf '%s\n' "$status" > "$work/exit-status"
exit "$status"
