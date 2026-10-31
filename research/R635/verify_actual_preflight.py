#!/usr/bin/env python3
"""Prove plan mismatch is refused before R53 reservation or worker launch.

This runs only the pinned SC117 coordinator's --plan-only path. Its remote
endpoint and R53 namespace are synthetic; the wrapper refuses every command
except --plan-only, and IP socket creation is denied, so this cannot reach a
remote worker. The configured local worker is the harmless /usr/bin/false stub.
The input model files are used only for the coordinator's bounded plan
preflight and are never copied or modified.
"""
from __future__ import annotations

import argparse
import base64
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys

from adapters.strata_tokenizer.adapter import load_sc117
from adapters.strata_tokenizer import prompt_runner


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ssh_string(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--tokenizer-manifest", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--binary-sha256", required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--worker-sha256", required=True)
    p.add_argument("--dense", type=Path, required=True)
    p.add_argument("--shard", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--ple-shard", type=Path, required=True)
    p.add_argument("--ple-manifest", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            probe = socket.socket(family, socket.SOCK_STREAM)
        except OSError as error:
            if error.errno not in (errno.EPERM, errno.EACCES, errno.EAFNOSUPPORT):
                raise
        else:
            probe.close()
            raise RuntimeError("run this verifier with IP socket creation denied")
    for path in (a.tokenizer_dir, a.tokenizer_manifest, a.binary, a.worker,
                 a.dense, a.shard, a.pack, a.index, a.ple_shard, a.ple_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(a.binary) != a.binary_sha256 or digest(a.worker) != a.worker_sha256:
        raise ValueError("pinned executable digest mismatch")
    a.work.mkdir(mode=0o700, parents=True, exist_ok=False)

    ledger = a.work / "synthetic-r53"
    ledger.mkdir(mode=0o700)
    lease_path = a.work / "synthetic-lease.lock"
    lease_fd = os.open(lease_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease_stat = os.fstat(lease_fd)
    ledger_stat = ledger.stat()
    lease = {"path": str(lease_path), "dev": lease_stat.st_dev,
             "ino": lease_stat.st_ino, "uid": lease_stat.st_uid}
    namespace = {"root": str(ledger), "dev": ledger_stat.st_dev,
                 "ino": ledger_stat.st_ino, "uid": ledger_stat.st_uid,
                 "base_id": "r53-r635-" + a.work.name[-12:]}

    host = "preflight.invalid"
    key_blob = ssh_string(b"ssh-ed25519") + ssh_string(hashlib.sha256(b"VF-R635-synthetic-host-key").digest())
    known_hosts = a.work / "synthetic-known-hosts"
    known_hosts.write_text(f"{host} ssh-ed25519 {base64.b64encode(key_blob).decode('ascii')}\n",
                           encoding="ascii")
    os.chmod(known_hosts, 0o600)

    for name in tuple(os.environ):
        if name.startswith("VF_"):
            del os.environ[name]
    os.environ.update({
        "VF_LOCAL_WORKER_SHA256": a.worker_sha256,
        "VF_RANK3_SSH_HOST": host,
        "VF_RANK3_SSH_WORKER": str(a.worker),
        "VF_RANK3_SSH_WORKER_SHA256": a.worker_sha256,
        "VF_RANK3_SSH_KNOWN_HOSTS": str(known_hosts),
        "VF_RANK3_SSH_KNOWN_HOSTS_SHA256": digest(known_hosts),
        "VF_SHARED_LEASE_FD": str(lease_fd),
        "VF_SHARED_LEASE_SPEC": json.dumps(lease, separators=(",", ":")),
        "VF_R53_NAMESPACE_SPEC": json.dumps(namespace, separators=(",", ":")),
        "VF_EXPERT_RANK_AFFINITY": "sticky",
        "VF_PREFILL_GROUP_SIZE": "1",
        "VF_SC117_NATIVE_BACKEND": "vector",
    })

    manifest = json.loads(a.tokenizer_manifest.read_text(encoding="utf-8"))
    hashes = {key: manifest[key] for key in
              ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json")}
    tokenizer, eos = load_sc117(a.tokenizer_dir, hashes)
    engine_paths = [str(a.dense), str(a.shard), str(a.pack), str(a.index),
                    str(a.worker), str(a.work / "not-opened-ledger"),
                    str(a.ple_shard), str(a.ple_manifest)]
    calls: list[list[str]] = []

    def plan_only_runner(command, **kwargs):
        if len(command) < 2 or command[1] != "--plan-only":
            raise AssertionError("full generation/worker launch blocked by test wrapper")
        calls.append(list(command))
        return prompt_runner._run_bounded(command, **kwargs)

    with prompt_runner._pinned_executable(str(a.binary), a.binary_sha256) as (executable, fds):
        prepared = prompt_runner.run_prompt(
            "Hi", tokenizer, eos, executable, engine_paths, max_new_tokens=2,
            chat=True, require_bound_plan=True, prepare_plan=True,
            runner=plan_only_runner, binary_fd=fds[0])
        actual = prepared["plan_id"]
        prefix, body = actual.split("-", 1)
        wrong_body = ("0" if body[0] != "0" else "1") + body[1:]
        expected = prefix + "-" + wrong_body
        os.environ["VF_EXPECTED_PLAN_ID"] = expected
        try:
            prompt_runner.run_prompt(
                "Hi", tokenizer, eos, executable, engine_paths, max_new_tokens=2,
                chat=True, require_bound_plan=True, runner=plan_only_runner,
                binary_fd=fds[0], reserve_r53=True)
        except ValueError as error:
            if str(error) != "engine receipt differs from expected plan":
                raise
        else:
            raise AssertionError("mismatched plan was accepted")

    files = sorted(p.name for p in ledger.iterdir())
    if files:
        raise AssertionError(f"reservation occurred before plan match: {files}")
    if len(calls) != 2 or any(call[1] != "--plan-only" for call in calls):
        raise AssertionError("a non-plan-only engine invocation was attempted")

    receipt = {
        "status": "PASS_PLAN_MISMATCH_REFUSED_BEFORE_RESERVATION",
        "plan_only_invocations": len(calls),
        "full_generation_invocations": 0,
        "allowed_socket_family": "AF_UNIX",
        "ip_socket_creation": "DENIED",
        "worker_stub": str(a.worker),
        "worker_stub_sha256": a.worker_sha256,
        "gpu_used": False,
        "real_r53_used": False,
        "synthetic_ledger_files": files,
        "actual_plan_id": actual,
        "deliberately_wrong_expected_plan_id": expected,
        "binary_sha256": a.binary_sha256,
        "worker_sha256": a.worker_sha256,
        "tokenizer_manifest_sha256": digest(a.tokenizer_manifest),
        "known_hosts_sha256": digest(known_hosts),
    }
    (a.work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                          encoding="utf-8")
    os.chmod(a.work / "receipt.json", 0o600)
    print(json.dumps(receipt, separators=(",", ":")))
    os.close(lease_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
