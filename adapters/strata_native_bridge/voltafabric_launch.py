#!/usr/bin/env python3
"""Launch pinned Strata with 48 identity-bound local VoltaFabric CPU sessions.

The launcher is opt-in and local-only. It does not create hardware leases, use
RDMA, or retry an uncertain VFB3 request. The checkpoint manifest binds the
per-expert digests consumed by the native coordinator; it is not a digest of
all dense checkpoint bytes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import socket
import stat
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAYERS = 48
HIDDEN = 2560
SC117_PACK_BYTES = 50_292_326_400
MAX_U64 = (1 << 64) - 1
MAX_INIT_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_INDEX_BYTES = 65536
MAX_NATIVE_PAYLOAD_BYTES = 1_250_000_000
MAX_AGGREGATE_PAYLOAD_BYTES = LAYERS * MAX_NATIVE_PAYLOAD_BYTES
MAX_STARTUP_TIMEOUT_MS = 1_800_000
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")


class LaunchError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LaunchError("duplicate JSON key")
        result[key] = value
    return result


def _json_load(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(LaunchError("non-finite JSON number")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f"invalid {label} JSON") from exc


def _read_regular(path_value: Any, limit: int, label: str) -> tuple[Path, bytes, os.stat_result]:
    if not isinstance(path_value, str) or not path_value:
        raise LaunchError(f"{label} path is required")
    path = Path(path_value)
    if not path.is_absolute():
        raise LaunchError(f"{label} path must be absolute")
    try:
        before = path.lstat()
    except OSError as exc:
        raise LaunchError(f"cannot stat {label}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise LaunchError(f"{label} must be a regular non-symlink file")
    if before.st_size > limit:
        raise LaunchError(f"{label} exceeds the configured metadata bound")
    try:
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise LaunchError(f"cannot read {label}") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != before.st_size:
        raise LaunchError(f"{label} changed while being read")
    return path, raw, after


def _regular_file_stat(path_value: Any, label: str) -> tuple[Path, os.stat_result]:
    """Inspect a potentially huge model file without reading it into memory."""
    if not isinstance(path_value, str) or not path_value or not Path(path_value).is_absolute():
        raise LaunchError(f"{label} path must be absolute")
    path = Path(path_value)
    try:
        info = path.lstat()
    except OSError as exc:
        raise LaunchError(f"cannot stat {label}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise LaunchError(f"{label} must be a regular non-symlink file")
    return path, info


def _executable(value: Any, label: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise LaunchError(f"{label} must be an absolute executable path")
    path = Path(value)
    try:
        info = path.lstat()
    except OSError as exc:
        raise LaunchError(f"cannot stat {label}") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(path, os.X_OK):
        raise LaunchError(f"{label} must be a regular executable file")
    return str(path)


def _u64(value: Any, label: str, *, nonzero: bool = False, leave_room: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_U64:
        raise LaunchError(f"{label} must be an unsigned 64-bit integer")
    if nonzero and value == 0:
        raise LaunchError(f"{label} must be nonzero")
    if leave_room and value == MAX_U64:
        raise LaunchError(f"{label} is exhausted")
    return value


def _identity_field(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise LaunchError(f"{label} contains unsupported characters or length")
    return value


def _check_manifest(manifest_path: str) -> dict[str, Any]:
    _, raw, _ = _read_regular(manifest_path, MAX_MANIFEST_BYTES, "launch manifest")
    manifest = _json_load(raw, "launch manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise LaunchError("unsupported launch manifest schema")

    run_id = _identity_field(manifest.get("run_id"), "run_id")
    epoch = _u64(manifest.get("epoch"), "epoch")
    first_batch = _u64(manifest.get("first_batch"), "first_batch", leave_room=True)
    first_attempt = _u64(manifest.get("first_attempt"), "first_attempt", nonzero=True, leave_room=True)
    startup_timeout = manifest.get("startup_timeout_ms")
    shutdown_timeout = manifest.get("shutdown_timeout_ms")
    if isinstance(startup_timeout, bool) or not isinstance(startup_timeout, int) or not 1 <= startup_timeout <= MAX_STARTUP_TIMEOUT_MS:
        raise LaunchError("startup_timeout_ms must be in 1..1800000")
    if isinstance(shutdown_timeout, bool) or not isinstance(shutdown_timeout, int) or not 1 <= shutdown_timeout <= 60_000:
        raise LaunchError("shutdown_timeout_ms must be in 1..60000")

    _, checkpoint_raw, _ = _read_regular(manifest.get("checkpoint_manifest_path"), MAX_MANIFEST_BYTES,
                                         "checkpoint manifest")
    checkpoint = _json_load(checkpoint_raw, "checkpoint manifest")
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise LaunchError("unsupported checkpoint manifest schema")
    _, plan_raw, _ = _read_regular(manifest.get("plan_path"), MAX_MANIFEST_BYTES, "plan")
    plan = _json_load(plan_raw, "plan")
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise LaunchError("unsupported plan schema")

    model_revision = "sha256:" + hashlib.sha256(checkpoint_raw).hexdigest()
    plan_id = "sha256:" + hashlib.sha256(plan_raw).hexdigest()
    pack_path_value = checkpoint.get("pack_path")
    pack_path, pack_stat = _regular_file_stat(pack_path_value, "SC117 expert pack")
    if pack_stat.st_size != SC117_PACK_BYTES:
        raise LaunchError("SC117 expert pack size does not match the pinned v3 layout")
    index = checkpoint.get("index")
    if not isinstance(index, str) or not index or len(index.encode("utf-8")) > MAX_INDEX_BYTES:
        raise LaunchError("checkpoint manifest index is missing or oversized")
    index_sha256 = hashlib.sha256(index.encode("utf-8")).hexdigest()
    if not index.startswith("# strata native experts v3:"):
        raise LaunchError("checkpoint manifest does not contain the pinned SC117 v3 index")

    model_experts = checkpoint.get("model_experts")
    if (not isinstance(model_experts, list) or len(model_experts) != 512 or
            any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 512 for v in model_experts) or
            len(set(model_experts)) != 512):
        raise LaunchError("checkpoint manifest must enumerate the 512 SC117 experts exactly once")
    bindings = checkpoint.get("bindings")
    if not isinstance(bindings, list) or not bindings or len(bindings) > 48 * 512 * 64:
        raise LaunchError("checkpoint binding list is empty or oversized")
    by_layer: list[list[dict[str, Any]]] = [[] for _ in range(LAYERS)]
    binding_keys: set[tuple[int, int, int]] = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            raise LaunchError("invalid checkpoint binding")
        layer = binding.get("layer")
        expert = binding.get("expert")
        rank = binding.get("rank")
        local = binding.get("local")
        binding_epoch = binding.get("epoch")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (layer, expert, rank, local, binding_epoch)):
            raise LaunchError("checkpoint binding coordinates must be integers")
        if not 0 <= layer < LAYERS or not 0 <= expert < 512 or rank < 0 or local < 0 or binding_epoch != epoch:
            raise LaunchError("checkpoint binding coordinate or epoch mismatch")
        if expert not in model_experts or (layer, rank, local) in binding_keys:
            raise LaunchError("checkpoint binding duplicates a layer/rank/local or unknown expert")
        binding_keys.add((layer, rank, local))
        if binding.get("index_sha256") != index_sha256:
            raise LaunchError("checkpoint binding index digest mismatch")
        blob_hash = binding.get("blob_sha256")
        if not isinstance(blob_hash, str) or not _DIGEST.fullmatch(blob_hash):
            raise LaunchError("checkpoint binding blob digest is invalid")
        full_binding = dict(binding)
        full_binding["model_revision"] = model_revision
        by_layer[layer].append(full_binding)
    for layer_bindings in by_layer:
        layer_bindings.sort(key=lambda item: (item["rank"], item["local"]))

    layer_templates = plan.get("layer_templates")
    if not isinstance(layer_templates, list) or len(layer_templates) != LAYERS:
        raise LaunchError("plan must provide 48 layer templates")
    endpoint_specs = manifest.get("endpoints")
    if not isinstance(endpoint_specs, list) or not endpoint_specs or len(endpoint_specs) > 64:
        raise LaunchError("endpoint list is empty or oversized")
    endpoints: dict[int, dict[str, Any]] = {}
    for item in endpoint_specs:
        if not isinstance(item, dict):
            raise LaunchError("invalid endpoint record")
        rank = item.get("rank")
        path = _executable(item.get("executable"), "rank endpoint")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank > 0xFFFFFFFF or rank in endpoints:
            raise LaunchError("rank endpoint ids must be unique u32 values")
        endpoints[rank] = {"rank": rank, "argv": [path]}

    ledger_paths = manifest.get("ledger_paths")
    payload_limits = manifest.get("resident_payload_limits")
    if not isinstance(ledger_paths, list) or len(ledger_paths) != LAYERS:
        raise LaunchError("one existing local ledger directory is required for each layer")
    if not isinstance(payload_limits, list) or len(payload_limits) != LAYERS:
        raise LaunchError("one resident payload cap is required for each layer")
    canonical_ledgers: set[Path] = set()
    normalized_ledgers: list[str] = []
    total_payload_cap = 0
    for layer, path_value in enumerate(ledger_paths):
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise LaunchError("ledger paths must be absolute")
        path = Path(path_value)
        try:
            info = path.lstat()
            canonical = path.resolve(strict=True)
        except OSError as exc:
            raise LaunchError(f"cannot inspect layer {layer} ledger") from exc
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or canonical in canonical_ledgers:
            raise LaunchError("ledger paths must be distinct existing real directories")
        canonical_ledgers.add(canonical)
        normalized_ledgers.append(str(canonical))
        cap = payload_limits[layer]
        if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= MAX_NATIVE_PAYLOAD_BYTES:
            raise LaunchError("per-layer resident payload cap is invalid")
        total_payload_cap += cap
    aggregate_cap = manifest.get("aggregate_resident_payload_limit")
    if (isinstance(aggregate_cap, bool) or not isinstance(aggregate_cap, int) or
            not total_payload_cap <= aggregate_cap <= MAX_AGGREGATE_PAYLOAD_BYTES):
        raise LaunchError("aggregate payload cap must cover all per-layer caps and stay within the hard limit")

    init_timeout = manifest.get("coordinator_timeout_ms")
    if isinstance(init_timeout, bool) or not isinstance(init_timeout, int) or not 1 <= init_timeout <= 60_000:
        raise LaunchError("coordinator_timeout_ms must be in 1..60000")
    strata_timeout = manifest.get("strata_timeout_ms", 5000)
    if isinstance(strata_timeout, bool) or not isinstance(strata_timeout, int) or not 1 <= strata_timeout <= 60_000:
        raise LaunchError("strata_timeout_ms must be in 1..60000")

    sessions: list[dict[str, Any]] = []
    for layer in range(LAYERS):
        template = copy.deepcopy(layer_templates[layer])
        if not isinstance(template, dict) or not isinstance(template.get("stamp"), dict):
            raise LaunchError(f"plan layer {layer} has no routing stamp")
        stamp = template["stamp"]
        stamp.update({"run_id": run_id, "model_revision": model_revision, "plan_id": plan_id,
                      "layer": layer, "batch": first_batch, "epoch": epoch})
        if template.get("hidden") != HIDDEN or template.get("experts") != model_experts:
            raise LaunchError(f"plan layer {layer} does not match the SC117 model shape")
        registry = template.get("registry")
        if not isinstance(registry, dict) or registry.get("epoch") != epoch:
            raise LaunchError(f"plan layer {layer} has a mismatched registry epoch")
        resident_order = template.get("resident_order")
        if not isinstance(resident_order, dict) or not resident_order:
            raise LaunchError(f"plan layer {layer} has no resident ranks")
        rank_ids: set[int] = set()
        for key, order in resident_order.items():
            if not isinstance(key, str) or not key.isdecimal() or not isinstance(order, list):
                raise LaunchError(f"plan layer {layer} has an invalid resident order")
            rank = int(key)
            if rank not in endpoints or rank in rank_ids or not order:
                raise LaunchError(f"plan layer {layer} references an unconfigured rank")
            rank_ids.add(rank)
        layer_bindings = by_layer[layer]
        if not layer_bindings:
            raise LaunchError(f"checkpoint has no resident bindings for layer {layer}")
        init = {
            "template": template,
            "model_experts": model_experts,
            "index": index,
            "pack_path": str(pack_path),
            "ledger_path": normalized_ledgers[layer],
            "endpoints": [endpoints[r] for r in sorted(rank_ids)],
            "bindings": layer_bindings,
            "resident_payload_limit": payload_limits[layer],
            "timeout_ms": init_timeout,
        }
        encoded = json.dumps(init, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_INIT_BYTES:
            raise LaunchError(f"layer {layer} coordinator init exceeds its frame bound")
        sessions.append({"layer": layer, "init": init, "expected_ranks": sorted(rank_ids)})

    strata_executable = _executable(manifest.get("strata_executable"), "Strata executable")
    bridge_executable = _executable(manifest.get("bridge_executable"), "VoltaFabric bridge executable")
    cwd_value = manifest.get("working_directory")
    if not isinstance(cwd_value, str) or not Path(cwd_value).is_absolute():
        raise LaunchError("working_directory must be absolute")
    cwd = Path(cwd_value)
    try:
        cwd_stat = cwd.lstat()
    except OSError as exc:
        raise LaunchError("cannot stat working_directory") from exc
    if not stat.S_ISDIR(cwd_stat.st_mode) or stat.S_ISLNK(cwd_stat.st_mode):
        raise LaunchError("working_directory must be a real directory")
    strata_args = manifest.get("strata_args")
    if not isinstance(strata_args, list) or any(not isinstance(arg, str) or "\0" in arg for arg in strata_args):
        raise LaunchError("strata_args must be an array of strings")
    reserved = {"--vf-session-fds", "--vf-run-id", "--vf-model-revision", "--vf-plan-id",
                "--vf-epoch", "--vf-first-batch", "--vf-first-attempt", "--vf-timeout-ms"}
    if any(arg in reserved or arg.startswith("--vf-") for arg in strata_args):
        raise LaunchError("strata_args may not override launcher-owned VoltaFabric arguments")
    if any(arg in {"--no_pool", "--no_capture"} for arg in strata_args):
        raise LaunchError("Strata bypass options are forbidden for VoltaFabric generation")
    if sys.platform != "linux":
        raise LaunchError("the inherited AF_UNIX Strata launcher is Linux-only")

    return {
        "run_id": run_id,
        "model_revision": model_revision,
        "plan_id": plan_id,
        "epoch": epoch,
        "first_batch": first_batch,
        "first_attempt": first_attempt,
        "startup_timeout_ms": startup_timeout,
        "shutdown_timeout_ms": shutdown_timeout,
        "strata_timeout_ms": strata_timeout,
        "bridge_executable": bridge_executable,
        "strata_executable": strata_executable,
        "strata_args": strata_args,
        "working_directory": str(cwd),
        "sessions": sessions,
    }


@dataclass
class BridgeSession:
    layer: int
    process: subprocess.Popen[bytes]
    channel: socket.socket
    identity: bytes


def _read_line(channel: socket.socket, deadline: float, limit: int = 1_048_576) -> bytes:
    result = bytearray()
    while len(result) < limit:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("coordinator handshake deadline")
        channel.settimeout(remaining)
        chunk = channel.recv(1)
        if not chunk:
            raise LaunchError("coordinator closed during handshake")
        if chunk == b"\n":
            return bytes(result)
        result.extend(chunk)
    raise LaunchError("coordinator handshake frame exceeds bound")


def _read_exact(channel: socket.socket, size: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("coordinator binary shutdown deadline")
        channel.settimeout(remaining)
        chunk = channel.recv(size - len(data))
        if not chunk:
            raise LaunchError("coordinator closed before binary shutdown reply")
        data.extend(chunk)
    return bytes(data)


def _start_bridge(config: dict[str, Any], session: dict[str, Any], deadline: float) -> BridgeSession:
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    child: subprocess.Popen[bytes] | None = None
    try:
        child = subprocess.Popen([config["bridge_executable"]], stdin=server, stdout=server,
                                 cwd=config["working_directory"], close_fds=True)
    except OSError as exc:
        client.close()
        server.close()
        raise LaunchError("could not start a layer coordinator") from exc
    finally:
        server.close()
    try:
        init_bytes = json.dumps(session["init"], separators=(",", ":"), allow_nan=False).encode("utf-8")
        client.sendall(init_bytes + b"\n")
        ready = _json_load(_read_line(client, deadline), "coordinator ready reply")
        if not isinstance(ready, dict) or ready.get("state") != "ready" or sorted(ready.get("ranks", [])) != session["expected_ranks"]:
            raise LaunchError(f"layer {session['layer']} coordinator did not admit the expected ranks")
        client.sendall(b'{"op":"enter_binary"}\n')
        binary_ready = _json_load(_read_line(client, deadline), "coordinator binary-ready reply")
        if binary_ready != {"state": "binary_ready"}:
            raise LaunchError(f"layer {session['layer']} coordinator refused binary mode")
        if child.poll() is not None:
            raise LaunchError(f"layer {session['layer']} coordinator exited during admission")
        identity = b"".join(session["init"]["template"]["stamp"][key].encode("ascii")
                             for key in ("run_id", "model_revision", "plan_id"))
        return BridgeSession(session["layer"], child, client, identity)
    except Exception:
        client.close()
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise


def _shutdown_frame(session: BridgeSession, config: dict[str, Any]) -> bytes:
    # The identity bytes were validated at creation; reconstruct integer fields from
    # the manifest-bound template to avoid storing another independently mutable copy.
    template = config["sessions"][session.layer]["init"]["template"]["stamp"]
    run_id = template["run_id"].encode("ascii")
    model = template["model_revision"].encode("ascii")
    plan = template["plan_id"].encode("ascii")
    header = struct.pack("<4sIQIIIQHHHHQ", b"VFB3", 2, 0, 0, 0, session.layer,
                         config["epoch"], len(run_id), len(model), len(plan), 0, 0)
    return header + run_id + model + plan


def _validate_shutdown_reply(session: BridgeSession, config: dict[str, Any], deadline: float) -> None:
    header = _read_exact(session.channel, 52, deadline)
    fields = struct.unpack("<4sIQIIIQHHHHQ", header)
    magic, status, batch, rows, columns, layer, epoch, run_len, model_len, plan_len, reserved, attempt = fields
    template = config["sessions"][session.layer]["init"]["template"]["stamp"]
    expected_lengths = tuple(len(template[key].encode("ascii")) for key in ("run_id", "model_revision", "plan_id"))
    if (magic != b"VFR3" or status != 2 or batch != 0 or rows != 0 or columns != HIDDEN or
        layer != session.layer or epoch != config["epoch"] or reserved != 0 or attempt != 0 or
        (run_len, model_len, plan_len) != expected_lengths):
        raise LaunchError(f"layer {session.layer} returned an invalid shutdown header")
    echoed = _read_exact(session.channel, run_len + model_len + plan_len, deadline)
    if echoed != session.identity:
        raise LaunchError(f"layer {session.layer} shutdown identity mismatch")


def _strata_child_environment() -> dict[str, str]:
    """Keep the pinned IQ3_S path off Strata's known-unsafe stage-buffer pinning.

    Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf documents that
    STRATA_STAGE_PIN=1 can corrupt IQ3_S after a long prompt. The launcher is
    bound to the SC117 IQ3 fixture, so do not inherit an opt-in from the shell.
    This only changes the child environment; it leaves the caller untouched.
    """
    env = os.environ.copy()
    env["STRATA_STAGE_PIN"] = "0"
    return env


def _finish_sessions(sessions: list[BridgeSession], config: dict[str, Any], graceful: bool) -> bool:
    failed = False
    deadline = time.monotonic() + config["shutdown_timeout_ms"] / 1000.0
    if graceful:
        for session in sessions:
            if session.process.poll() is not None:
                failed = True
                continue
            try:
                session.channel.sendall(_shutdown_frame(session, config))
            except OSError:
                failed = True
        for session in sessions:
            if session.process.poll() is None:
                try:
                    _validate_shutdown_reply(session, config, deadline)
                except (OSError, TimeoutError, LaunchError, struct.error):
                    failed = True
    for session in sessions:
        try:
            session.channel.close()
        except OSError:
            failed = True
    wait_deadline = time.monotonic() + config["shutdown_timeout_ms"] / 1000.0
    for session in sessions:
        try:
            session.process.wait(timeout=max(0.001, wait_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            failed = True
            session.process.terminate()
            try:
                session.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                session.process.kill()
                session.process.wait()
        if session.process.returncode != 0:
            failed = True
    return not failed


def launch(config: dict[str, Any]) -> int:
    sessions: list[BridgeSession] = []
    strata_child: subprocess.Popen[bytes] | None = None
    started_generation = False
    try:
        deadline = time.monotonic() + config["startup_timeout_ms"] / 1000.0
        for session in config["sessions"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("48-session startup deadline")
            sessions.append(_start_bridge(config, session, deadline))
        fds = tuple(session.channel.fileno() for session in sessions)
        if len(fds) != LAYERS or len(set(fds)) != LAYERS or any(fd < 3 for fd in fds):
            raise LaunchError("launcher did not construct 48 unique inheritable session descriptors")
        if any(session.process.poll() is not None for session in sessions):
            raise LaunchError("a coordinator exited before Strata generation started")
        stamp = config["sessions"][0]["init"]["template"]["stamp"]
        args = [
            config["strata_executable"], *config["strata_args"],
            "--vf-session-fds", ",".join(str(fd) for fd in fds),
            "--vf-run-id", stamp["run_id"],
            "--vf-model-revision", stamp["model_revision"],
            "--vf-plan-id", stamp["plan_id"],
            "--vf-epoch", str(config["epoch"]),
            "--vf-first-batch", str(config["first_batch"]),
            "--vf-first-attempt", str(config["first_attempt"]),
            "--vf-timeout-ms", str(config["strata_timeout_ms"]),
        ]
        strata_child = subprocess.Popen(args, cwd=config["working_directory"], pass_fds=fds,
                                        close_fds=True, env=_strata_child_environment())
        started_generation = True
        try:
            result = strata_child.wait()
        except KeyboardInterrupt:
            strata_child.terminate()
            try:
                strata_child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                strata_child.kill()
                strata_child.wait()
            result = 130
        # On a successful Strata return, its PoolFleetOwner has already sent a
        # terminal VFB3 request to every layer. On failures, do not replay; closing
        # the streams makes any in-flight result UNKNOWN and the coordinator exits.
        cleanup_ok = _finish_sessions(sessions, config, graceful=not started_generation)
        if result == 0 and not cleanup_ok:
            return 1
        return result
    except KeyboardInterrupt:
        if strata_child is not None and strata_child.poll() is None:
            strata_child.terminate()
            try:
                strata_child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                strata_child.kill()
                strata_child.wait()
        _finish_sessions(sessions, config, graceful=not started_generation)
        return 130
    except Exception as exc:
        if strata_child is not None and strata_child.poll() is None:
            strata_child.terminate()
            try:
                strata_child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                strata_child.kill()
                strata_child.wait()
        cleanup_ok = _finish_sessions(sessions, config, graceful=not started_generation)
        if not cleanup_ok:
            print("voltafabric-launch: coordinator cleanup was incomplete; do not replay any submitted attempt", file=sys.stderr)
        if isinstance(exc, KeyboardInterrupt):
            return 130
        raise LaunchError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="absolute local launch manifest path")
    args = parser.parse_args(argv)
    try:
        return launch(_check_manifest(args.manifest))
    except LaunchError as exc:
        print(f"voltafabric-launch: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
