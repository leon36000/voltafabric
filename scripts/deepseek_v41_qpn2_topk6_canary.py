#!/usr/bin/env python3
"""One-shot DeepSeek top-k=6 QPN2 layer canary for the pinned A133 candidate.

The test uses six real compressed layer-0 experts and Rust RankDispatch, but its
two-token routing vectors are a deterministic routing stress fixture, not output
from the model's router. It does not qualify the complete model or service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import struct

ROOT = Path(__file__).resolve().parents[1]
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
EXPERT_IDS = (0, 1, 2, 96, 192, 383)
LOCAL_IDS = tuple(range(6))
TOKENS = 2
TOP_K = 6
HIDDEN = 5120
INTERMEDIATE = 2304
SHAPE = [6, 4608, 5120, 5120, 2304]
PAYLOAD_BYTES = 119_446_656
MAX_RESIDENT_BYTES = 128 * 1024 * 1024
TERMINAL_STATUSES = {
    "FAILED_BEFORE_GPU",
    "A133_DEEPSEEK_QPN2_TOPK6_LAYER_PASS",
    "A133_DEEPSEEK_QPN2_TOPK6_LAYER_FAILURE_NO_REPLAY",
    "TERMINAL_GPU_FAILURE_AFTER_LAUNCH_NO_REPLAY",
    "TERMINAL_FAILURE_AFTER_CONTEXT_NO_KERNEL_NO_REPLAY",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_exact(actual, expected) -> bool:
    """Compare JSON values without Python's bool/int equality aliasing."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _json_exact(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _read_bounded_regular(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("RankDispatch inputs must be regular files")
        if before.st_size > limit:
            raise ValueError(f"RankDispatch input exceeds the {limit}-byte cap")
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if len(raw) > limit:
            raise ValueError(f"RankDispatch input exceeds the {limit}-byte cap")
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or len(raw) != after.st_size
        ):
            raise ValueError("RankDispatch input changed while being read")
        return raw
    finally:
        os.close(fd)


def expected_artifact_info(*, layer: int, epoch: int) -> dict:
    return {
        "model_revision": REVISION,
        "layer": layer,
        "rank": 0,
        "epoch": epoch,
        "shape": SHAPE,
        "physical_experts": [
            {"id": expert, "local": local}
            for expert, local in zip(EXPERT_IDS, LOCAL_IDS, strict=True)
        ],
        "resident_order": list(LOCAL_IDS),
        "payload_bytes": PAYLOAD_BYTES,
    }


def validate_shape(*, rows, hidden, intermediate, top_k) -> bool:
    values = (rows, hidden, intermediate, top_k)
    if any(type(value) is not int for value in values):
        raise ValueError("DeepSeek top-k=6 dimensions must be integers")
    if values != (TOKENS, HIDDEN, INTERMEDIATE, TOP_K):
        raise ValueError("only the exact A133 two-token top-k=6 shape is admitted")
    return True


def _stamp_is_exact(value, expected) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "run_id",
            "model_revision",
            "plan_id",
            "layer",
            "batch",
            "epoch",
        }
        and value == expected
        and all(type(value[key]) is type(expected[key]) for key in expected)
    )


def validate_rank_dispatch(document, *, expected_identity):
    """Validate full Rust problem/dispatch identity and all 12 token-expert slots."""
    from scripts import deepseek_v41_qpn2_layer_canary as base

    if not isinstance(expected_identity, dict) or set(expected_identity) != {
        "run_id",
        "model_revision",
        "plan_id",
        "layer",
        "batch",
        "epoch",
    }:
        raise ValueError("expected A133 dispatch Stamp is incomplete")
    if any(
        type(expected_identity[key]) is not int for key in ("layer", "batch", "epoch")
    ) or any(
        type(expected_identity[key]) is not str
        for key in ("run_id", "model_revision", "plan_id")
    ):
        raise ValueError("expected A133 dispatch Stamp has invalid JSON types")
    if expected_identity["model_revision"] != REVISION:
        raise ValueError("A133 dispatch revision differs from the pinned model")
    if not isinstance(document, dict) or set(document) != {"problem", "dispatch"}:
        raise ValueError("RankDispatch must contain problem and dispatch")
    problem, dispatch = document["problem"], document["dispatch"]
    if not isinstance(problem, dict) or not isinstance(dispatch, dict):
        raise ValueError("RankDispatch problem and dispatch must be objects")
    if set(problem) != {
        "stamp",
        "hidden",
        "experts",
        "preferred_owners",
        "registry",
        "resident_order",
        "routes",
    }:
        raise ValueError("RankDispatch problem fields differ from the Rust contract")
    stamp = problem.get("stamp")
    if not _stamp_is_exact(stamp, expected_identity) or not _stamp_is_exact(
        dispatch.get("stamp"), expected_identity
    ):
        raise ValueError("RankDispatch Stamp is stale or incoherent")

    copies = [
        {"expert": expert, "local": local, "rank": 0}
        for expert, local in zip(EXPERT_IDS, LOCAL_IDS, strict=True)
    ]
    if (
        type(problem.get("hidden")) is not int
        or problem["hidden"] != HIDDEN
        or not _json_exact(problem.get("experts"), list(EXPERT_IDS))
        or not isinstance(problem.get("preferred_owners"), dict)
        or problem["preferred_owners"] != {}
        or not _json_exact(
            problem.get("registry"),
            {"epoch": expected_identity["epoch"], "copies": copies},
        )
        or not _json_exact(problem.get("resident_order"), {"0": list(LOCAL_IDS)})
    ):
        raise ValueError("RankDispatch registry or resident map differs from A133")
    if (
        set(dispatch)
        != {"hidden", "rank", "resident_order", "tokens", "slots", "stamp"}
        or type(dispatch.get("hidden")) is not int
        or dispatch["hidden"] != HIDDEN
        or type(dispatch.get("rank")) is not int
        or dispatch["rank"] != 0
        or not _json_exact(dispatch.get("resident_order"), list(LOCAL_IDS))
        or type(dispatch.get("tokens")) is not int
        or dispatch["tokens"] != TOKENS
    ):
        raise ValueError("RankDispatch rank, dimensions or physical order differ")

    routes = problem.get("routes")
    if not isinstance(routes, list) or len(routes) != TOKENS:
        raise ValueError("RankDispatch must contain exactly two token routes")
    route_ids, route_weights = [], []
    expected = {}
    id_to_local = dict(zip(EXPERT_IDS, LOCAL_IDS, strict=True))
    for token, route in enumerate(routes):
        if not isinstance(route, list) or len(route) != TOP_K:
            raise ValueError("each A133 token must have exactly six top-k routes")
        ids, weights = [], []
        for slot, choice in enumerate(route):
            if not isinstance(choice, dict) or set(choice) != {"expert", "weight"}:
                raise ValueError("RankDispatch route entry is malformed")
            expert, weight = choice["expert"], choice["weight"]
            if type(expert) is not int or expert not in id_to_local:
                raise ValueError(
                    "RankDispatch route expert is outside the six residents"
                )
            if (
                type(weight) not in (int, float)
                or not base.is_finite_f32(weight)
                or not 0.0 < float(weight) <= 1.0
            ):
                raise ValueError(
                    "RankDispatch route weight is not positive finite FP32"
                )
            ids.append(expert)
            weights.append(float(weight))
            expected[(token, slot)] = (expert, id_to_local[expert], float(weight))
        if len(set(ids)) != TOP_K or set(ids) != set(EXPERT_IDS):
            raise ValueError("each token must select six unique resident experts")
        if not math.isclose(math.fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("top-k route coefficients must preserve unit mass")
        route_ids.append(ids)
        route_weights.append(weights)

    entries = dispatch.get("slots")
    if not isinstance(entries, list) or len(entries) != TOKENS * TOP_K:
        raise ValueError("RankDispatch must cover all 12 A133 logical slots")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"physical_expert", "slot"}:
            raise ValueError("RankDispatch physical slot is malformed")
        slot = entry["slot"]
        if not isinstance(slot, dict) or set(slot) != {
            "expert",
            "local",
            "rank",
            "slot",
            "token",
            "weight",
        }:
            raise ValueError("RankDispatch logical slot identity is malformed")
        token, index = slot["token"], slot["slot"]
        if (
            type(token) is not int
            or type(index) is not int
            or not 0 <= token < TOKENS
            or not 0 <= index < TOP_K
            or (token, index) in seen
        ):
            raise ValueError("duplicate or out-of-domain RankDispatch slot")
        expert, local, weight = expected[(token, index)]
        if (
            type(entry["physical_expert"]) is not int
            or entry["physical_expert"] != local
            or type(slot["expert"]) is not int
            or slot["expert"] != expert
            or type(slot["local"]) is not int
            or slot["local"] != local
            or type(slot["rank"]) is not int
            or slot["rank"] != 0
            or type(slot["weight"]) not in (int, float)
            or not base.is_finite_f32(slot["weight"])
            or struct.pack("<f", float(slot["weight"])) != struct.pack("<f", weight)
        ):
            raise ValueError("RankDispatch global/local slot or coefficient differs")
        seen.add((token, index))
    if seen != {(token, index) for token in range(TOKENS) for index in range(TOP_K)}:
        raise ValueError("RankDispatch slot coverage is incomplete")
    return {
        "route_ids": route_ids,
        "route_weights": route_weights,
        "global_to_local": id_to_local,
        "problem": problem,
        "dispatch": dispatch,
        "stamp": stamp,
    }


def validate_artifact_metadata(info, expected):
    if not isinstance(info, dict) or not _json_exact(info, expected):
        raise ValueError("resident artifact identity differs from pinned A133 metadata")
    if info.get("shape") != SHAPE or info.get("payload_bytes") != PAYLOAD_BYTES:
        raise ValueError("resident artifact shape or compressed size differs")
    if [item.get("id") for item in info.get("physical_experts", [])] != list(
        EXPERT_IDS
    ):
        raise ValueError("resident artifact must contain the exact six A133 experts")
    return info


def _cpu_gate_up_oracle(np, tensors, inputs, route_ids):
    from scripts import deepseek_v41_qpn2_layer_canary as base

    codes = tensors["codes13"].numpy().reshape(6, -1)
    scales = tensors["scales13"].numpy().reshape(6, -1)
    factors = tensors["gscales13"].numpy()
    exact = np.empty((TOKENS * TOP_K, 4608), dtype=np.float64)
    absolute = np.empty_like(exact)
    route_local = []
    fraction_errors = []
    id_to_local = dict(zip(EXPERT_IDS, LOCAL_IDS, strict=True))
    for token, ids in enumerate(route_ids):
        route_local.extend(id_to_local[expert] for expert in ids)
    for local, expert in enumerate(EXPERT_IDS):
        weight = base.decode_dense_qpn2(
            np,
            memoryview(codes[local]),
            memoryview(scales[local]),
            np.asarray(factors[local], dtype="<f4").tobytes(),
            4608,
            HIDDEN,
        )
        token_exact, token_absolute = base.dense_dot_oracle(np, weight, inputs)
        for token, ids in enumerate(route_ids):
            for slot, route_expert in enumerate(ids):
                if route_expert != expert:
                    continue
                row = token * TOP_K + slot
                exact[row] = token_exact[token]
                absolute[row] = token_absolute[token]
                for column in (0, 31, 32, 2304, 4607):
                    rational = base.fraction_dot(np, weight, inputs[token], column)
                    fraction_errors.append(
                        abs(float(rational) - float(token_exact[token, column]))
                    )
        del weight
    if max(fraction_errors, default=0.0) > 2.0**-45:
        raise ValueError("A133 gate/up rational samples disagree with FP64 oracle")
    return exact, absolute, route_local, max(fraction_errors, default=0.0)


def _down_oracle(
    np, tensors, route_local, activation_actual, activation_reference, down_actual
):
    from scripts import deepseek_v41_qpn2_layer_canary as base

    codes = tensors["codes2"].numpy().reshape(6, -1)
    scales = tensors["scales2"].numpy().reshape(6, -1)
    factors = tensors["gscales2"].numpy()
    exact_actual = np.empty_like(down_actual, dtype=np.float64)
    absolute = np.empty_like(exact_actual)
    exact_reference = np.empty_like(exact_actual)
    bound = np.empty_like(exact_actual)
    delta = activation_actual.astype(np.float64) - activation_reference.astype(
        np.float64
    )
    unit = 2.0**-24
    operations = 2 * INTERMEDIATE + 16
    gamma = operations * unit / (1.0 - operations * unit)
    for local in LOCAL_IDS:
        rows = np.asarray(
            [index for index, value in enumerate(route_local) if value == local],
            dtype=np.int64,
        )
        if rows.size != TOKENS:
            raise ValueError("A133 route fixture must use each resident once per token")
        weight = base.decode_dense_qpn2(
            np,
            memoryview(codes[local]),
            memoryview(scales[local]),
            np.asarray(factors[local], dtype="<f4").tobytes(),
            HIDDEN,
            INTERMEDIATE,
        )
        actual_part, absolute_part = base.dense_dot_oracle(
            np, weight, activation_actual[rows]
        )
        reference_part, _ = base.dense_dot_oracle(
            np, weight, activation_reference[rows]
        )
        propagated = base.dense_magnitude_dot_oracle(np, weight, delta[rows])
        accumulation = gamma * absolute_part
        half_round = 2.0**-11 * (np.abs(reference_part) + accumulation) + 2.0**-25
        exact_actual[rows] = actual_part
        absolute[rows] = absolute_part
        exact_reference[rows] = reference_part
        bound[rows] = accumulation + half_round + propagated
        del weight

    kernel = base.compare_full_output(
        np, down_actual, exact_actual, absolute, INTERMEDIATE
    )
    error = np.abs(down_actual.astype(np.float64) - exact_reference)
    ratio = np.divide(error, bound, out=np.zeros_like(error), where=bound > 0)
    outside = int(np.count_nonzero(error > bound + 2.0**-52))
    stagewise = {
        "values_checked": int(down_actual.size),
        "outputs_outside_bound": outside,
        "max_abs_error": float(error.max(initial=0.0)),
        "max_abs_bound": float(bound.max(initial=0.0)),
        "max_error_bound_ratio": float(ratio.max(initial=0.0)),
        "activation_error_propagated": True,
        "correct": outside == 0,
    }
    return {
        "kernel_vs_actual_activation": kernel,
        "stagewise_full_down": stagewise,
        "exact_reference": exact_reference,
        "bound": bound,
    }


def _combine_oracle(np, route_weights, down_actual, exact_down, down_bound, combined):
    from scripts import deepseek_v41_qpn2_layer_canary as base

    coeff32 = np.asarray(route_weights, dtype=np.float32)
    coeff64 = coeff32.astype(np.float64)
    exact_layer = np.zeros((TOKENS, HIDDEN), dtype=np.float64)
    propagated = np.zeros_like(exact_layer)
    product_error = np.zeros_like(exact_layer)
    product_abs = np.zeros_like(exact_layer)
    sequential = np.zeros((TOKENS, HIDDEN), dtype=np.float32)
    for token in range(TOKENS):
        for slot in range(TOP_K):
            row = token * TOP_K + slot
            weight64, weight32 = coeff64[token, slot], coeff32[token, slot]
            exact_layer[token] += exact_down[row] * weight64
            propagated[token] += down_bound[row] * abs(weight64)
            products64 = down_actual[row].astype(np.float64) * weight64
            products32 = (down_actual[row].astype(np.float32) * weight32).astype(
                np.float64
            )
            product_error[token] += np.abs(products32 - products64)
            product_abs[token] += np.abs(products32)
            sequential[token] += products32.astype(np.float32)
    accumulation_bound = (TOP_K * 2.0**-24 / (1.0 - TOP_K * 2.0**-24)) * product_abs
    half_round = (
        2.0**-11
        * (np.abs(exact_layer) + propagated + product_error + accumulation_bound)
        + 2.0**-25
    )
    layer_bound = propagated + product_error + accumulation_bound + half_round
    layer_error = np.abs(combined.astype(np.float64) - exact_layer)
    ratio = np.divide(
        layer_error, layer_bound, out=np.zeros_like(layer_error), where=layer_bound > 0
    )
    outside = int(np.count_nonzero(layer_error > layer_bound + 2.0**-52))
    rounded_from_actual_down = sequential.astype(np.float16)
    local_combine = base.compare_binary16_ulp(
        np, combined, rounded_from_actual_down, max_ulps=1.0
    )
    return {
        "values_checked": int(combined.size),
        "outputs_outside_weighted_fp64_bound": outside,
        "max_abs_error": float(layer_error.max(initial=0.0)),
        "max_abs_bound": float(layer_bound.max(initial=0.0)),
        "max_error_bound_ratio": float(ratio.max(initial=0.0)),
        "independent_fp32_route_accumulate": local_combine,
        "coefficients_by_token": coeff32.astype(float).tolist(),
        "coefficient_application_count_per_route": 1,
        "correct": outside == 0 and local_combine["correct"],
    }


def full_layer_pass_eligible(report) -> bool:
    required = (
        "rust_rank_dispatch_bound",
        "gate_up_correct",
        "swiglu_correct",
        "down_correct",
        "weighted_combine_correct",
        "full_layer_correct",
    )
    return (
        isinstance(report, dict)
        and all(report.get(key) is True for key in required)
        and type(report.get("kernel_launch_count")) is int
        and report["kernel_launch_count"] == 2
        and type(report.get("provider_synchronized_launches")) is int
        and report["provider_synchronized_launches"] == 2
    )


def receipt_path(run_id):
    from scripts.deepseek_v41_qpn2_layer_canary import validate_run_id

    explicit = os.environ.get("VF_H1_RECEIPT_DIR")
    root = Path(explicit).resolve(strict=True) if explicit else ROOT.resolve()
    directory = root if explicit else root / ".runs/deepseek-v41-qpn2-topk6-canary"
    candidate = directory / (validate_run_id(run_id) + ".json")
    try:
        candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("A133 receipt escapes its approved root") from exc
    return candidate


def classify_receipt(report):
    return (
        report.get("status")
        if isinstance(report, dict) and report.get("status") in TERMINAL_STATUSES
        else "UNKNOWN_DO_NOT_REPLAY"
    )


def _run_canary(args, report, receipt):
    from scripts import deepseek_v41_qpn2_layer_canary as base
    from scripts.hardware_canary import (
        QPN2Provider,
        align_reference,
        check_cuda_proof,
    )
    from scripts.resident_artifact import load_stack

    expected_info = expected_artifact_info(layer=args.layer, epoch=args.artifact_epoch)
    artifact = load_stack(
        args.artifact,
        expected=expected_info,
        expected_sha256=args.artifact_sha256,
    )
    validate_artifact_metadata(artifact.metadata, expected_info)
    dispatch_raw = _read_bounded_regular(args.dispatch, 1024 * 1024)
    if sha256(dispatch_raw) != args.dispatch_sha256:
        raise ValueError("A133 RankDispatch SHA-256 mismatch")
    routed = validate_rank_dispatch(
        json.loads(dispatch_raw),
        expected_identity={
            "run_id": args.run_id,
            "model_revision": args.model_revision,
            "plan_id": args.plan_id,
            "layer": args.layer,
            "batch": args.batch,
            "epoch": args.dispatch_epoch,
        },
    )
    provenance_raw = _read_bounded_regular(args.dispatch_provenance, 1024 * 1024)
    provenance = base.validate_dispatch_provenance(
        json.loads(provenance_raw), args.dispatch_sha256
    )
    validate_shape(
        rows=routed["dispatch"]["tokens"],
        hidden=routed["dispatch"]["hidden"],
        intermediate=INTERMEDIATE,
        top_k=TOP_K,
    )
    if args.model_revision != REVISION or args.batch != 1 or args.layer != 0:
        raise ValueError("A133 only admits the pinned layer-0 batch-1 route plan")
    if any(not path.is_file() for path in (args.library, args.cuda_proof, args.cudart)):
        raise ValueError("A133 SM70 library, proof, or CUDA runtime is missing")

    import numpy as np

    inputs = base.make_input_batches(np, HIDDEN)[0]
    gate_exact, gate_absolute, route_local, rational_error = _cpu_gate_up_oracle(
        np, artifact.tensors, inputs, routed["route_ids"]
    )
    report.update(
        model_revision=REVISION,
        plan_id=args.plan_id,
        layer=args.layer,
        batch=args.batch,
        artifact_epoch=args.artifact_epoch,
        dispatch_epoch=args.dispatch_epoch,
        resident_artifact_sha256=args.artifact_sha256,
        resident_artifact_bytes=args.artifact.stat().st_size,
        resident_payload_bytes=PAYLOAD_BYTES,
        artifact_identity=artifact.metadata,
        rank_dispatch_sha256=args.dispatch_sha256,
        rank_dispatch_provenance_sha256=sha256(provenance_raw),
        rust_dispatch_binary_sha256=provenance["binary_sha256"],
        rank_dispatch_stamp=routed["stamp"],
        route_ids=routed["route_ids"],
        route_weights=routed["route_weights"],
        global_to_local={
            str(key): value for key, value in routed["global_to_local"].items()
        },
        routing_fixture_kind="deterministic stress vectors; not model-router output",
        adapter_entrypoint="adapters.skinny_moe.moe_apply",
        cpu_gate_up_oracle={
            "status": "PASS_BEFORE_CUDA_CONTEXT",
            "values_checked": int(gate_exact.size),
            "rational_samples": TOKENS * TOP_K * 5,
            "max_rational_fp64_delta": rational_error,
            "temporary_dense_cpu_weight_matrices_simultaneous": 1,
        },
        dense_weight_gpu_allocation=False,
        cuda_executed=False,
        kernel_launch_count=0,
        automatic_retry=False,
        replay_allowed=False,
    )
    torch = __import__("torch")
    preflight = base.gpu_preflight(torch, args.gpu_uuid)
    library_sha = sha256(args.library.read_bytes())
    check_cuda_proof(args.cuda_proof, library_sha)
    report.update(
        status="REQUEST_ARMED",
        expected_gpu_uuid=args.gpu_uuid,
        library_sha256=library_sha,
        cuda_compile_proof_sha256=sha256(args.cuda_proof.read_bytes()),
        prelaunch_preflight=preflight,
        cuda_context_initialized=False,
        notes=[
            "Exactly six pinned compressed layer-0 experts, 12 route slots, batch=2, top-k=6.",
            "Synthetic deterministic route fixture exercises the production adapter and FFI; it is not model router output.",
            "One gate/up launch and one down launch; no dense GPU weights, graph, throughput, or full-model qualification.",
        ],
    )
    receipt.update(report)

    provider = None
    try:
        provider = QPN2Provider(
            args.library,
            args.cudart,
            0,
            args.gpu_uuid,
            ack_hardware_gate=True,
            gate_path=ROOT / "docs/HARDWARE_GATE.md",
        )
        provider.full_expert_enabled = True
        provider.deepseek_probe_enabled = True
        provider.adapter_layer_enabled = True

        def on_progress(event):
            report["cuda_context_initialized"] = True
            report["kernel_launch_count"] = provider.launches
            report["cuda_executed"] = provider.launches > 0
            report["status"] = (
                "GPU_EXECUTION_MAY_HAVE_STARTED"
                if event.get("execution_phase") == "ENQUEUE_PENDING"
                else "GPU_RUNNING"
            )
            report["last_provider_event"] = event
            receipt.update(report)

        provider.progress = on_progress
        report.update(device=provider.identity, cuda_context_initialized=True)
        receipt.update(report)

        from adapters import skinny_moe as adapter
        from types import SimpleNamespace

        expert_map = torch.full(
            (max(EXPERT_IDS) + 1,), -1, dtype=torch.int64, device=provider.device
        )
        for expert, local in zip(EXPERT_IDS, LOCAL_IDS, strict=True):
            expert_map[expert] = local
        layer = SimpleNamespace(expert_map=expert_map)
        owner = adapter.load_resident_artifact(
            layer,
            args.artifact,
            expected_info=expected_info,
            expected_sha256=args.artifact_sha256,
            device=provider.device,
        )
        resident_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                owner._qc13,
                owner._qs13,
                owner._gs13,
                owner._qc2,
                owner._qs2,
                owner._gs2,
            )
        )
        if resident_bytes != PAYLOAD_BYTES or resident_bytes > MAX_RESIDENT_BYTES:
            raise ValueError(
                "A133 compressed resident device bytes exceed the 128 MiB cap"
            )
        report.update(
            resident_artifact_load_calls=1,
            resident_compressed_device_bytes=resident_bytes,
            resident_shape=list(owner.shape),
            resident_artifact_binding=list(owner.artifact_binding[:4])
            + [list(owner.artifact_binding[4])],
        )
        ids = torch.tensor(
            routed["route_ids"], dtype=torch.int64, device=provider.device
        )
        coefficients = torch.tensor(
            routed["route_weights"], dtype=torch.float32, device=provider.device
        )
        dispatch_mask = adapter.bind_rank_dispatch(
            layer, routed["dispatch"], routed["problem"]["stamp"], ids, coefficients
        )
        report["rust_rank_dispatch_bound"] = bool(
            dispatch_mask.shape == ids.shape
            and dispatch_mask.dtype == torch.bool
            and bool(dispatch_mask.all())
        )
        if not report["rust_rank_dispatch_bound"]:
            raise ValueError("Rust RankDispatch did not select all 12 A133 routes")
        x = torch.from_numpy(inputs).to(provider.device).contiguous()
        recorded_inputs, recorded_outputs = [], []
        original_gemm = provider.gemm_qpn2_moe

        def recording_gemm(activation, *gemm_args):
            recorded_inputs.append(activation.detach().clone())
            output = original_gemm(activation, *gemm_args)
            recorded_outputs.append(output)
            return output

        provider.gemm_qpn2_moe = recording_gemm
        combined = adapter.moe_apply(
            layer,
            x,
            coefficients,
            ids,
            None,
            None,
            kernel=provider,
            align=align_reference,
            selected_mask=dispatch_mask,
        )
        provider.synchronize()
        report.update(
            kernel_launch_count=provider.launches,
            provider_synchronized_launches=provider.synchronized_launches,
            cuda_executed=provider.launches > 0,
            activation_h2d_bytes=int(inputs.nbytes),
        )
        if (
            provider.launches != 2
            or provider.synchronized_launches != 2
            or len(recorded_inputs) != 2
            or len(recorded_outputs) != 2
        ):
            raise ValueError(
                "A133 adapter must synchronize one gate/up and one down launch"
            )

        gate_actual = recorded_outputs[0].detach().cpu().contiguous().numpy()
        activation_actual = recorded_inputs[1].detach().cpu().contiguous().numpy()
        down_actual = recorded_outputs[1].detach().cpu().contiguous().numpy()
        combined_actual = combined.detach().cpu().contiguous().numpy()
        gate = base.compare_full_output(
            np, gate_actual, gate_exact, gate_absolute, HIDDEN
        )
        report["gate_up"] = gate
        report["gate_up_correct"] = gate["correct"]
        activation_reference = base.swiglu_binary16_reference(np, gate_actual)
        swiglu = base.compare_binary16_ulp(
            np, activation_actual, activation_reference, max_ulps=1.0
        )
        report["swiglu"] = swiglu
        report["swiglu_correct"] = swiglu["correct"]
        down = _down_oracle(
            np,
            artifact.tensors,
            route_local,
            activation_actual,
            activation_reference,
            down_actual,
        )
        exact_down = down.pop("exact_reference")
        down_bound = down.pop("bound")
        report["down"] = down
        report["down_correct"] = bool(
            down["kernel_vs_actual_activation"]["correct"]
            and down["stagewise_full_down"]["correct"]
        )
        combine = _combine_oracle(
            np,
            routed["route_weights"],
            down_actual,
            exact_down,
            down_bound,
            combined_actual,
        )
        report["weighted_combine"] = combine
        report["weighted_combine_correct"] = combine["correct"]
        report["full_layer_correct"] = bool(
            report["gate_up_correct"]
            and report["swiglu_correct"]
            and report["down_correct"]
            and report["weighted_combine_correct"]
            and report["rust_rank_dispatch_bound"]
        )
        report["outputs_checked_total"] = int(
            gate_actual.size
            + activation_actual.size
            + down_actual.size
            + combined_actual.size
        )
        report["temporary_dense_cpu_peak_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        eligible = full_layer_pass_eligible(report)
        report["terminal_pass_eligible"] = eligible
        report["status"] = (
            "A133_DEEPSEEK_QPN2_TOPK6_LAYER_PASS"
            if eligible
            else "A133_DEEPSEEK_QPN2_TOPK6_LAYER_FAILURE_NO_REPLAY"
        )
        report["replay_allowed"] = False
        receipt.update(report)
        print(json.dumps(report, sort_keys=True))
        return 0 if eligible else 2
    except BaseException as exc:
        pending = bool(provider is not None and provider.pending)
        launched = bool(provider is not None and provider.launches > 0)
        context = bool(torch.cuda.is_initialized())
        report["status"] = (
            "UNKNOWN_DO_NOT_REPLAY"
            if pending
            else (
                "TERMINAL_GPU_FAILURE_AFTER_LAUNCH_NO_REPLAY"
                if launched
                else (
                    "TERMINAL_FAILURE_AFTER_CONTEXT_NO_KERNEL_NO_REPLAY"
                    if context
                    else "FAILED_BEFORE_GPU"
                )
            )
        )
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["kernel_launch_count"] = provider.launches if provider else 0
        report["provider_synchronized_launches"] = (
            provider.synchronized_launches if provider else 0
        )
        report["cuda_executed"] = report["kernel_launch_count"] > 0
        report["replay_allowed"] = False
        receipt.update(report)
        print(json.dumps(report, sort_keys=True))
        return 3 if pending else 2


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="validate and execute one explicit A133 attempt")
    for name in (
        "artifact",
        "dispatch",
        "dispatch-provenance",
        "library",
        "cuda-proof",
        "cudart",
    ):
        run.add_argument("--" + name, type=Path, required=True)
    for name in (
        "artifact-sha256",
        "dispatch-sha256",
        "model-revision",
        "plan-id",
        "gpu-uuid",
    ):
        run.add_argument("--" + name, required=True)
    for name in ("layer", "batch", "artifact-epoch", "dispatch-epoch"):
        run.add_argument("--" + name, type=int, required=True)
    run.add_argument("--run-id", required=True)
    status = sub.add_parser("status", help="inspect one receipt without replay")
    status.add_argument("--run-id", required=True)
    return parser


def _status(run_id):
    target = receipt_path(run_id)
    if target.is_symlink() or not target.is_file():
        report = {
            "run_id": run_id,
            "status": "UNKNOWN_DO_NOT_REPLAY",
            "reason": "no safe terminal receipt exists; absence is not proof of non-execution",
            "replay_allowed": False,
        }
    else:
        try:
            report = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            report = {"status": "UNKNOWN_DO_NOT_REPLAY", "receipt_error": str(exc)}
        report["reconciled_state"] = classify_receipt(report)
        report["replay_allowed"] = False
    print(json.dumps(report, sort_keys=True))
    return 0 if classify_receipt(report) in TERMINAL_STATUSES else 3


def main(argv=None):
    from scripts.deepseek_v41_qpn2_layer_canary import validate_run_id
    from scripts.hardware_canary import Receipt

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        validate_run_id(args.run_id)
        if args.command == "status":
            return _status(args.run_id)
        if args.model_revision != REVISION:
            raise ValueError("only the pinned DeepSeek checkpoint revision is admitted")
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in (args.artifact_sha256, args.dispatch_sha256)
        ):
            raise ValueError(
                "artifact and dispatch SHA-256 values must be lowercase hex"
            )
        report = {
            "schema_version": 1,
            "run_id": args.run_id,
            "operation_id": args.run_id,
            "phase": "A133_DEEPSEEK_QPN2_TOPK6_LAYER_CANARY",
            "status": "VALIDATING_NO_GPU",
            "resident_artifact_sha256_expected": args.artifact_sha256,
            "rank_dispatch_sha256_expected": args.dispatch_sha256,
            "model_revision": args.model_revision,
            "plan_id": args.plan_id,
            "layer": args.layer,
            "batch": args.batch,
            "artifact_epoch": args.artifact_epoch,
            "dispatch_epoch": args.dispatch_epoch,
            "expected_gpu_uuid": args.gpu_uuid,
            "cuda_executed": False,
            "cuda_context_initialized": False,
            "kernel_launch_count": 0,
            "automatic_retry": False,
            "replay_allowed": False,
        }
        try:
            receipt = Receipt(receipt_path(args.run_id), report)
        except FileExistsError:
            prior_path = receipt_path(args.run_id)
            try:
                prior = json.loads(prior_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                prior = {"status": "UNKNOWN_DO_NOT_REPLAY", "receipt_error": str(exc)}
            prior["refusal"] = "A133 run ID already has a receipt; no replay"
            prior["reconciled_state"] = classify_receipt(prior)
            prior["replay_allowed"] = False
            print(json.dumps(prior, sort_keys=True))
            return 3
        try:
            return _run_canary(args, report, receipt)
        except BaseException as exc:
            report.update(
                status="FAILED_BEFORE_GPU",
                error=f"{type(exc).__name__}: {exc}",
                cuda_executed=False,
                kernel_launch_count=0,
                replay_allowed=False,
            )
            receipt.update(report)
            print(json.dumps(report, sort_keys=True))
            return 2
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
