# SPDX-License-Identifier: Apache-2.0
# Modified by VoltaFabric: compact local slots, explicit replica ownership and initialized combine.
"""Adapted Skinny QPN2-MoE frontier for compact global/local expert routing.

Compressed codes and scales stay resident; the existing CUDA extension remains
the production provider. CPU tests explicitly inject an oracle. This adapter uses
dynamic compaction and has not been qualified for CUDA graphs or execution on a
V100. Per-tile global factors preserve distinct gate/up checkpoint factors.
"""

import math
import os
from dataclasses import dataclass

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

SKINNY_MOE_ENABLED = os.environ.get("VLLM_SKINNY_MOE", "0") == "1"

_route_counts: dict[tuple, int] = {}
_route_seen: set = set()


@dataclass(frozen=True, slots=True)
class _ResidentWeights:
    """Private handle for validated compressed weights owned by one layer.

    The layer exposes no tensor aliases through its public attributes. PyTorch
    tensor storage is still mutable to code that deliberately reaches into this
    private object, so this is an ownership boundary for the adapter contract,
    not a security boundary against arbitrary Python introspection.
    """

    _qc13: torch.Tensor
    _qs13: torch.Tensor
    _gs13: torch.Tensor
    _qc2: torch.Tensor
    _qs2: torch.Tensor
    _gs2: torch.Tensor
    shape: tuple[int, int, int, int, int]
    splitk13: int
    splitk2: int
    artifact_binding: tuple[str, int, int, int, tuple[int, ...]] | None = None


def _layer_shape(layer):
    """Resolve dimensions from the private owner when one carries the stack."""
    resident = getattr(layer, "_skinny_moe_resident", None)
    if isinstance(resident, _ResidentWeights):
        return resident.shape
    shape = getattr(layer, "skinny_moe_shape", None)
    if shape is None:
        raise ValueError("layer has no installed Skinny resident shape")
    return tuple(shape)


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservatively detect shared PyTorch storage without treating a data
    pointer as proof of immutability.
    """
    alias_check = getattr(torch._C, "_is_alias_of", None)
    if alias_check is not None:
        return bool(alias_check(left, right))
    # Older supported PyTorch builds may not expose _is_alias_of. StorageImpl
    # identity is used only to decide whether to snapshot; no lifetime or
    # immutability claim is derived from an address.
    return left.untyped_storage()._cdata == right.untyped_storage()._cdata


def _owned_packed_plane(tensor, source_tensors, already_owned):
    result = tensor.detach().contiguous()
    if any(_shares_storage(result, source) for source in source_tensors):
        result = result.clone()
    if any(_shares_storage(result, other) for other in already_owned):
        result = result.clone()
    return result


def _validate_packed_projection(codes, scales, gscales, *, e, n, k, device):
    _validate_packed_projection_layout(
        codes, scales, gscales, e=e, n=n, k=k, device=device
    )
    _validate_sm70_scale_domain(scales, gscales, k)


def _validate_packed_projection_layout(codes, scales, gscales, *, e, n, k, device):
    """Check copied-plane ABI and ownership without rescanning numerical data."""
    if codes.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("packed QPN2 codes and scales must use the uint8 ABI")
    if gscales.dtype != torch.float32:
        raise ValueError("packed QPN2 global factors must use float32")
    if (
        codes.numel() != e * n * k // 2
        or scales.numel() != e * n * k // 16
        or tuple(gscales.shape) != (e, n // 32)
    ):
        raise ValueError("packed QPN2 plane lengths differ from resident shape")
    if any(
        tensor.device != device or not tensor.is_contiguous()
        for tensor in (codes, scales, gscales)
    ):
        raise ValueError("packed QPN2 planes must be contiguous on one device")


def _ext():
    """Resolve Skinny and reject a module without the required MoE entrypoint."""
    from vllm.model_executor.kernels.linear.nvfp4.marlin import (
        _get_skinny_ext,
    )

    extension = _get_skinny_ext()
    if not callable(getattr(extension, "gemm_qpn2_moe", None)):
        raise RuntimeError(
            "Skinny extension does not export callable gemm_qpn2_moe; "
            "a compatible quantized MoE provider is required"
        )
    return extension


def _validate_sm70_scale_domain(
    scales,
    gscales,
    k,
    *,
    invalid_scale_message="SM70 group scales must be unsigned finite E4M3",
):
    """Admission for the existing kernel's two rounded FP16 scale steps.

    QPN2 permutations stay within a 32-row tile, so each tile maximum is valid
    for both checkpoint row order and resident fragment order. No dense weights
    or full-sized floating scale plane is allocated. This is not a GPU oracle.
    """
    if (
        not 64 <= k <= 65536
        or k % 64
        or scales.dtype != torch.uint8
        or gscales.dtype != torch.float32
    ):
        raise ValueError("SM70 scale domain dimensions or dtypes differ")
    tile_bytes = (k // 16) * 32
    if scales.numel() != gscales.numel() * tile_bytes:
        raise ValueError("SM70 scale domain plane lengths differ")
    if scales.numel() == 0:
        return
    maxima = scales.reshape(-1, tile_bytes).amax(dim=1)
    factors = gscales.reshape(-1)
    exponent = torch.div(maxima.int(), 8, rounding_mode="floor")
    mantissa = (maxima % 8).float()
    scale = torch.where(
        exponent == 0,
        mantissa * 2**-9,
        torch.ldexp(1.0 + mantissa / 8.0, exponent - 7),
    )
    gm = (factors * 16384.0).to(torch.float16).float()
    # Compute all three flags on the device, then preserve error precedence
    # with one compact host extraction. Callers co-locate both input planes.
    # RNE overflows at the halfway point 65520, including the tie to infinity.
    invalid_scale, invalid_factor, overflow = _collect_any_flags(
        maxima >= 0x7F,
        ~torch.isfinite(factors) | (factors <= 0),
        ~torch.isfinite(gm) | (scale * gm >= 65520.0),
    )
    if invalid_scale:
        raise ValueError(invalid_scale_message)
    if invalid_factor:
        raise ValueError("SM70 global factors must be positive finite")
    if overflow:
        raise ValueError("SM70 half-scale intermediate overflow")


def _prepack_stacked(codes: torch.Tensor, scales: torch.Tensor, prepack=None):
    """Prepack [E, N, K/2] u8 codes + [E, N, K/16] fp8-as-u8 scales for all
    experts in one _qpn_prepack call: tiles never straddle expert boundaries
    (N % 32 == 0), so viewing the stack as one [E*N, K/2] matrix yields an
    expert-major fragment-order layout directly."""
    if prepack is None:
        from vllm.model_executor.kernels.linear.nvfp4.marlin import _qpn_prepack

        prepack = _qpn_prepack
    e, n, k2 = codes.shape
    qc, qs = prepack(
        codes.reshape(e * n, k2).contiguous(),
        scales.reshape(e * n, scales.shape[-1]).contiguous(),
    )
    if qc is None or qs is None:
        raise ValueError("expert shape not tile-aligned")
    return qc.contiguous(), qs.contiguous()


def _pick_splitk(k: int, n_tiles_total: int) -> int:
    # K/16 groups must divide by splitk; prefer deeper splits for skinny
    # grids, shallower when the slot dimension already fills the machine.
    g = k // 16
    for s in (8, 4, 2):
        if g % s == 0:
            return s
    return 1


def prepack_experts(layer, *, prepack=None) -> None:
    """Replace Marlin repack: build QPN fragment-order expert blocks and a
    per-(expert, tile) gscale table, then publish one private resident owner.

    An explicitly injected prepack is for CPU verification; production uses the
    existing vLLM/Skinny exporter, with no automatic CPU fallback. Packed
    outputs are revalidated after prepack and snapshotted only if they share
    storage with mutable checkpoint or sibling output planes.
    """
    w13 = layer.w13_weight.data  # [E, N13, K/2] u8
    w13_s = layer.w13_weight_scale.data  # [E, N13, K/16] fp8
    w13_s2 = layer.w13_weight_scale_2.data  # [E, 2] f32
    w2 = layer.w2_weight.data  # [E, N2, K2/2] u8
    w2_s = layer.w2_weight_scale.data
    w2_s2 = layer.w2_weight_scale_2.data  # [E] f32 (or [E,1])

    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("checkpoint expert codes must have three dimensions")
    e, n13, k13_2 = w13.shape
    e2, n2, k2_2 = w2.shape
    k13, k2 = k13_2 * 2, k2_2 * 2
    if (
        e > 1024
        or e != e2
        or not n13
        or not n2
        or not k13
        or not k2
        or n13 % 64
        or n2 % 32
        or k13 % 64
        or k2 % 64
        or n13 // 2 != k2
        or n2 != k13
    ):
        raise ValueError(
            "checkpoint gate/up/down resident shapes differ or are not tile aligned"
        )
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise ValueError("checkpoint codes must be uint8")
    if w13_s.dtype not in (torch.uint8, torch.float8_e4m3fn) or w2_s.dtype not in (
        torch.uint8,
        torch.float8_e4m3fn,
    ):
        raise ValueError("checkpoint group scales must encode F8_E4M3")
    if tuple(w13_s.shape) != (e, n13, k13 // 16) or tuple(w2_s.shape) != (
        e,
        n2,
        k2 // 16,
    ):
        raise ValueError("checkpoint group scale shape differs from codes")
    if (
        tuple(w13_s2.shape) != (e, 2)
        or tuple(w2_s2.shape) not in ((e,), (e, 1))
        or w13_s2.dtype != torch.float32
        or w2_s2.dtype != torch.float32
    ):
        raise ValueError(
            "checkpoint global factors must be distinct F32 gate/up and down"
        )
    if any(t.device != w13.device for t in (w2, w13_s, w2_s, w13_s2, w2_s2)):
        raise ValueError("checkpoint planes must share a device")
    if any(
        bool((~torch.isfinite(t)).any()) or bool((t <= 0).any())
        for t in (w13_s2, w2_s2)
    ):
        raise ValueError("checkpoint global factor is nonfinite or nonpositive")
    e, n13, k13_2 = w13.shape
    _, n2, k2_2 = w2.shape

    tn13 = n13 // 32
    tn2 = n2 // 32
    # gate rows are [0, n13/2), up rows [n13/2, n13): per-tile gscale.
    gs13 = torch.empty(e, tn13, dtype=torch.float32, device=w13.device)
    gs13[:, : tn13 // 2] = w13_s2[:, 0:1].float()
    gs13[:, tn13 // 2 :] = w13_s2[:, 1:2].float()
    gs2 = w2_s2.reshape(e, 1).float().expand(e, tn2).contiguous()

    checkpoint_scale_error = "checkpoint group scales must be unsigned finite E4M3"
    _validate_sm70_scale_domain(
        w13_s.view(torch.uint8),
        gs13,
        k13,
        invalid_scale_message=checkpoint_scale_error,
    )
    _validate_sm70_scale_domain(
        w2_s.view(torch.uint8),
        gs2,
        k2,
        invalid_scale_message=checkpoint_scale_error,
    )
    qc13, qs13 = _prepack_stacked(w13, w13_s.view(torch.uint8), prepack)
    checkpoint_planes = (w13, w13_s, w13_s2, w2, w2_s, w2_s2)
    owned_planes = []
    qc13 = _owned_packed_plane(qc13, checkpoint_planes, owned_planes)
    owned_planes.append(qc13)
    qs13 = _owned_packed_plane(qs13, checkpoint_planes, owned_planes)
    owned_planes.append(qs13)
    _validate_packed_projection(qc13, qs13, gs13, e=e, n=n13, k=k13, device=w13.device)

    qc2, qs2 = _prepack_stacked(w2, w2_s.view(torch.uint8), prepack)
    qc2 = _owned_packed_plane(qc2, checkpoint_planes, owned_planes)
    owned_planes.append(qc2)
    qs2 = _owned_packed_plane(qs2, checkpoint_planes, owned_planes)
    owned_planes.append(qs2)
    gs13 = gs13.detach().contiguous()
    gs2 = gs2.detach().contiguous()
    _validate_packed_projection(qc2, qs2, gs2, e=e, n=n2, k=k2, device=w13.device)

    shape = (e, n13, k13_2 * 2, n2, k2_2 * 2)
    splitk13 = _pick_splitk(k13_2 * 2, e * tn13)
    splitk2 = _pick_splitk(k2_2 * 2, e * tn2)
    resident = _ResidentWeights(
        qc13, qs13, gs13, qc2, qs2, gs2, shape, splitk13, splitk2
    )
    # Remove legacy public tensor aliases if this layer was repacked in place.
    for name in (
        "skinny_moe_qc13",
        "skinny_moe_qs13",
        "skinny_moe_gs13",
        "skinny_moe_qc2",
        "skinny_moe_qs2",
        "skinny_moe_gs2",
    ):
        if hasattr(layer, name):
            delattr(layer, name)
    layer.skinny_moe_shape = shape
    layer.skinny_moe_splitk13 = splitk13
    layer.skinny_moe_splitk2 = splitk2
    layer._skinny_moe_resident = resident

    logger.info(
        "Skinny MoE prepack: E=%d w13[N=%d K=%d] w2[N=%d K=%d] "
        "splitk=(%d,%d) bytes=%.2f GiB",
        e,
        n13,
        k13_2 * 2,
        n2,
        k2_2 * 2,
        splitk13,
        splitk2,
        (qc13.numel() + qs13.numel() + qc2.numel() + qs2.numel()) / 2**30,
    )


def load_resident_artifact(
    layer, path, *, expected_info, expected_sha256, device
) -> _ResidentWeights:
    """Load a Rust QPN2 artifact once into private compressed device buffers.

    The Rust-produced Safetensors file is validated on CPU, including its complete
    identity, plane layout, and SM70 scale domain, before any device allocation.
    Validated owned CPU planes transfer to the CPU owner without another copy.
    Non-CPU targets receive separate device buffers before atomic publication.
    This does not decode or retain dense weights.
    """
    if getattr(layer, "_skinny_moe_resident", None) is not None:
        raise ValueError("layer already owns resident weights; use a fresh layer")
    if type(expected_info) is not dict:
        raise ValueError("complete expected resident artifact metadata is required")

    from scripts.resident_artifact import load_stack

    artifact = load_stack(path, expected=expected_info, expected_sha256=expected_sha256)
    info = artifact.metadata
    e, n13, k13, n2, k2 = info["shape"]
    if e <= 0:
        raise ValueError("an executable layer needs at least one resident expert")
    shape = (e, n13, k13, n2, k2)
    previous_shape = getattr(layer, "skinny_moe_shape", None)
    if previous_shape is not None and tuple(previous_shape) != shape:
        raise ValueError("artifact shape differs from the layer")
    if any(
        hasattr(layer, "skinny_moe_" + name)
        for name in ("qc13", "qs13", "gs13", "qc2", "qs2", "gs2")
    ):
        raise ValueError("layer already exposes legacy QPN2 plane aliases")

    target = torch.device(device)
    global_ids = [expert["id"] for expert in info["physical_experts"]]
    expert_map = getattr(layer, "expert_map", None)
    if expert_map is None:
        if global_ids != list(range(e)):
            raise ValueError(
                "global-to-physical expert map is required for this resident artifact"
            )
    else:
        if (
            not isinstance(expert_map, torch.Tensor)
            or expert_map.ndim != 1
            or expert_map.dtype not in (torch.int32, torch.int64)
            or expert_map.device != target
        ):
            raise ValueError(
                "global-to-physical expert map must be an integer vector on the target device"
            )
        mapping = expert_map.detach().to(device="cpu").tolist()
        owned = [physical for physical in mapping if physical >= 0]
        if (
            any(physical < -1 or physical >= e for physical in mapping)
            or len(set(owned)) != len(owned)
            or any(
                global_id >= len(mapping) or mapping[global_id] != physical
                for physical, global_id in enumerate(global_ids)
            )
        ):
            raise ValueError("global-to-physical expert map differs from the artifact")

    planes = artifact.tensors
    _validate_packed_projection_layout(
        planes["codes13"],
        planes["scales13"],
        planes["gscales13"],
        e=e,
        n=n13,
        k=k13,
        device=torch.device("cpu"),
    )
    _validate_packed_projection_layout(
        planes["codes2"],
        planes["scales2"],
        planes["gscales2"],
        e=e,
        n=n2,
        k=k2,
        device=torch.device("cpu"),
    )

    # load_stack already cloned private, contiguous CPU tensors. The artifact
    # remains local to this call, so on CPU ownership can move to the resident
    # rather than duplicating every compressed plane a second time.
    device_planes = {
        name: tensor.to(device=target, copy=target.type != "cpu").contiguous()
        for name, tensor in planes.items()
    }
    resident = _ResidentWeights(
        device_planes["codes13"],
        device_planes["scales13"],
        device_planes["gscales13"],
        device_planes["codes2"],
        device_planes["scales2"],
        device_planes["gscales2"],
        shape,
        _pick_splitk(k13, e * (n13 // 32)),
        _pick_splitk(k2, e * (n2 // 32)),
        (
            info["model_revision"],
            info["layer"],
            info["rank"],
            info["epoch"],
            tuple(info["resident_order"]),
        ),
    )
    _validate_packed_projection_layout(
        resident._qc13,
        resident._qs13,
        resident._gs13,
        e=e,
        n=n13,
        k=k13,
        device=target,
    )
    _validate_packed_projection_layout(
        resident._qc2,
        resident._qs2,
        resident._gs2,
        e=e,
        n=n2,
        k=k2,
        device=target,
    )
    # The single owner assignment is the publication point. Every validation and
    # copy above completes first, so a rejection leaves the layer unchanged.
    layer._skinny_moe_resident = resident
    return resident


def _collect_any_flags(*predicates):
    """Collect a small group of boolean failures with one host synchronization.

    Each predicate is reduced on its device first. Stacking the scalar results
    and copying that compact flag vector to the host avoids serial ``bool(tensor)``
    synchronizations while retaining the caller's individual failure reasons.
    """
    if not predicates:
        return ()
    device = predicates[0].device
    if any(
        predicate.dtype != torch.bool or predicate.device != device
        for predicate in predicates
    ):
        raise ValueError(
            "grouped guard predicates must be boolean tensors on one device"
        )
    reduced = torch.stack(tuple(predicate.any() for predicate in predicates))
    return tuple(bool(flag) for flag in reduced.detach().cpu().tolist())


def _compute_compact_qpn2(
    x,
    tokens,
    local_ids,
    *,
    resident,
    shape,
    planes,
    splitk13,
    splitk2,
    kernel,
    align,
):
    """Run the existing two-projection QPN2 path for already validated local rows."""
    e, n13, k13, n2, k2 = shape
    if tokens.numel() == 0:
        return torch.empty((0, n2), dtype=torch.float16, device=x.device)
    if align is None:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        align = moe_align_block_size
    sorted_ids, expert_blocks, ntpp = align(local_ids.reshape(-1, 1), 8, e)
    sorted_ids = sorted_ids.to(torch.int32).contiguous()
    expert_blocks = expert_blocks.to(torch.int32).contiguous()
    ntpp = ntpp.to(torch.int32).contiguous()
    if kernel is None:
        kernel = _ext()
    compact_x = x.index_select(0, tokens).to(torch.float16).contiguous()
    if bool((~torch.isfinite(compact_x)).any()):
        raise ValueError("FP16 compact input overflow")
    s = tokens.numel()
    qc13, qs13, gs13 = planes["13"]
    h1 = kernel.gemm_qpn2_moe(
        compact_x,
        qc13,
        qs13,
        gs13,
        expert_blocks,
        sorted_ids,
        ntpp,
        s,
        n13,
        k13,
        1,
        splitk13,
        1,
    )
    if (
        tuple(h1.shape) != (s, n13)
        or h1.dtype != torch.float16
        or h1.device != x.device
    ):
        raise ValueError("gate/up provider returned a malformed compact buffer")
    act = (torch.nn.functional.silu(h1[:, :k2]) * h1[:, k2:]).contiguous()
    gate_up_failures = _collect_any_flags(~torch.isfinite(h1), ~torch.isfinite(act))
    if gate_up_failures[0]:
        raise ValueError("nonfinite gate/up provider output")
    if gate_up_failures[1]:
        raise ValueError("nonfinite SwiGLU activation")
    qc2, qs2, gs2 = planes["2"]
    h2 = kernel.gemm_qpn2_moe(
        act,
        qc2,
        qs2,
        gs2,
        expert_blocks,
        sorted_ids,
        ntpp,
        s,
        n2,
        k2,
        1,
        splitk2,
        1,
    )
    if tuple(h2.shape) != (s, n2) or h2.dtype != torch.float16 or h2.device != x.device:
        raise ValueError("down provider returned a malformed compact buffer")
    if bool((~torch.isfinite(h2)).any()):
        raise ValueError("nonfinite down provider output")
    return h2


def _validate_local_routes(topk_ids, expert_map, local_experts):
    """Validate every logical route and resident mapping without compacting slots."""
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be an integer [tokens,topk] tensor")
    if local_experts < 0 or topk_ids.shape[1] == 0:
        raise ValueError("invalid local expert count or empty top-k")
    route_checks = [topk_ids < 0]
    if topk_ids.shape[1] > 1:
        ordered = topk_ids.sort(dim=1).values
        route_checks.append(ordered[:, 1:] == ordered[:, :-1])
    if expert_map is None:
        route_checks.append(topk_ids >= local_experts)
        route_failures = _collect_any_flags(*route_checks)
        if route_failures[0]:
            raise ValueError("negative global expert ID")
        if len(route_failures) > 2 and route_failures[1]:
            raise ValueError("duplicate global expert ID within token top-k")
        if route_failures[-1]:
            raise ValueError("global expert outside unpartitioned resident stack")
    else:
        route_failures = _collect_any_flags(*route_checks)
        if route_failures[0]:
            raise ValueError("negative global expert ID")
        if len(route_failures) > 1 and route_failures[1]:
            raise ValueError("duplicate global expert ID within token top-k")
        if (
            expert_map.ndim != 1
            or expert_map.dtype not in (torch.int32, torch.int64)
            or expert_map.device != topk_ids.device
        ):
            raise ValueError(
                "expert_map must be an integer global-to-physical vector on the same device"
            )
        map_failures = _collect_any_flags(
            expert_map < -1,
            expert_map >= local_experts,
            topk_ids >= expert_map.numel(),
        )
        if map_failures[0] or map_failures[1]:
            raise ValueError("expert_map physical index outside resident stack")
        if map_failures[2]:
            raise ValueError("unknown global expert ID")
        owned = expert_map[expert_map >= 0]
        if torch.unique(owned).numel() != owned.numel():
            raise ValueError("distinct global experts alias one physical stack")


def _select_local_slots(topk_ids, expert_map, local_experts, selected_mask):
    """Return original flattened slots, token rows and physical resident expert indices."""
    _validate_local_routes(topk_ids, expert_map, local_experts)
    flattened = topk_ids.reshape(-1).long()
    local = flattened if expert_map is None else expert_map[flattened].long()
    selected = local >= 0
    if selected_mask is not None:
        if (
            selected_mask.shape != topk_ids.shape
            or selected_mask.dtype != torch.bool
            or selected_mask.device != topk_ids.device
        ):
            raise ValueError("selected_mask must identify exact routed copies")
        # A selected owner without a local replica is a plan error, never silently omitted.
        if _collect_any_flags(selected_mask.reshape(-1) & ~selected)[0]:
            raise ValueError("selected slot has no local resident replica")
        selected &= selected_mask.reshape(-1)
    positions = torch.nonzero(selected, as_tuple=False).reshape(-1)
    return (
        positions,
        torch.div(positions, topk_ids.shape[1], rounding_mode="floor"),
        local[positions],
    )


def bind_rank_dispatch(layer, rank_dispatch, expected_stamp, topk_ids, topk_weights):
    """Bind one Rust RankDispatch to exact tensor slots before partitioned execution.

    The coordinator supplies the current full Stamp; an old layer/batch/epoch is
    rejected. This builds an ephemeral owner mask, never a mask cached on layer.
    """
    e, n13, k13, n2, k2 = _layer_shape(layer)
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("route tensors differ")
    if rank_dispatch.get("stamp") != expected_stamp or set(expected_stamp) != {
        "run_id",
        "model_revision",
        "plan_id",
        "layer",
        "batch",
        "epoch",
    }:
        raise ValueError("rank dispatch has stale or incomplete Stamp")
    if (
        rank_dispatch.get("hidden") != k13
        or rank_dispatch.get("tokens") != topk_ids.shape[0]
    ):
        raise ValueError("rank dispatch dimensions differ from tensor frontier")
    resident = rank_dispatch.get("resident_order")
    if not isinstance(resident, list) or len(resident) != e or len(set(resident)) != e:
        raise ValueError("physical resident order differs from stack")
    owner = getattr(layer, "_skinny_moe_resident", None)
    if isinstance(owner, _ResidentWeights) and owner.artifact_binding is not None:
        (
            model_revision,
            artifact_layer,
            artifact_rank,
            artifact_epoch,
            artifact_order,
        ) = owner.artifact_binding
        if (
            expected_stamp["model_revision"] != model_revision
            or expected_stamp["layer"] != artifact_layer
            or expected_stamp["epoch"] < artifact_epoch
            or rank_dispatch.get("rank") != artifact_rank
            or tuple(resident) != artifact_order
        ):
            raise ValueError(
                "rank dispatch artifact identity differs from installed resident"
            )
    # Validate tensor IDs/mapping before indexing the dispatch.
    _validate_local_routes(topk_ids, getattr(layer, "expert_map", None), e)
    mask = torch.zeros_like(topk_ids, dtype=torch.bool)
    seen = set()
    route_rows = []
    expected_weights = []
    for routed in rank_dispatch.get("slots", []):
        s = routed["slot"]
        physical = routed["physical_expert"]
        token, slot = s["token"], s["slot"]
        if not (
            type(token) is int
            and type(slot) is int
            and type(physical) is int
            and 0 <= token < topk_ids.shape[0]
            and 0 <= slot < topk_ids.shape[1]
            and 0 <= physical < e
        ):
            raise ValueError("rank slot index outside tensor frontier")
        if (
            (token, slot) in seen
            or s["rank"] != rank_dispatch["rank"]
            or resident[physical] != s["local"]
        ):
            raise ValueError("duplicate slot or foreign physical resident")
        if type(s["expert"]) is not int:
            raise ValueError("dispatch global/local expert differs from tensor route")
        weight = s["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("dispatch coefficient differs from tensor route")
        route_rows.append((token, slot, s["expert"], physical))
        expected_weights.append(weight)
        seen.add((token, slot))

    # Validate all route identities and coefficients with batched device
    # comparisons. Scalar ``int(tensor)``/``bool(tensor)`` in the loop above
    # synchronized the host once or more for every routed slot.
    route_table = torch.tensor(
        route_rows, dtype=torch.int64, device=topk_ids.device
    ).reshape(-1, 4)
    token_indices, slot_indices, expected_experts, expected_physical = (
        route_table.unbind(dim=1)
    )
    actual_experts = topk_ids[token_indices, slot_indices].long()
    mapping = getattr(layer, "expert_map", None)
    actual_physical = (
        actual_experts if mapping is None else mapping[actual_experts].long()
    )
    expected_coefficients = topk_weights.new_tensor(expected_weights)
    actual_coefficients = topk_weights[token_indices, slot_indices]
    failures = _collect_any_flags(
        (actual_experts != expected_experts) | (actual_physical != expected_physical),
        ~torch.isfinite(expected_coefficients)
        | (expected_coefficients < 0)
        | (expected_coefficients != actual_coefficients),
    )
    if failures[0]:
        raise ValueError("dispatch global/local expert differs from tensor route")
    if failures[1]:
        raise ValueError("dispatch coefficient differs from tensor route")
    mask[token_indices, slot_indices] = True
    return mask


def moe_apply_local(
    layer,
    x,
    topk_weights,
    topk_ids,
    shared_experts,
    shared_experts_input,
    *,
    kernel=None,
    align=None,
    selected_mask=None,
):
    """Return (original flattened slot positions, unweighted compact FP16 h2).

    These rows are suitable for the existing Rust Completion.value contract;
    the collector applies the original coefficient once. Bind the current
    RankDispatch before execution and keep its Stamp with the resulting rows.
    The real Skinny CUDA provider remains the default. CPU providers are explicit
    verification oracles. Dynamic compaction is not CUDA-graph-qualified.
    """
    resident = getattr(layer, "_skinny_moe_resident", None)
    shape = _layer_shape(layer)
    if resident is not None:
        if not isinstance(resident, _ResidentWeights) or resident.shape != shape:
            raise ValueError("private resident weight owner is malformed")
        planes = {
            "13": (resident._qc13, resident._qs13, resident._gs13),
            "2": (resident._qc2, resident._qs2, resident._gs2),
        }
        splitk13, splitk2 = resident.splitk13, resident.splitk2
    else:
        planes = {
            name: (
                getattr(layer, "skinny_moe_qc" + name),
                getattr(layer, "skinny_moe_qs" + name),
                getattr(layer, "skinny_moe_gs" + name),
            )
            for name in ("13", "2")
        }
        splitk13 = layer.skinny_moe_splitk13
        splitk2 = layer.skinny_moe_splitk2
    e, n13, k13, n2, k2 = shape
    if (
        x.ndim != 2
        or topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
        or x.shape[0] != topk_ids.shape[0]
    ):
        raise ValueError("token/route/weight shape mismatch")
    if (
        x.shape[1] != k13
        or n13 % 64
        or n2 % 32
        or k13 % 64
        or k2 % 64
        or n13 // 2 != k2
        or n2 != k13
    ):
        raise ValueError("QPN2 MoE gate/up/down shapes are not tile aligned")
    if (
        x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        or not topk_weights.is_floating_point()
    ):
        raise ValueError("floating activations and coefficients required")
    if x.device != topk_ids.device or x.device != topk_weights.device:
        raise ValueError("routes and activations must share the device")
    if any(
        _collect_any_flags(
            ~torch.isfinite(x),
            ~torch.isfinite(topk_weights),
            topk_weights < 0,
        )
    ):
        raise ValueError("nonfinite input or invalid routing coefficient")
    if getattr(layer, "apply_router_weight_on_input", False):
        raise ValueError("Skinny applies router weights on output")
    if getattr(layer, "expert_map", None) is not None and selected_mask is None:
        raise ValueError(
            "partitioned execution requires an explicit dispatch owner mask"
        )
    positions, tokens, local_ids = _select_local_slots(
        topk_ids, getattr(layer, "expert_map", None), e, selected_mask
    )
    if shared_experts is not None:
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExpertsOrder,
        )

        shared_experts.apply(
            shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
        )
    if positions.numel() == 0:
        return positions, torch.empty((0, n2), dtype=torch.float16, device=x.device)
    for name, n, k in [("13", n13, k13), ("2", n2, k2)]:
        codes, scales, gscales = planes[name]
        if (
            codes.dtype != torch.uint8
            or scales.dtype != torch.uint8
            or gscales.dtype != torch.float32
        ):
            raise ValueError("packed QPN2 plane dtypes differ from ABI")
        if (
            codes.numel() != e * n * k // 2
            or scales.numel() != e * n * k // 16
            or tuple(gscales.shape) != (e, n // 32)
        ):
            raise ValueError("packed QPN2 plane lengths differ from resident shape")
        if any(
            t.device != x.device or not t.is_contiguous()
            for t in (codes, scales, gscales)
        ):
            raise ValueError(
                "packed QPN2 planes must be contiguous on the activation device"
            )
        if resident is None:
            if bool((~torch.isfinite(gscales)).any()) or bool((gscales <= 0).any()):
                raise ValueError(
                    "packed QPN2 scale must be unsigned finite E4M3 and global factor positive"
                )
            _validate_sm70_scale_domain(scales, gscales, k)
    return positions, _compute_compact_qpn2(
        x,
        tokens,
        local_ids,
        resident=resident,
        shape=shape,
        planes=planes,
        splitk13=splitk13,
        splitk2=splitk2,
        kernel=kernel,
        align=align,
    )


def moe_apply_dispatch_local(
    layer,
    x,
    rank_dispatch,
    expected_stamp,
    *,
    token_ids,
    kernel=None,
    align=None,
):
    """Run one compact Rust RankDispatch as unweighted rows with full identities.

    `x` contains one row for each sorted token in `token_ids`; it does not contain
    placeholder routes for remote experts. The returned identities are
    `(token, slot, expert, rank, local)` tuples in dispatch order. The existing
    Rust collector retains the validated coefficients and applies them once.
    """
    required_stamp = {
        "run_id",
        "model_revision",
        "plan_id",
        "layer",
        "batch",
        "epoch",
    }
    dispatch_keys = {
        "stamp",
        "rank",
        "hidden",
        "tokens",
        "resident_order",
        "slots",
    }
    if (
        type(expected_stamp) is not dict
        or set(expected_stamp) != required_stamp
        or type(rank_dispatch) is not dict
        or set(rank_dispatch) != dispatch_keys
        or rank_dispatch.get("stamp") != expected_stamp
    ):
        raise ValueError("rank dispatch has stale or incomplete Stamp or fields")
    for name in ("run_id", "model_revision", "plan_id"):
        if (
            type(expected_stamp[name]) is not str
            or not expected_stamp[name].strip()
            or len(expected_stamp[name]) > 256
        ):
            raise ValueError("rank dispatch Stamp contains an invalid identity")
    for name in ("layer", "batch", "epoch"):
        limit = 2**32 if name == "layer" else 2**64
        if (
            type(expected_stamp[name]) is not int
            or not 0 <= expected_stamp[name] < limit
        ):
            raise ValueError("rank dispatch Stamp contains an invalid numeric identity")

    resident = getattr(layer, "_skinny_moe_resident", None)
    shape = _layer_shape(layer)
    if resident is not None and (
        not isinstance(resident, _ResidentWeights) or resident.shape != shape
    ):
        raise ValueError("private resident weight owner is malformed")
    e, n13, k13, n2, k2 = shape
    rank = rank_dispatch["rank"]
    hidden = rank_dispatch["hidden"]
    token_count = rank_dispatch["tokens"]
    order = rank_dispatch["resident_order"]
    routed_slots = rank_dispatch["slots"]
    if (
        type(rank) is not int
        or not 0 <= rank < 2**32
        or type(hidden) is not int
        or hidden != k13
        or type(token_count) is not int
        or not 0 <= token_count <= 4096
        or type(order) is not list
        or len(order) != e
        or any(type(local) is not int or not 0 <= local < 2**32 for local in order)
        or len(set(order)) != len(order)
        or type(routed_slots) is not list
        or len(routed_slots) > 262144
        or n13 % 64
        or n2 % 32
        or k13 % 64
        or k2 % 64
        or n13 // 2 != k2
        or n2 != k13
    ):
        raise ValueError("rank dispatch dimensions or resident order are invalid")
    if resident is not None and resident.artifact_binding is not None:
        revision, artifact_layer, artifact_rank, artifact_epoch, artifact_order = (
            resident.artifact_binding
        )
        if (
            expected_stamp["model_revision"] != revision
            or expected_stamp["layer"] != artifact_layer
            or expected_stamp["epoch"] < artifact_epoch
            or rank != artifact_rank
            or tuple(order) != artifact_order
        ):
            raise ValueError("dispatch identity differs from installed resident")
    if getattr(layer, "apply_router_weight_on_input", False):
        raise ValueError("Skinny applies router weights on output")

    if (
        type(token_ids) is not list
        or any(
            type(token) is not int or not 0 <= token < token_count
            for token in token_ids
        )
        or token_ids != sorted(set(token_ids))
        or x.ndim != 2
        or tuple(x.shape) != (len(token_ids), k13)
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    ):
        raise ValueError("compacted input token IDs or activation shape are invalid")
    if any(_collect_any_flags(~torch.isfinite(x))):
        raise ValueError("nonfinite compacted activation")

    token_rows = {token: row for row, token in enumerate(token_ids)}
    seen_slots = set()
    seen_experts = set()
    route_rows = []
    identities = []
    dispatch_tokens = set()
    for routed in routed_slots:
        if type(routed) is not dict or set(routed) != {"slot", "physical_expert"}:
            raise ValueError("malformed local RankDispatch entry")
        slot = routed["slot"]
        physical = routed["physical_expert"]
        if type(slot) is not dict or set(slot) != {
            "token",
            "slot",
            "expert",
            "rank",
            "local",
            "weight",
        }:
            raise ValueError("malformed local route identity")
        token, index, expert, owner, local = (
            slot["token"],
            slot["slot"],
            slot["expert"],
            slot["rank"],
            slot["local"],
        )
        weight = slot["weight"]
        weight_valid = isinstance(weight, (int, float)) and not isinstance(weight, bool)
        if weight_valid:
            try:
                weight_valid = math.isfinite(float(weight)) and weight >= 0
            except OverflowError:
                weight_valid = False
        if (
            any(
                type(value) is not int
                for value in (token, index, expert, owner, local, physical)
            )
            or not 0 <= token < token_count
            or not 0 <= index < 64
            or not 0 <= expert < 2**32
            or owner != rank
            or not 0 <= physical < e
            or order[physical] != local
            or not weight_valid
        ):
            raise ValueError("foreign, out-of-domain or invalid local route")
        if (token, index) in seen_slots or (token, expert) in seen_experts:
            raise ValueError("duplicate local slot or expert within token")
        seen_slots.add((token, index))
        seen_experts.add((token, expert))
        dispatch_tokens.add(token)
        route_rows.append((token_rows.get(token, -1), physical, expert))
        identities.append((token, index, expert, owner, local))
    if dispatch_tokens != set(token_ids):
        raise ValueError(
            "compacted input rows must cover exactly the dispatched tokens"
        )
    if not routed_slots:
        if token_ids:
            raise ValueError("empty dispatch cannot carry compacted inputs")
        return [], torch.empty((0, n2), dtype=torch.float16, device=x.device)

    expert_map = getattr(layer, "expert_map", None)
    experts = torch.tensor(
        [expert for _, _, expert in route_rows], dtype=torch.int64, device=x.device
    )
    physical_ids = torch.tensor(
        [physical for _, physical, _ in route_rows],
        dtype=torch.int64,
        device=x.device,
    )
    if expert_map is None:
        if any(expert != physical for _, physical, expert in route_rows):
            raise ValueError("global expert differs from identity resident mapping")
    else:
        if (
            not isinstance(expert_map, torch.Tensor)
            or expert_map.ndim != 1
            or expert_map.dtype not in (torch.int32, torch.int64)
            or expert_map.device != x.device
            or any(expert >= expert_map.numel() for _, _, expert in route_rows)
        ):
            raise ValueError(
                "global-to-physical expert map has the wrong ABI or domain"
            )
        if any(_collect_any_flags(expert_map[experts].long() != physical_ids)):
            raise ValueError("global expert does not map to dispatched physical expert")

    planes = (
        {
            "13": (resident._qc13, resident._qs13, resident._gs13),
            "2": (resident._qc2, resident._qs2, resident._gs2),
        }
        if resident is not None
        else {
            name: (
                getattr(layer, "skinny_moe_qc" + name),
                getattr(layer, "skinny_moe_qs" + name),
                getattr(layer, "skinny_moe_gs" + name),
            )
            for name in ("13", "2")
        }
    )
    splitk13 = resident.splitk13 if resident is not None else layer.skinny_moe_splitk13
    splitk2 = resident.splitk2 if resident is not None else layer.skinny_moe_splitk2
    for name, n, k in [("13", n13, k13), ("2", n2, k2)]:
        codes, scales, gscales = planes[name]
        if (
            codes.dtype != torch.uint8
            or scales.dtype != torch.uint8
            or gscales.dtype != torch.float32
            or codes.numel() != e * n * k // 2
            or scales.numel() != e * n * k // 16
            or tuple(gscales.shape) != (e, n // 32)
            or any(
                t.device != x.device or not t.is_contiguous()
                for t in (codes, scales, gscales)
            )
        ):
            raise ValueError("packed QPN2 planes differ from resident ABI")
        if resident is None:
            if bool((~torch.isfinite(gscales)).any()) or bool((gscales <= 0).any()):
                raise ValueError("packed QPN2 scale or global factor is invalid")
            _validate_sm70_scale_domain(scales, gscales, k)

    tokens = torch.tensor(
        [row for row, _, _ in route_rows], dtype=torch.int64, device=x.device
    )
    local_ids = physical_ids
    values = _compute_compact_qpn2(
        x,
        tokens,
        local_ids,
        resident=resident,
        shape=shape,
        planes=planes,
        splitk13=splitk13,
        splitk2=splitk2,
        kernel=kernel,
        align=align,
    )
    return identities, values


def moe_apply(
    layer,
    x,
    topk_weights,
    topk_ids,
    shared_experts,
    shared_experts_input,
    *,
    kernel=None,
    align=None,
    selected_mask=None,
):
    """Existing local weighted combine, using the same compact raw execution.

    Routing coefficients are applied here. Do not submit this combined result as
    an unweighted Completion.value to a second collector.
    """
    positions, h2 = moe_apply_local(
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
        kernel=kernel,
        align=align,
        selected_mask=selected_mask,
    )
    out = torch.zeros(
        (x.shape[0], _layer_shape(layer)[3]), dtype=torch.float32, device=x.device
    )
    if positions.numel() == 0:
        return out.to(x.dtype)
    tokens = torch.div(positions, topk_ids.shape[1], rounding_mode="floor")
    contributions = h2.float() * topk_weights.reshape(-1).index_select(
        0, positions
    ).float().reshape(-1, 1)
    out.index_add_(0, tokens, contributions)
    result = out.to(x.dtype)
    if any(_collect_any_flags(~torch.isfinite(out), ~torch.isfinite(result))):
        raise ValueError("nonfinite FP32 combine or output cast")
    return result
