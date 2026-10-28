"""Bounded full-expert H1 projections and a temporary dense CPU oracle."""

from dataclasses import dataclass
from array import array
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import uuid

import numpy as np
import torch

from scripts.hardware_canary import rust_call
from scripts.resident_artifact import load_stack

REVISION = "1355db6a052410cfd62085d94b58866fd0f2c3c5"
SAMPLE_SHA = "7ede48f5bb49caa171eaf03c691b8d78d11e8644c465e4d008f8bcedf52c1c3a"

# Linux asm-generic fcntl.h values; CPython may omit these constants.
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
_F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
_F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x0004)
_F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
NAMES = ("codes13", "scales13", "gscales13", "codes2", "scales2", "gscales2")
META = dict(
    model_revision=REVISION,
    layer=0,
    rank=0,
    epoch=1,
    payload_bytes=1769856,
    physical_experts=[dict(id=0, local=900)],
    resident_order=[900],
    shape=[1, 1024, 2048, 2048, 512],
)
POLICY = dict(atol=0.0, rtol=0.0)
PROBES = {
    2048: ((0, 0.5), (17, -0.5), (513, 0.5), (1025, -0.5), (1537, 0.5), (2047, -0.5)),
    512: ((0, 0.5), (17, -0.5), (129, 0.5), (257, -0.5), (385, 0.5), (511, -0.5)),
}


def _half(value):
    try:
        result = struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError as exc:
        raise ValueError(
            "nonfinite half intermediate in strict full-expert reference"
        ) from exc
    if not math.isfinite(result):
        raise ValueError("nonfinite half intermediate in strict full-expert reference")
    return result


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def staged_weight(nibble, scale_byte, global_factor):
    """Exact scalar model of CUDA half conversions/multiplies, before MMA."""
    if type(nibble) is not int or not 0 <= nibble < 16:
        raise ValueError("FP4 nibble outside domain")
    if type(scale_byte) is not int or not 0 <= scale_byte < 127:
        raise ValueError("unsigned finite E4M3 scale required")
    if not math.isfinite(global_factor) or global_factor <= 0:
        raise ValueError("positive finite global factor required")
    # E4M3 conversion uses its bit embedding in half followed by *256.
    embedded = struct.unpack("<e", struct.pack("<H", scale_byte << 7))[0]
    scale = _half(embedded * 256.0)
    gm = _half(_f32(_f32(global_factor) * 16384.0))
    combined = _half(scale * gm)
    biased = struct.unpack(
        "<e", struct.pack("<H", ((nibble & 7) << 9) | ((nibble & 8) << 12))
    )[0]
    return _half(biased * combined)


def _sample(path):
    """Read the pinned compressed file with bounded regular-file access."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 8 <= before.st_size <= 2 * 1024 * 1024
        ):
            raise ValueError("full-expert sample must be a bounded regular file")
        chunks, size = [], 0
        while True:
            chunk = os.read(fd, min(65536, 2 * 1024 * 1024 + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > 2 * 1024 * 1024:
                raise ValueError("full-expert sample grew beyond cap")
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("full-expert sample changed while reading")
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    sample_sha256 = hashlib.sha256(raw).hexdigest()
    if sample_sha256 != SAMPLE_SHA:
        raise ValueError("full-expert sample differs from pinned content")
    header_size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_size])
    return raw, sample_sha256, header, memoryview(raw)[8 + header_size :]


def _references(header, data):
    """Native row-major nibble/scales only at probe columns; no dense plane."""
    output = {}
    for plane, n, k in (
        ("gate_proj", 512, 2048),
        ("up_proj", 512, 2048),
        ("down_proj", 2048, 512),
    ):
        prefix = "model.language_model.layers.0.mlp.experts.0." + plane + "."

        def span(suffix):
            start, end = header[prefix + suffix]["data_offsets"]
            return data[start:end]

        codes, scales = span("weight"), span("weight_scale")
        factor = struct.unpack("<f", span("weight_scale_2"))[0]
        values, subnormal_weights, subnormal_outputs, subnormal_fp4_inputs = [], 0, 0, 0
        for column, amplitude in PROBES[k]:
            row_values = []
            for row in range(n):
                byte = codes[row * (k // 2) + column // 2]
                nibble = (byte >> (4 * (column % 2))) & 15
                # The biased decoder operand 0.5/16384 is itself subnormal,
                # even when its eventual weight and output are normal.
                subnormal_fp4_inputs += (nibble & 7) == 1
                weight = staged_weight(
                    nibble, scales[row * (k // 16) + column // 16], factor
                )
                value = _half(_f32(weight * amplitude))
                subnormal_weights += 0 < abs(weight) < 2**-14
                subnormal_outputs += 0 < abs(value) < 2**-14
                row_values.append(value)
            values.append(row_values)
        output[plane] = dict(
            n=n,
            k=k,
            probes=PROBES[k],
            values=values,
            subnormal_weights=subnormal_weights,
            subnormal_outputs=subnormal_outputs,
            subnormal_fp4_inputs=subnormal_fp4_inputs,
        )
    return output


def references(path):
    """Read the pinned source once and build the strict sparse CPU oracle."""
    _, _, header, data = _sample(path)
    return _references(header, data)


def dense_projection_reference(path, plane, activations):
    """CPU FP64 dot oracle over exact staged FP16 checkpoint weights.

    The dense matrix exists only inside this bounded CPU reference call; the
    serving owner continues to hold the compressed codes and scales.
    """
    dimensions = {
        "gate_proj": (512, 2048),
        "up_proj": (512, 2048),
        "down_proj": (2048, 512),
    }
    if plane not in dimensions:
        raise ValueError("unknown full-expert projection")
    n, k = dimensions[plane]
    raw, _, header, data = _sample(path)
    prefix = "model.language_model.layers.0.mlp.experts.0." + plane + "."

    def span(suffix):
        start, end = header[prefix + suffix]["data_offsets"]
        return data[start:end]

    code_bytes = span("weight")
    scale_bytes = span("weight_scale")
    factor = struct.unpack("<f", span("weight_scale_2"))[0]
    if len(code_bytes) != n * k // 2 or len(scale_bytes) != n * k // 16:
        raise ValueError("full-expert projection byte lengths differ")
    lookup = [
        [staged_weight(nibble, scale_byte, factor) for nibble in range(16)]
        for scale_byte in range(127)
    ]
    dense_half = array("d")
    for row in range(n):
        code_start = row * (k // 2)
        scale_start = row * (k // 16)
        for column in range(k):
            byte = code_bytes[code_start + column // 2]
            nibble = (byte >> (4 * (column % 2))) & 15
            scale_byte = scale_bytes[scale_start + column // 16]
            dense_half.append(lookup[scale_byte][nibble])
    weights = torch.frombuffer(dense_half, dtype=torch.float64).reshape(n, k)
    inputs = torch.as_tensor(activations).detach().to(device="cpu", dtype=torch.float64)
    if (
        inputs.ndim != 2
        or inputs.shape[1] != k
        or not bool(torch.isfinite(inputs).all())
    ):
        raise ValueError("dense CPU oracle activations have invalid shape or values")
    output = inputs @ weights.transpose(0, 1)
    result = output.to(torch.float16)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("dense CPU oracle output is nonfinite after FP16 cast")
    del raw, weights, dense_half, output
    return result


def _digest(artifact):
    if artifact.metadata != META:
        raise ValueError("full-expert static identity differs")
    digest = hashlib.sha256(
        json.dumps(artifact.metadata, sort_keys=True, allow_nan=False).encode()
    )
    e, n13, k13, n2, k2 = META["shape"]
    shapes = dict(
        codes13=(e * n13 * k13 // 2,),
        scales13=(e * n13 * k13 // 16,),
        codes2=(e * n2 * k2 // 2,),
        scales2=(e * n2 * k2 // 16,),
        gscales13=(e, n13 // 32),
        gscales2=(e, n2 // 32),
    )
    if set(artifact.tensors) != set(NAMES):
        raise ValueError("full-expert tensor set differs")
    for name in NAMES:
        tensor = artifact.tensors[name]
        dtype = torch.float32 if name.startswith("gscales") else torch.uint8
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shapes[name]
            or not tensor.is_contiguous()
        ):
            raise ValueError("full-expert CPU tensor dtype/shape differs")
        digest.update(tensor.view(torch.uint8).reshape(-1).numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedFullExpert:
    sample: Path
    path: Path
    artifact: object
    digest: str
    artifact_sha256: str
    sample_sha256: str


def _sealed_checkpoint_memfd(raw):
    """Create an anonymous, sealed source descriptor for the Rust exporter."""
    if not (
        hasattr(os, "memfd_create")
        and hasattr(os, "MFD_CLOEXEC")
        and hasattr(os, "MFD_ALLOW_SEALING")
    ):
        raise RuntimeError(
            "sealed Linux memfd support is required for full-expert export"
        )

    required_seals = _F_SEAL_WRITE | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_SEAL
    fd = os.memfd_create(
        "voltafabric-full-expert-source",
        flags=os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise OSError("short write creating full-expert memfd")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o400)
        fcntl.fcntl(fd, _F_ADD_SEALS, required_seals)
        seals = fcntl.fcntl(fd, _F_GET_SEALS)
        if seals & required_seals != required_seals:
            raise RuntimeError("full-expert memfd did not retain all required seals")

        if os.fstat(fd).st_size != len(raw):
            raise ValueError("full-expert memfd size differs from pinned bytes")
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        if total != len(raw) or digest.digest() != hashlib.sha256(raw).digest():
            raise ValueError("full-expert memfd differs from pinned bytes")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd)
        raise


def prepare_full_expert(binary, sample, directory):
    sample = Path(sample)
    raw, sample_sha256, header, data = _sample(sample)
    _references(header, data)  # Oracle and Rust exporter use these exact pinned bytes.
    path = Path(directory) / ("full-expert-" + uuid.uuid4().hex + ".safetensors")
    source_fd = None
    try:
        source_fd = _sealed_checkpoint_memfd(raw)
        source_request = dict(
            path=f"/proc/self/fd/{source_fd}",
            spec=dict(
                model_revision=REVISION,
                layer=0,
                rank=0,
                registry=dict(epoch=1, copies=[dict(expert=0, rank=0, local=900)]),
                resident_order=[900],
                experts=[dict(id=0, local=900, source_expert=0)],
                expected_metadata=dict(
                    model="nvidia/Qwen3.6-35B-A3B-NVFP4",
                    revision=REVISION,
                    license="Apache-2.0",
                    layer="0",
                    expert="0",
                ),
            ),
        )
        receipt = rust_call(
            binary,
            "resident-export",
            dict(
                source=dict(kind="checkpoint", request=source_request),
                output=str(path),
            ),
            pass_fds=(source_fd,),
        )
    finally:
        if source_fd is not None:
            os.close(source_fd)
    if (
        receipt["cuda_executed"] is not False
        or receipt["scope"] != "CPU_COMPRESSED_EXPORT_ONLY"
        or receipt["meta"] != META
    ):
        raise ValueError("unexpected full-expert export identity/scope")
    artifact_sha256 = _file_sha256(path)
    artifact = load_stack(path, expected=META, expected_sha256=artifact_sha256)
    return PreparedFullExpert(
        sample,
        path,
        artifact,
        _digest(artifact),
        artifact_sha256,
        sample_sha256,
    )


def execute_full_expert(provider, prepared, cases):
    """Three actual provider calls on reserved GPU, or explicitly injected test oracle."""
    from scripts.hardware_canary import align_reference, compare

    reference = references(prepared.sample)
    fresh = load_stack(prepared.path, expected=META)
    if (
        _digest(prepared.artifact) != prepared.digest
        or _digest(fresh) != prepared.digest
    ):
        raise ValueError("full-expert compressed snapshot changed")
    t, device = provider.torch, provider.device
    # Six compressed resident buffers copied once; no dense persistent weights.
    tensors = {
        name: value.to(device).contiguous() for name, value in fresh.tensors.items()
    }
    for name, start, end in (
        ("gate_proj", 0, 512),
        ("up_proj", 512, 1024),
        ("down_proj", 0, 2048),
    ):
        spec = reference[name]
        n, k = spec["n"], spec["k"]
        suffix = "2" if name == "down_proj" else "13"
        codes = tensors["codes" + suffix][start * k // 2 : end * k // 2]
        scales = tensors["scales" + suffix][start * k // 16 : end * k // 16]
        globals_ = tensors["gscales" + suffix][:, start // 32 : end // 32].contiguous()
        x = t.zeros((6, k), dtype=t.float16, device=device)
        for row, (column, amplitude) in enumerate(spec["probes"]):
            x[row, column] = amplitude
        ids = t.zeros((6, 1), dtype=t.int64, device=device)
        slots, blocks, padded = align_reference(ids, 8, 1)
        y = provider.gemm_qpn2_moe(
            x, codes, scales, globals_, blocks, slots, padded, 6, n, k, 1, 4, 1
        )
        if y.dtype != t.float16 or y.device != device or tuple(y.shape) != (6, n):
            raise ValueError("full-expert provider output differs from prepared shape")
        provider.synchronize()
        result = compare(
            y.detach().cpu().tolist(),
            spec["values"],
            POLICY,
            synchronized=provider.synchronized_launches == provider.launches,
        )
        cases.append(
            dict(
                name="real_full_expert_projection",
                plane=name,
                shape=[n, k],
                probes=spec["probes"],
                oracle="STRICT_CPU_HALF_STAGE_REFERENCE",
                subnormal_weights=spec["subnormal_weights"],
                subnormal_outputs=spec["subnormal_outputs"],
                subnormal_fp4_inputs=spec["subnormal_fp4_inputs"],
                gpu_subnormal_status="CPU_ORACLE_ONLY"
                if device.type == "cpu"
                else "MATCHED_FOR_PROBES_ONLY",
                full_layer_inference=False,
                **result,
            )
        )
    return cases


def execute_full_expert_adapter(provider, prepared, cases):
    """Run two distinct EXEC batches through one compressed Skinny owner.

    This exercises the persistent local-owner boundary used by H1. It does not
    run the Rust session transport or prove CUDA unless the supplied provider is
    a hardware-gated CUDA provider.
    """
    from adapters.persistent_gpu_owner import PersistentQPN2Owner
    from scripts.hardware_canary import align_reference, compare

    if _digest(prepared.artifact) != prepared.digest:
        raise ValueError("full-expert prepared compressed snapshot changed")
    t, device = provider.torch, provider.device
    owner = PersistentQPN2Owner.load(
        artifact_path=prepared.path,
        expected_info=META,
        expected_sha256=prepared.artifact_sha256,
        provider=provider,
        device=device,
    )
    if owner.artifact_sha256 != prepared.artifact_sha256:
        raise ValueError("persistent owner published a different artifact digest")

    _, n13, k13, n2, _ = META["shape"]
    inputs = t.zeros((6, k13), dtype=t.float16, device=device)
    for row, (column, amplitude) in enumerate(
        references(prepared.sample)["gate_proj"]["probes"]
    ):
        inputs[row, column] = amplitude
    batches = (inputs, (-inputs).contiguous())
    attempts, batch_ids, batch_results = [], [], []
    first_launch = provider.launches
    first_sync = provider.synchronized_launches
    for attempt, (batch_id, activations) in enumerate(zip((6, 7), batches), start=1):
        stamp = dict(
            run_id="full-expert-persistent-owner-canary",
            model_revision=REVISION,
            plan_id="full-expert-persistent-owner-canary",
            layer=0,
            batch=batch_id,
            epoch=1,
        )
        dispatch = dict(
            stamp=stamp,
            rank=0,
            hidden=k13,
            tokens=6,
            resident_order=[900],
            slots=[
                {
                    "physical_expert": 0,
                    "slot": {
                        "token": token,
                        "slot": 0,
                        "expert": 0,
                        "rank": 0,
                        "local": 900,
                        # The owner returns unweighted rows for Rust's collector.
                        "weight": 0.25,
                    },
                }
                for token in range(6)
            ],
        )
        output = owner.execute(
            dispatch,
            token_ids=list(range(6)),
            activations=activations,
            attempt=attempt,
            align=align_reference,
        )
        expected_ids = tuple((token, 0, 0, 0, 900) for token in range(6))
        raw = output.as_raw_output_bytes(expected_ids)
        actual = t.from_numpy(
            np.frombuffer(raw["fp16_le_bytes"], dtype="<f2")
            .reshape(raw["rows"], raw["columns"])
            .astype(np.float16, copy=True)
        )

        gate = dense_projection_reference(prepared.sample, "gate_proj", activations)
        up = dense_projection_reference(prepared.sample, "up_proj", activations)
        activation_reference = [
            [
                _half(
                    _half(float(gate_value) / (1.0 + math.exp(-float(gate_value))))
                    * float(up_value)
                )
                for gate_value, up_value in zip(gate_row, up_row, strict=True)
            ]
            for gate_row, up_row in zip(gate.tolist(), up.tolist(), strict=True)
        ]
        expected = dense_projection_reference(
            prepared.sample,
            "down_proj",
            t.tensor(activation_reference, dtype=t.float16),
        )
        result = compare(
            actual.detach().cpu().tolist(),
            expected.tolist(),
            {"atol": 2.0**-22, "rtol": 2.0**-9},
            synchronized=provider.synchronized_launches == provider.launches,
        )
        attempts.append(output.attempt)
        batch_ids.append(output.stamp["batch"])
        batch_results.append(result)

    projection_calls = provider.launches - first_launch
    synchronized_projection_calls = provider.synchronized_launches - first_sync
    if projection_calls != 4 or synchronized_projection_calls != projection_calls:
        raise RuntimeError(
            "two persistent EXEC batches did not synchronize four projections"
        )
    owner.close()
    worst = max(batch_results, key=lambda result: result["max_abs_error"])
    cases.append(
        dict(
            name="real_full_expert_persistent_owner",
            shape=[6, n2],
            projections=["fused_gate_up", "down"],
            compressed_owner_path=True,
            resident_artifact_sha256=prepared.artifact_sha256,
            resident_payload_bytes=META["payload_bytes"],
            resident_dispatch_bound=True,
            artifact_loads=1,
            exec_batches=2,
            attempts=attempts,
            batch_ids=batch_ids,
            exec_has_artifact_payload=False,
            routing_coefficients_applied=False,
            projection_calls=projection_calls,
            synchronized_projection_calls=synchronized_projection_calls,
            oracle="FP64_DENSE_DOT_AFTER_EXACT_HALF_WEIGHT_STAGES",
            full_layer_inference=False,
            batch_results=batch_results,
            **worst,
        )
    )
    return cases
