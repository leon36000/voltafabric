# SPDX-License-Identifier: Apache-2.0
"""Single-rank, load-once owner around the existing Skinny QPN2 adapter.

This is a local execution component, not a network protocol or a replacement for
the Rust session worker. The caller owns the authenticated/session boundary. The
owner keeps one validated compressed resident, accepts compact RankDispatch
EXECs without weight bytes, synchronizes before returning FP16 rows, and leaves
routing coefficients to the Rust collector.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import struct
from threading import RLock
from types import SimpleNamespace
from typing import Any

import torch

from adapters import skinny_moe
from scripts.resident_artifact import _validate_info


_STAMP_KEYS = {"run_id", "model_revision", "plan_id", "layer", "batch", "epoch"}
_DISPATCH_KEYS = {"stamp", "rank", "hidden", "tokens", "resident_order", "slots"}
_MAX_RAW_OUTPUT_VALUES = 16_777_216


class OwnerRejected(ValueError):
    """Request rejected before the QPN2 provider was entered."""

    started = False


class OwnerUnknown(RuntimeError):
    """Provider was entered but execution or synchronization did not reconcile."""

    started = True


@dataclass(frozen=True, slots=True)
class OwnerOutput:
    """Synchronized, unweighted little-endian FP16 rows with full slot identities."""

    stamp: dict[str, Any]
    rank: int
    attempt: int
    identities: tuple[tuple[int, int, int, int, int], ...]
    rows: int
    columns: int
    fp16_le_bytes: bytes

    @property
    def fp16_bits(self):
        """Compatibility view for CPU oracles; hot paths should keep the byte body."""
        if (
            type(self.fp16_le_bytes) is not bytes
            or len(self.fp16_le_bytes) % 2
            or len(self.fp16_le_bytes) // 2 > _MAX_RAW_OUTPUT_VALUES
        ):
            raise OwnerRejected("owner FP16 byte payload is malformed")
        count = len(self.fp16_le_bytes) // 2
        return struct.unpack(f"<{count}H", self.fp16_le_bytes)

    def as_raw_output_bytes(self, expected_identities):
        """Return the exact Rust FP16 body, reordering rows by complete identity.

        This avoids materializing one Python integer per activation on the owner
        hot path. The consumer must frame these bytes with the returned metadata.
        """
        if type(expected_identities) not in (list, tuple):
            raise OwnerRejected(
                "expected completion identities must be a list or tuple"
            )
        expected = tuple(
            tuple(row) if type(row) in (list, tuple) else row
            for row in expected_identities
        )
        if any(
            type(identity) is not tuple
            or len(identity) != 5
            or any(type(value) is not int for value in identity)
            or not 0 <= identity[0] < 2**64
            or not 0 <= identity[1] < 64
            or not 0 <= identity[2] < 2**32
            or not 0 <= identity[3] < 2**32
            or not 0 <= identity[4] < 2**32
            for identity in expected
        ):
            raise OwnerRejected("expected completion identity is malformed")
        if (
            type(self.identities) is not tuple
            or any(
                type(identity) is not tuple
                or len(identity) != 5
                or any(type(value) is not int for value in identity)
                for identity in self.identities
            )
            or type(self.fp16_le_bytes) is not bytes
        ):
            raise OwnerRejected("owner completion rows are malformed")
        if (
            type(self.rows) is not int
            or type(self.columns) is not int
            or self.rows < 0
            or self.rows > 4096
            or self.columns <= 0
            or self.columns > 65_536
            or self.rows * self.columns > _MAX_RAW_OUTPUT_VALUES
            or self.rows != len(self.identities)
            or len(self.fp16_le_bytes) != self.rows * self.columns * 2
            or len(set(expected)) != len(expected)
            or len(set(self.identities)) != len(self.identities)
            or set(expected) != set(self.identities)
            or type(self.rank) is not int
            or not 0 <= self.rank < 2**32
            or any(identity[3] != self.rank for identity in expected)
            or type(self.attempt) is not int
            or not 0 < self.attempt < 2**64
        ):
            raise OwnerRejected(
                "owner rows do not exactly cover the Rust completion plan"
            )
        # IEEE binary16 exponent 0x1f encodes Inf/NaN. NumPy checks the wire bytes
        # without creating Python integers or copying the payload.
        import numpy as np

        exponent_bytes = np.frombuffer(self.fp16_le_bytes, dtype=np.uint8)[1::2]
        if bool(np.any((exponent_bytes & 0x7C) == 0x7C)):
            raise OwnerRejected("owner completion contains non-finite FP16 bits")
        row_bytes = self.columns * 2
        row_by_identity = {
            identity: row for row, identity in enumerate(self.identities)
        }
        if expected == self.identities:
            ordered_bytes = self.fp16_le_bytes
        else:
            ordered_bytes = b"".join(
                self.fp16_le_bytes[
                    row_by_identity[identity] * row_bytes : (
                        row_by_identity[identity] + 1
                    )
                    * row_bytes
                ]
                for identity in expected
            )
        return {
            "stamp": deepcopy(self.stamp),
            "rank": self.rank,
            "attempt": self.attempt,
            "rows": self.rows,
            "columns": self.columns,
            "fp16_le_bytes": ordered_bytes,
        }

    def as_raw_output(self, expected_identities):
        """Order rows for Rust's identity-implicit completion packet.

        RankDispatch and Rust completion order are not assumed to match. The
        caller must pass the Rust plan's full `(token, slot, expert, rank,
        local)` identity sequence; this method rejects missing, duplicate or
        foreign rows before constructing the strict RawOutput shape.
        """
        if type(expected_identities) not in (list, tuple):
            raise OwnerRejected(
                "expected completion identities must be a list or tuple"
            )
        expected = tuple(
            tuple(row) if type(row) in (list, tuple) else row
            for row in expected_identities
        )
        if any(
            type(identity) is not tuple
            or len(identity) != 5
            or any(type(value) is not int for value in identity)
            or not 0 <= identity[0] < 2**64
            or not 0 <= identity[1] < 64
            or not 0 <= identity[2] < 2**32
            or not 0 <= identity[3] < 2**32
            or not 0 <= identity[4] < 2**32
            for identity in expected
        ):
            raise OwnerRejected("expected completion identity is malformed")
        packed = self.as_raw_output_bytes(expected)
        return {
            "stamp": packed["stamp"],
            "rank": packed["rank"],
            "rows": packed["rows"],
            "columns": packed["columns"],
            "fp16_bits": struct.unpack(
                f"<{self.rows * self.columns}H", packed["fp16_le_bytes"]
            ),
        }


class _TrackedKernel:
    def __init__(self, provider):
        self.provider = provider
        self.calls = 0

    def gemm_qpn2_moe(self, *args, **kwargs):
        # Set before entering the opaque provider. An exception here does not
        # prove that the CUDA operation was not accepted.
        self.calls += 1
        return self.provider.gemm_qpn2_moe(*args, **kwargs)


class PersistentQPN2Owner:
    """Own one compressed QPN2 artifact and execute serial rank-local batches.

    The production caller should provide ``StrictQPN2Provider``. A CPU oracle
    can be injected for contract tests. This object deliberately has no EXEC
    parameter for an artifact, so callers cannot accidentally resend/reload
    weights on each batch. The caller must exclusively transfer provider and
    private resident-tensor ownership to this object for its lifetime.
    """

    def __init__(self, *, layer, resident, provider, info, digest, device):
        self._layer = layer
        self._resident = resident
        self._provider = provider
        self._device = torch.device(device)
        self._digest = digest
        self._revision = info["model_revision"]
        self._layer_id = info["layer"]
        self._rank = info["rank"]
        self._artifact_epoch = info["epoch"]
        self._epoch = info["epoch"]
        self._order = tuple(info["resident_order"])
        self._shape = tuple(info["shape"])
        self._last_attempt = 0
        self._state = "READY"
        self._lock = RLock()

    @classmethod
    def load(
        cls,
        *,
        artifact_path,
        expected_info,
        expected_sha256,
        provider,
        expert_map=None,
        device=None,
    ):
        """Validate and install compressed planes once, then publish the owner."""
        try:
            info, _ = _validate_info(expected_info)
        except (TypeError, ValueError) as exc:
            raise OwnerRejected(f"invalid resident identity: {exc}") from exc
        if (
            type(expected_sha256) is not str
            or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
        ):
            raise OwnerRejected("resident SHA-256 must be 64 lowercase hex digits")
        if not callable(getattr(provider, "gemm_qpn2_moe", None)) or not callable(
            getattr(provider, "synchronize", None)
        ):
            raise OwnerRejected("provider must expose QPN2 execution and synchronize")
        if any(
            type(getattr(provider, name, None)) is not int
            or getattr(provider, name) < 0
            for name in ("admissions", "launches", "synchronized_launches")
        ):
            raise OwnerRejected(
                "provider must expose nonnegative admission, launch and sync counts"
            )
        if not (
            provider.admissions == provider.launches == provider.synchronized_launches
        ):
            raise OwnerRejected("provider has unreconciled work before artifact load")

        target = torch.device(
            device if device is not None else getattr(provider, "device", "cpu")
        )
        provider_device = getattr(provider, "device", target)
        if torch.device(provider_device) != target:
            raise OwnerRejected("provider device differs from owner device")
        mapping = None
        global_ids = [entry["id"] for entry in info["physical_experts"]]
        if expert_map is None:
            if global_ids != list(range(len(global_ids))):
                raise OwnerRejected(
                    "sparse global expert IDs require an explicit expert map"
                )
        else:
            if (
                not isinstance(expert_map, torch.Tensor)
                or expert_map.ndim != 1
                or expert_map.dtype not in (torch.int32, torch.int64)
            ):
                raise OwnerRejected(
                    "expert map must be a one-dimensional integer tensor"
                )
            try:
                mapping = expert_map.detach().to(device=target, copy=True).contiguous()
                host_mapping = mapping.detach().cpu().tolist()
            except Exception as exc:
                raise OwnerRejected(
                    f"cannot take ownership of expert map: {exc}"
                ) from exc
            physical_count = len(info["physical_experts"])
            if any(value < -1 or value >= physical_count for value in host_mapping):
                raise OwnerRejected(
                    "expert map contains an out-of-domain physical index"
                )
            if len({value for value in host_mapping if value >= 0}) != sum(
                value >= 0 for value in host_mapping
            ):
                raise OwnerRejected("expert map aliases multiple global experts")
            if any(
                expert >= len(host_mapping) or host_mapping[expert] != physical
                for physical, expert in enumerate(global_ids)
            ):
                raise OwnerRejected(
                    "expert map differs from resident global identities"
                )

        layer = SimpleNamespace(
            skinny_moe_shape=tuple(info["shape"]),
            expert_map=mapping,
            apply_router_weight_on_input=False,
        )
        try:
            resident = skinny_moe.load_resident_artifact(
                layer,
                artifact_path,
                expected_info=info,
                expected_sha256=expected_sha256,
                device=target,
            )
        except Exception as exc:
            raise OwnerRejected(
                f"compressed resident installation failed: {exc}"
            ) from exc

        binding = getattr(resident, "artifact_binding", None)
        expected_binding = (
            info["model_revision"],
            info["layer"],
            info["rank"],
            info["epoch"],
            tuple(info["resident_order"]),
        )
        if (
            getattr(layer, "_skinny_moe_resident", None) is not resident
            or binding != expected_binding
            or tuple(resident.shape) != tuple(info["shape"])
        ):
            raise OwnerRejected("loader did not publish the expected resident owner")
        if any(
            tensor.device != target or not tensor.is_contiguous()
            for tensor in (
                resident._qc13,
                resident._qs13,
                resident._gs13,
                resident._qc2,
                resident._qs2,
                resident._gs2,
            )
        ):
            raise OwnerRejected("resident planes differ from the target device ABI")
        return cls(
            layer=layer,
            resident=resident,
            provider=provider,
            info=deepcopy(info),
            digest=expected_sha256,
            device=target,
        )

    @property
    def state(self):
        return self._state

    @property
    def epoch(self):
        return self._epoch

    @property
    def artifact_sha256(self):
        return self._digest

    def rebind_epoch(self, epoch):
        """Rebind immutable weights to one strictly newer session epoch."""
        with self._lock:
            if self._state != "READY":
                raise OwnerRejected(f"cannot rebind owner in {self._state} state")
            if type(epoch) is not int or not 0 <= epoch < 2**64:
                raise OwnerRejected("epoch must be an unsigned 64-bit integer")
            if epoch <= self._epoch:
                raise OwnerRejected("epoch must be newer than the installed binding")
            self._epoch = epoch

    def _validate_stamp_and_attempt(self, dispatch, attempt):
        if type(attempt) is not int or not 0 < attempt < 2**64:
            raise OwnerRejected("attempt must be a positive unsigned 64-bit integer")
        if attempt <= self._last_attempt:
            raise OwnerRejected("attempt is not newer than the previous EXEC")
        if type(dispatch) is not dict or set(dispatch) != _DISPATCH_KEYS:
            raise OwnerRejected("EXEC RankDispatch fields differ")
        current = dispatch.get("stamp")
        if type(current) is not dict or set(current) != _STAMP_KEYS:
            raise OwnerRejected("EXEC Stamp fields differ")
        if (
            current["model_revision"] != self._revision
            or current["layer"] != self._layer_id
            or current["epoch"] != self._epoch
            or dispatch.get("rank") != self._rank
            or dispatch.get("resident_order") != list(self._order)
            or dispatch.get("hidden") != self._shape[2]
        ):
            raise OwnerRejected(
                "EXEC identity differs from the current resident binding"
            )

    def execute(self, dispatch, *, token_ids, activations, attempt, align=None):
        """Execute one compact rank batch; no artifact/weight argument is accepted."""
        with self._lock:
            if self._state.startswith("POISONED_UNKNOWN"):
                raise OwnerUnknown(
                    "previous EXEC outcome is unknown; owner cannot be reused"
                )
            if self._state != "READY":
                raise OwnerRejected(f"owner is not executable in {self._state} state")
            self._validate_stamp_and_attempt(dispatch, attempt)
            token_ids_valid = type(token_ids) is list and all(
                type(token) is int for token in token_ids
            )
            if (
                not isinstance(activations, torch.Tensor)
                or activations.ndim != 2
                or activations.shape[1] != self._shape[2]
                or activations.dtype
                not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
                or not token_ids_valid
                or token_ids != sorted(set(token_ids))
            ):
                raise OwnerRejected(
                    "EXEC compact activation rows or token IDs are invalid"
                )
            self._last_attempt = attempt
            admissions_before = self._provider.admissions
            launches_before = self._provider.launches
            kernel = _TrackedKernel(self._provider)
            try:
                input_rows = activations
                if input_rows.device != self._device:
                    input_rows = input_rows.to(
                        device=self._device, copy=True
                    ).contiguous()
                identities, values = skinny_moe.moe_apply_dispatch_local(
                    self._layer,
                    input_rows,
                    dispatch,
                    dispatch["stamp"],
                    token_ids=token_ids,
                    kernel=kernel,
                    align=align,
                )
                if identities:
                    if kernel.calls != 2 or (
                        self._provider.admissions - admissions_before != 2
                    ):
                        raise RuntimeError(
                            "one EXEC must issue gate/up and down exactly once"
                        )
                    self._provider.synchronize()
                    if (
                        self._provider.launches - launches_before != 2
                        or self._provider.admissions != self._provider.launches
                        or self._provider.launches
                        != self._provider.synchronized_launches
                    ):
                        raise RuntimeError(
                            "provider did not reconcile all accepted launches"
                        )
                elif kernel.calls or self._provider.admissions != admissions_before:
                    raise RuntimeError(
                        "empty local dispatch unexpectedly entered the provider"
                    )
                if (
                    values.ndim != 2
                    or values.shape[0] != len(identities)
                    or values.shape[1] != self._shape[3]
                    or values.dtype != torch.float16
                    or values.device != self._device
                    or not bool(torch.isfinite(values).all())
                ):
                    raise RuntimeError(
                        "provider returned malformed or non-finite FP16 rows"
                    )
                host_bits = (
                    values.detach().contiguous().view(torch.int16).to("cpu").numpy()
                )
                fp16_le_bytes = host_bits.astype("<i2", copy=False).tobytes(order="C")
                return OwnerOutput(
                    stamp=deepcopy(dispatch["stamp"]),
                    rank=self._rank,
                    attempt=attempt,
                    identities=tuple(
                        tuple(int(value) for value in identity)
                        for identity in identities
                    ),
                    rows=len(identities),
                    columns=self._shape[3],
                    fp16_le_bytes=fp16_le_bytes,
                )
            except BaseException as exc:
                if self._provider.admissions != admissions_before:
                    self._state = "POISONED_UNKNOWN"
                    raise OwnerUnknown(
                        f"EXEC outcome is unknown after opaque launch admission; do not replay: {exc}"
                    ) from exc
                raise OwnerRejected(
                    f"EXEC rejected before provider admission: {exc}"
                ) from exc

    def close(self):
        """Explicitly drain provider work before releasing the resident tensors."""
        with self._lock:
            if self._state in ("CLOSED", "POISONED_UNKNOWN_CLOSED"):
                return
            was_unknown = self._state == "POISONED_UNKNOWN"
            try:
                close = getattr(self._provider, "close", None)
                if callable(close):
                    close()
                else:
                    self._provider.synchronize()
            except BaseException as exc:
                self._state = "POISONED_UNKNOWN"
                raise OwnerUnknown(
                    f"owner close could not reconcile provider state: {exc}"
                ) from exc
            self._layer._skinny_moe_resident = None
            self._layer.expert_map = None
            self._resident = None
            self._layer = None
            self._state = "POISONED_UNKNOWN_CLOSED" if was_unknown else "CLOSED"
