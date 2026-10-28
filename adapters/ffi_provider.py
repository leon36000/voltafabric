# SPDX-License-Identifier: Apache-2.0
"""Strict SM70 provider for the existing Skinny tensor interface and C ABI.

The constructor reuses the diagnostic provider's device/UUID/toolchain gate;
its tiny-shape limits remain unchanged. This separate service reference admits
the real C ABI's bounded shapes under explicit memory budgets. It never loads
dense weights, implements a CPU fallback, or treats enqueue as completion.

Weights already installed by load_resident_artifact stay on their device. This
strict path revalidates their values every call: borrowed mutable tensors are
not immutable owners. Callers must not mutate any input/weight while pending.
All borrowed inputs, fresh device metadata and host mirrors survive until an
explicit successful synchronize/close. No destructor silently synchronizes.
"""

import ctypes as C
import threading

from scripts.hardware_canary import QPN2Provider, make_params
from adapters.skinny_moe import _validate_sm70_scale_domain


class StrictQPN2Provider(QPN2Provider):
    """Real C enqueue, explicit completion, bounded pending resources.

    A launch/completion failure poisons this instance permanently. Outstanding
    buffers can still be drained explicitly, but no execution is replayed.
    This is a correction reference, not a GPU performance qualification.

    The caller orders production of inputs/weights on the current device stream
    (or waits for their producer events there) before entering this interface.
    record_stream protects allocator reuse; it is not a producer dependency.
    max_resident_bytes bounds one borrowed projection, not total GPU residency.
    max_pending_bytes bounds retained input plus output/device metadata across
    pending calls; HOST mirrors and validation scratch are separate allocations.
    """

    def __init__(
        self,
        library,
        cudart,
        device,
        expected_uuid,
        *,
        max_resident_bytes,
        max_pending_bytes,
        ack_hardware_gate=False,
        gate_path=None,
    ):
        for budget in (max_resident_bytes, max_pending_bytes):
            if type(budget) is not int or not 0 < budget <= 2**63 - 1:
                raise ValueError("positive bounded memory budgets required")
        super().__init__(
            library,
            cudart,
            device,
            expected_uuid,
            ack_hardware_gate=ack_hardware_gate,
            gate_path=gate_path,
        )
        self.max_resident_bytes = max_resident_bytes
        self.max_pending_bytes = max_pending_bytes
        self._pending_bytes = 0
        self._failed = self._closed = False
        # Increment immediately before entering the opaque launch call. Provider
        # validation failures before this point are known not to have launched;
        # failures after it leave the device outcome unknown.
        self.admissions = 0
        self._lock = threading.RLock()

    def gemm_qpn2_moe(
        self,
        x,
        codes,
        scales,
        global_scales,
        expert_blocks,
        sorted_slots,
        ntpp,
        s,
        n,
        k,
        in_div,
        splitk,
        nacc,
    ):
        with self._lock:
            if self._closed:
                raise RuntimeError("provider is closed")
            if self._failed:
                raise RuntimeError(
                    "provider failed; prior execution outcome may be unknown"
                )
            t = self.torch
            tensors = (
                x,
                codes,
                scales,
                global_scales,
                expert_blocks,
                sorted_slots,
                ntpp,
            )
            dtypes = (t.float16, t.uint8, t.uint8, t.float32, t.int32, t.int32, t.int32)
            for tensor, dtype in zip(tensors, dtypes):
                if (
                    not isinstance(tensor, t.Tensor)
                    or tensor.device != self.device
                    or tensor.dtype != dtype
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        "FFI tensors must be contiguous typed device buffers"
                    )
            if (
                x.ndim != 2
                or global_scales.ndim != 2
                or expert_blocks.ndim != 1
                or sorted_slots.ndim != 1
                or ntpp.numel() != 1
            ):
                raise ValueError("malformed activation/global/metadata shape")
            e = global_scales.shape[0]
            if (
                any(type(v) is not int for v in (s, n, k, in_div, splitk, nacc))
                or not 1 <= e <= 1024
                or not 1 <= s <= 262144
                or not 32 <= n <= 65536
                or n % 32
                or not 64 <= k <= 65536
                or k % 64
                or not 1 <= in_div <= 64
                or s % in_div
                or tuple(x.shape) != (s // in_div, k)
                or tuple(global_scales.shape) != (e, n // 32)
                or (splitk, nacc)
                not in ((2, 1), (4, 1), (8, 1), (8, 2), (16, 1), (16, 2))
                or (k // 16) % splitk
                or not 1 <= expert_blocks.numel() <= 65535
                or sorted_slots.numel() != expert_blocks.numel() * 8
                or codes.numel() != e * n * k // 2
                or scales.numel() != e * n * k // 16
            ):
                raise ValueError("projection differs from bounded C ABI shape")
            resident_bytes = codes.numel() + scales.numel() + global_scales.numel() * 4
            # New per-call output and exact metadata plus retained activation.
            # Weights remain resident and are not charged again per pending call.
            workspace_bytes = (x.numel() + s * n) * 2 + (
                expert_blocks.numel() + sorted_slots.numel() + 1
            ) * 4
            if resident_bytes > self.max_resident_bytes:
                raise ValueError("resident projection exceeds explicit memory budget")
            if self._pending_bytes + workspace_bytes > self.max_pending_bytes:
                raise ValueError("pending workspace exceeds explicit memory budget")
            if not bool(t.isfinite(x).all()):
                raise ValueError("nonfinite FP16 provider activations")
            _validate_sm70_scale_domain(scales, global_scales, k)
            stream = t.cuda.current_stream(self.device)
            # Read DEVICE values only through Torch's supported transfer. Snapshot
            # to fresh HOST mirrors, then upload those exact values on our stream.
            host_blocks = expert_blocks.detach().cpu().contiguous().clone()
            host_slots = sorted_slots.detach().cpu().contiguous().clone()
            host_ntpp = ntpp.detach().cpu().contiguous().clone()
            blocks = host_blocks.to(self.device, copy=True).contiguous()
            slots = host_slots.to(self.device, copy=True).contiguous()
            padded = host_ntpp.to(self.device, copy=True).contiguous()
            y = t.full((s, n), float("nan"), dtype=t.float16, device=self.device)
            held = (x, codes, scales, global_scales, blocks, slots, padded, y)
            names = (
                "x",
                "codes",
                "scales",
                "gscales",
                "block_expert",
                "sorted_slots",
                "ntpp",
                "y",
            )
            p = make_params(
                dimensions=dict(
                    experts=e,
                    n=n,
                    k=k,
                    slots=s,
                    input_rows=x.shape[0],
                    in_div=in_div,
                    splitk=splitk,
                    nacc=nacc,
                ),
                buffers={
                    name: (tensor.data_ptr(), tensor.numel())
                    for name, tensor in zip(names, held)
                },
                expert_blocks=host_blocks.tolist(),
                sorted_slots=host_slots.tolist(),
                ntpp=int(host_ntpp.item()),
            )
            status = self.library.vf_qpn2_moe_validate(C.byref(p))
            if status:
                raise ValueError(
                    "C FFI metadata validation rejected projection: " + str(status)
                )
            # The C launch is invisible to Torch's allocator. Mark every storage
            # on the actual stream as well as retaining references: even caller
            # GC after an error must not recycle a tensor before GPU completion.
            for tensor in held:
                tensor.record_stream(stream)
            # Retain BEFORE the first opaque enqueue; errors do not prove that
            # the device did nothing. Never clear these on a launch exception.
            self.pending.append((stream, held, host_blocks, host_slots, host_ntpp, p))
            self._pending_bytes += workspace_bytes
            cuda_error = C.c_int32()
            try:
                with t.cuda.device(self.device):
                    self.admissions += 1
                    status = self.library.vf_qpn2_moe_launch(
                        C.byref(p), C.c_void_p(stream.cuda_stream), C.byref(cuda_error)
                    )
                if status or cuda_error.value:
                    raise RuntimeError(
                        "real QPN2 launch failed: FFI=%d CUDA=%d"
                        % (status, cuda_error.value)
                    )
            except BaseException:
                self._failed = True
                raise
            self.launches += 1
            return y

    def synchronize(self):
        with self._lock:
            try:
                super().synchronize()
            except BaseException:
                self._failed = True
                raise
            self._pending_bytes = 0

    def close(self):
        """Drain explicitly. Failure retains all outstanding borrowed buffers."""
        with self._lock:
            self.synchronize()
            self._closed = True
