# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to vLLM and 1Cat-vLLM.
# Unchanged _qpn_prepack from gilby/volta-nvfp4 commit b1a2181702e609dc92591343e46c63d1324ade43.
import torch

def _qpn_prepack(codes: torch.Tensor, scales: torch.Tensor):
    """Fragment-order prepack for gemm_qpn: [tile N/32][group K/16]
    [lane 32] x 8B, nibbles pre-interleaved so the TM decoder's (i, i+4)
    output is exactly the adjacent-k B-fragment register pair. Pure
    permutation of the checkpoint bytes."""
    n, k2 = codes.shape
    k = k2 * 2
    if n % 32 or k % 64:
        return None, None
    dev = codes.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=dev)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7,
                           8, 10, 12, 14, 9, 11, 13, 15], device=dev)
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k)
    g = torch.arange(groups, device=dev)
    kidx = g.view(groups, 1) * 16 + korder.view(1, 16)
    qc = torch.empty(tiles, groups, 32, 8, dtype=torch.uint8, device=dev)
    qs = torch.empty(tiles, groups, 32, dtype=torch.uint8, device=dev)
    # Chunk the gather: the int64 broadcast-index intermediates are 16x
    # the payload — one-shot on lm_head (37984x5120) spikes 2.4 GiB and
    # OOM'd the memory-profiler forward. Cap transients at ~300 MB.
    chunk = max(1, 36864 // groups)
    for t0 in range(0, tiles, chunk):
        t1 = min(t0 + chunk, tiles)
        tt = t1 - t0
        ncol = (torch.arange(t0, t1, device=dev).view(tt, 1) * 32
                + col.view(1, 32))
        nb = nib[ncol.view(tt, 1, 32, 1).expand(tt, groups, 32, 16),
                 kidx.view(1, groups, 1, 16).expand(tt, groups, 32, 16)]
        qc[t0:t1] = nb[..., 0::2] | (nb[..., 1::2] << 4)
        qs[t0:t1] = scales[ncol.view(tt, 1, 32).expand(tt, groups, 32),
                           g.view(1, groups, 1).expand(tt, groups, 32)]
    del nib
    return qc.view(-1).contiguous(), qs.view(-1).contiguous()
