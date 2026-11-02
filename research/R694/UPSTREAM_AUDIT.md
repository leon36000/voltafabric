# R694 upstream audit notes

## VarioPath

Primary record: https://arxiv.org/abs/2609.34340 (v1, 2026-09-28),
“VarioPath: Workload-Aware All-to-All Communication for PCIe GPU Clusters.”
The abstract describes an offline topology analyzer/catalog and an online,
demand-aware schedule for each AlltoAllv invocation. The paper is a design
reference only; this lot neither ports a kernel nor adopts its reported
performance numbers. No corresponding source repository or software license was
verified in this audit.

## MoE CPU-GPU Collaborative Inference

Primary paper: https://arxiv.org/abs/2512.16473.
Linked source: https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference
Pinned commit: 1687abe25d79b27fe26367620a24d79c72eed491
Pinned LICENSE: MIT.

The inspected src/model.py implementation pins expert tensors in host memory,
uses a BF16 GPU cache and per-layer FIFO/LRU bookkeeping, and performs CPU work
on misses while scheduling cache copies. The code path is a research prototype
tied to its stated PyTorch/CUDA/GPU setup. In the inspected cache path, tags and
valid markers are int8 and the forward path handles exactly two selected
experts. These are reasons to adapt the idea only after format/range and
synchronization contracts are explicit; this lot copies no upstream code and
does not infer V100 compatibility.

## Prior engine reuse

R693's separate report records the llama.cpp MIT pin and the use of its public
model/decode API with a local seam patch. That is the only code-level donor
integration in these two consecutive lots. No source from the cache repository
or VarioPath was copied.

