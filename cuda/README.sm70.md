# QPN2-MoE SM70 boundary

This is a C ABI wrapper around exact donor device fragments from
v100-skinny commit b1a2181702e609dc92591343e46c63d1324ade43. It is not a
qualified Torch extension. No dense expert weight matrices are accepted.
The kernel, fp8 helper and MMA macro are MIT; the explicitly derived
TurboMind dequant8_tm helper carries Apache-2.0 attribution and license.
See third_party/ATTRIBUTION, the preserved licenses and NOTICE.

Rebuild without calling a GPU:

    bash cuda/build_sm70.sh /path/to/VoltaFabric-offline-root

The isolated CUDA 12.8.93 compiler and GCC 14.3 package binaries live only
under the offline root's .tools. Official component/archive sizes and SHA256
were checked before extraction. The local CUDA header has exactly six
external math declaration exception specifications changed to noexcept(true)
to match the host libc; original, diff and hashes are in .tools/cuda12/compat
and evidence/toolchain-compat.json. There is no unsupported compiler flag.
The toolchains must not be redistributed in a source/delivery pack.

The header documents ABI v1, compressed QPN fragment layout, length units,
splitk/nacc configurations and borrowed buffers. Host metadata is immutable
and must exactly match its device mirror. Validation rejects negative IDs,
out-of-range physical experts, duplicate/missing output slots, malformed
padding, bad dimensions, lengths and pointer alignment. It never reads
device memory. Synthetic pointers in CPU tests are intentional opaque values.
The six launch configurations are those of the donor Torch wrapper:
(2,1), (4,1), (8,1), (8,2), (16,1), (16,2).

The shared library links against libcudart.so.12; consumers supply the matching
CUDA runtime search path. Borrowed buffers and stream stay valid until
asynchronous completion. Launch success is only an enqueue result; the wrapper
does not initialize output, synchronize or copy data. The integrator must
upload validated metadata, handle empty ranks/batches without launch and
prove device allocation, mirror equality, output coverage/completion and
numerical results at the GPU gate. One expert GEMM is exposed; activation,
second GEMM, routing weights and collection belong to integration.

CUDA_COMPILED requires the compiled object/archive and a sm_70 ELF shown by
cuobjdump. Compilation and CPU validation provide no GPU correctness,
latency, throughput or memory residency qualification. The attempted optional
SASS dump failed because the system nvdisasm rejects SM70; its log is retained.
