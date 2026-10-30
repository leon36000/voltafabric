# Experimental native GGML expert provider

This CPU-only, single-token component computes SC117 native expert blobs using the GGML CPU type traits at llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The operation follows the ordinary quantized gate/up, SiLU, quantized intermediate, and down sequence used by Strata at `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (`src/kernels/cpu/native_expert.cpp`). Both upstream projects use MIT licenses; their notices are included here. The upstream GGML sources are not vendored in this repository.

The API is experimental. It takes caller-owned packed bytes and one FP32 row, validates the supported SC117 type pairs and geometry, and produces one unweighted FP32 row. The caller must bind and authenticate the blob and its layer/expert/model identity, keep weights resident, and apply routing coefficients exactly once. No Rust Session or actual checkpoint is connected here. The GGML activation quantizers and dot arithmetic are distinct from the QPN2/MXFP4 implementations and require independent model-level qualification.

To build, provide an isolated **clean** llama.cpp checkout of the exact commit:

```
cmake -S adapters/native_ggml -B <build-dir> -DGGML_SOURCE_DIR=<llama-checkout> -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build <build-dir> --target vf_native_ggml_test --parallel 1
ctest --test-dir <build-dir> --output-on-failure
```

CUDA, Metal, Vulkan, OpenMP and native CPU specialization are disabled by this isolated project. The source revision is checked at configuration; do not point it at a live installation. CPU test vectors are synthetic and homogeneous, exercise seven observed format pairs and selected failure conditions, and compare the GGML dot path to scalar multiplication of independently dequantized rows and activations. The scalar oracle still uses GGML dequantization and quantization, and is not a model-level or cross-device equivalence proof.

The optional Rust feature `native-ggml-cpu` links the three static libraries from an already configured and built isolated CMake directory. Set `VF_GGML_NATIVE_BUILD` to that absolute build directory. The small `vf-ggml-native-ffi` crate owns the C ABI pointer and copies one admitted expert into a native resident; the main crate remains free of unsafe code. `native_ggml_cpu::Resident` checks the supplied layout, expert span, model binding and payload hash again before loading and returns unweighted FP32 values. The supplied model revision and hashes still require trusted provenance; this is not a session, durable identity claim, worker isolation, GPU backend or model parity proof.

```
VF_GGML_NATIVE_BUILD=<build-dir> cargo test --offline --locked --features native-ggml-cpu --test strata_native_provider
```
