#!/usr/bin/env bash
# Compile only: never launches CUDA code or queries a device.
set -euo pipefail
VF_TOOL_ROOT=${1:?Pass the canonical offline root containing .tools}
VF_TOOL_ROOT=$(cd "$VF_TOOL_ROOT" && pwd)
VF_SOURCE_DIR=$(cd "$(dirname "$0")/.." && pwd)
VF_NVCC=${VF_NVCC:-"$VF_TOOL_ROOT/.tools/cuda12/components/cuda_nvcc-linux-x86_64-12.8.93-archive/bin/nvcc"}
VF_CUDART=${VF_CUDART:-"$VF_TOOL_ROOT/.tools/cuda12/components/cuda_cudart-linux-x86_64-12.8.90-archive"}
VF_CCCL=${VF_CCCL:-"$VF_TOOL_ROOT/.tools/cuda12/components/cuda_cccl-linux-x86_64-12.8.90-archive"}
VF_CC=${VF_CC:-"$VF_TOOL_ROOT/.tools/gcc14/usr/bin/g++-14"}
VF_CUOBJDUMP=${VF_CUOBJDUMP:-"$VF_TOOL_ROOT/.tools/cuda12/components/cuda_cuobjdump-linux-x86_64-12.8.90-archive/bin/cuobjdump"}
# Without --keep, nvcc embeds tmpxft_<pid>_..._qpn2_moe_sm70.cudafe1.cpp in the
# object, so every build differs by those bytes alone. --keep names the
# intermediate after the input instead, which makes the .o/.a/.so byte-identical
# across invocations and paths; the keep directory itself does not affect output.
VF_KEEP_DIR=${VF_KEEP_DIR:-"$VF_TOOL_ROOT/.runs/nvcc-keep"}
cd "$VF_SOURCE_DIR"
mkdir -p build evidence "$VF_KEEP_DIR"
exec > >(tee evidence/rebuild.log) 2>&1
set -x
"$VF_NVCC" --version
"$VF_CC" --version
gcc -std=c11 -Wall -Wextra -Werror -pedantic -fsanitize=address,undefined -g \
  tests/qpn2_moe_validate.c cuda/qpn2_moe_validate.c -o build/validate_cpu
ASAN_OPTIONS=detect_leaks=1 ./build/validate_cpu
"$VF_NVCC" -std=c++17 -O3 -Xcompiler -fPIC -arch=sm_70 -ccbin "$VF_CC" \
  -I "$VF_CUDART/include" -I "$VF_CCCL/include" \
  --keep --keep-dir "$VF_KEEP_DIR" \
  -c cuda/qpn2_moe_sm70.cu -o build/qpn2_moe_sm70.o
gcc -std=c11 -O2 -fPIC -Wall -Wextra -Werror -pedantic \
  -c cuda/qpn2_moe_validate.c -o build/qpn2_moe_validate.o
# -D keeps the archive deterministic (zeroed mtime/uid/gid/mode).
ar rcsD build/libvf_qpn2_moe_sm70.a build/qpn2_moe_sm70.o build/qpn2_moe_validate.o
gcc -shared -Wl,--no-undefined build/qpn2_moe_sm70.o build/qpn2_moe_validate.o \
  -L "$VF_CUDART/lib" -lcudart -o build/libvf_qpn2_moe_sm70.so
"$VF_CUOBJDUMP" --list-elf build/qpn2_moe_sm70.o
"$VF_CUOBJDUMP" --dump-elf build/qpn2_moe_sm70.o > evidence/ffi_dump_elf.log
nm -g --defined-only build/libvf_qpn2_moe_sm70.a
readelf -d build/libvf_qpn2_moe_sm70.so
sha256sum cuda/qpn2_moe_ffi.h cuda/qpn2_moe_core.cuh cuda/qpn2_moe_sm70.cu \
  cuda/qpn2_moe_validate.c tests/qpn2_moe_validate.c cuda/build_sm70.sh \
  build/qpn2_moe_sm70.o build/libvf_qpn2_moe_sm70.a \
  build/libvf_qpn2_moe_sm70.so > evidence/SHA256SUMS
cat evidence/SHA256SUMS
