#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'GGUF build_moe_ffn test: %s\n' "$1" >&2
  exit 2
}

: "${VF_LLAMA_SOURCE_DIR:?VF_LLAMA_SOURCE_DIR is required}"
: "${VF_LLAMA_LIBRARY:?VF_LLAMA_LIBRARY is required}"
: "${VF_TEST_OUTPUT_DIR:?VF_TEST_OUTPUT_DIR is required}"
: "${CXX:?CXX is required}"

expected_revision=3cf03257f219afbe7334045ff7c6a06ac68c627d
actual_revision=$(git -C "$VF_LLAMA_SOURCE_DIR" rev-parse HEAD)
[[ "$actual_revision" == "$expected_revision" ]] || die "unexpected donor revision: $actual_revision"
donor_dirty=$(git -C "$VF_LLAMA_SOURCE_DIR" status --porcelain -- ggml)
[[ -z "$donor_dirty" ]] || die "donor GGML dependency has local changes"

for library in \
  libggml-cpu.so.0.24.0 \
  libggml-base.so.0.24.0 \
  libggml.so.0.24.0; do
  [[ -f "$(dirname -- "$VF_LLAMA_LIBRARY")/$library" ]] || die "missing pinned donor library: $library"
done
[[ -f "$VF_LLAMA_LIBRARY" ]] || die "pinned llama library does not exist"
[[ "$(basename -- "$VF_LLAMA_LIBRARY")" == "libllama.so.0.4.1" ]] || die "unexpected llama library filename"

mkdir -p "$VF_TEST_OUTPUT_DIR"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
binary="$VF_TEST_OUTPUT_DIR/qwen2moe_llama_gguf_build_moe_ffn_test"

"$CXX" -std=c++17 -Wall -Wextra -Werror -O0 -g \
  -I"$script_dir" \
  -I"$VF_LLAMA_SOURCE_DIR/src" \
  -I"$VF_LLAMA_SOURCE_DIR/include" \
  -I"$VF_LLAMA_SOURCE_DIR/ggml/include" \
  -I"$VF_LLAMA_SOURCE_DIR/ggml/src" \
  "$script_dir/qwen2moe_llama_gguf_build_moe_ffn_test.cpp" \
  "$script_dir/route_graph_transaction.cpp" \
  "$script_dir/qwen2moe_route_invocation.cpp" \
  "$script_dir/qwen2moe_llama_route_bridge.cpp" \
  "$VF_LLAMA_LIBRARY" \
  "$(dirname -- "$VF_LLAMA_LIBRARY")/libggml-cpu.so.0.24.0" \
  "$(dirname -- "$VF_LLAMA_LIBRARY")/libggml-base.so.0.24.0" \
  "$(dirname -- "$VF_LLAMA_LIBRARY")/libggml.so.0.24.0" \
  -Wl,-rpath,"$(dirname -- "$VF_LLAMA_LIBRARY")" \
  -o "$binary"

donor_library_dir=$(dirname -- "$VF_LLAMA_LIBRARY")
LD_LIBRARY_PATH="$donor_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$binary"

