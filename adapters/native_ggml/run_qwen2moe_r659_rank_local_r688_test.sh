#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'R688 same-process donor/VoltaFabric CPU test: %s\n' "$1" >&2
  exit 2
}

: "${VF_LLAMA_SOURCE_DIR:?VF_LLAMA_SOURCE_DIR is required}"
: "${VF_LLAMA_LIBRARY:?VF_LLAMA_LIBRARY is required}"
: "${VF_GGML_NATIVE_BUILD:?VF_GGML_NATIVE_BUILD must name the verified R655 static CPU build}"
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

for archive in \
  "$VF_GGML_NATIVE_BUILD/libvf_native_ggml.a" \
  "$VF_GGML_NATIVE_BUILD/ggml/src/libggml-cpu.a" \
  "$VF_GGML_NATIVE_BUILD/ggml/src/libggml-base.a"; do
  [[ -f "$archive" ]] || die "missing verified VoltaFabric/GGML CPU archive: $archive"
done

mkdir -p "$VF_TEST_OUTPUT_DIR"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
loader_manifest="$script_dir/r688/rust-loader/Cargo.toml"
cargo_target="$VF_TEST_OUTPUT_DIR/cargo-target"
worker="$cargo_target/debug/vf-rank-session"
loader="$cargo_target/release/libvf_r688_loader.so"
binary="$VF_TEST_OUTPUT_DIR/qwen2moe_llama_r659_rank_local_r688_test"

CARGO_TARGET_DIR="$cargo_target" VF_GGML_NATIVE_BUILD="$VF_GGML_NATIVE_BUILD" \
  cargo build --offline --manifest-path "$repo_root/Cargo.toml" \
  --features native-ggml-cpu --bin vf-rank-session
CARGO_TARGET_DIR="$cargo_target" VF_GGML_NATIVE_BUILD="$VF_GGML_NATIVE_BUILD" \
  cargo build --offline --manifest-path "$loader_manifest" --release
[[ -x "$worker" ]] || die "rank-session worker was not built"
[[ -f "$loader" ]] || die "VoltaFabric Rust loader library was not built"

"$CXX" -std=c++17 -Wall -Wextra -Werror -O0 -g \
  -I"$script_dir" \
  -I"$VF_LLAMA_SOURCE_DIR/src" \
  -I"$VF_LLAMA_SOURCE_DIR/include" \
  -I"$VF_LLAMA_SOURCE_DIR/ggml/include" \
  -I"$VF_LLAMA_SOURCE_DIR/ggml/src" \
  "$script_dir/qwen2moe_llama_r659_rank_local_r688_test.cpp" \
  "$script_dir/route_graph_transaction.cpp" \
  "$script_dir/qwen2moe_route_invocation.cpp" \
  "$script_dir/qwen2moe_llama_route_bridge.cpp" \
  -L"$cargo_target/release" -Wl,--no-as-needed -lvf_r688_loader \
  "$VF_LLAMA_LIBRARY" \
  "$(dirname -- "$VF_LLAMA_LIBRARY")/libggml-cpu.so.0.24.0" \
  "$(dirname -- "$VF_LLAMA_LIBRARY")/libggml-base.so.0.24.0" \
  "$(dirname -- "$VF_LLAMA_LIBRARY")/libggml.so.0.24.0" \
  -Wl,-rpath,"$cargo_target/release" \
  -Wl,-rpath,"$(dirname -- "$VF_LLAMA_LIBRARY")" \
  -o "$binary"

donor_library_dir=$(dirname -- "$VF_LLAMA_LIBRARY")
VF_RANK_SESSION_BIN="$worker" \
LD_LIBRARY_PATH="$cargo_target/release:$donor_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$binary"
