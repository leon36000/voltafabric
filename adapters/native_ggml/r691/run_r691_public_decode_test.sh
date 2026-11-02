#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'R691 pinned llama.cpp public decode test: %s\n' "$1" >&2
  exit 2
}

: "${VF_LLAMA_SOURCE_DIR:?VF_LLAMA_SOURCE_DIR is required}"
: "${VF_LLAMA_LIBRARY:?VF_LLAMA_LIBRARY is required}"
: "${VF_GGML_NATIVE_BUILD:?VF_GGML_NATIVE_BUILD is required}"
: "${VF_VOLTA_REPO:?VF_VOLTA_REPO is required}"
: "${VF_TEST_OUTPUT_DIR:?VF_TEST_OUTPUT_DIR is required}"
: "${CXX:?CXX is required}"

expected_revision=3cf03257f219afbe7334045ff7c6a06ac68c627d
actual_revision=$(git -C "$VF_LLAMA_SOURCE_DIR" rev-parse HEAD)
[[ "$actual_revision" == "$expected_revision" ]] || die "unexpected donor revision: $actual_revision"
donor_ggml_dirty=$(git -C "$VF_LLAMA_SOURCE_DIR" status --porcelain -- ggml)
[[ -z "$donor_ggml_dirty" ]] || die "donor GGML dependency has local changes"
[[ -f "$VF_LLAMA_LIBRARY" ]] || die "pinned llama.cpp library is missing"
[[ -f "$VF_GGML_NATIVE_BUILD/libvf_native_ggml.a" ]] || die "R655 native GGML archive missing"

adapter_dir="$VF_VOLTA_REPO/adapters/native_ggml"
test_dir="$(dirname -- "${BASH_SOURCE[0]}")"
loader_manifest="$adapter_dir/r691/rust-loader/Cargo.toml"
[[ -f "$loader_manifest" ]] || die "R691 Rust loader manifest missing"
mkdir -p "$VF_TEST_OUTPUT_DIR"
cargo_target="$VF_TEST_OUTPUT_DIR/cargo-target"
worker="$cargo_target/debug/vf-rank-session"
loader="$cargo_target/release/libvf_r691_loader.so"
binary="$VF_TEST_OUTPUT_DIR/r691_public_decode_test"

CARGO_TARGET_DIR="$cargo_target" VF_GGML_NATIVE_BUILD="$VF_GGML_NATIVE_BUILD" \
  cargo build --offline --manifest-path "$VF_VOLTA_REPO/Cargo.toml" \
  --features native-ggml-cpu --bin vf-rank-session
CARGO_TARGET_DIR="$cargo_target" VF_GGML_NATIVE_BUILD="$VF_GGML_NATIVE_BUILD" \
  cargo build --offline --manifest-path "$loader_manifest" --release
[[ -x "$worker" ]] || die "rank-session worker was not built"
[[ -f "$loader" ]] || die "R691 Session loader was not built"

"$CXX" -std=c++17 -Wall -Wextra -Werror -O0 -g \
  -I"$adapter_dir" \
  -I"$VF_LLAMA_SOURCE_DIR/src" \
  -I"$VF_LLAMA_SOURCE_DIR/include" \
  -I"$VF_LLAMA_SOURCE_DIR/ggml/include" \
  -I"$VF_LLAMA_SOURCE_DIR/ggml/src" \
  "$test_dir/r691_public_decode_test.cpp" \
  "$adapter_dir/route_graph_transaction.cpp" \
  "$adapter_dir/qwen2moe_route_invocation.cpp" \
  "$adapter_dir/qwen2moe_llama_route_bridge.cpp" \
  -L"$cargo_target/release" -Wl,--no-as-needed -lvf_r691_loader \
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
