#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'R692 GGUF checkpoint inventory test: %s\n' "$1" >&2
  exit 2
}

: "${VF_LLAMA_SOURCE_DIR:?VF_LLAMA_SOURCE_DIR is required}"
: "${VF_GGML_BASE_LIBRARY:?VF_GGML_BASE_LIBRARY is required}"
: "${VF_GGUF_INVENTORY_FIXTURE:?VF_GGUF_INVENTORY_FIXTURE is required}"
: "${VF_TEST_OUTPUT_DIR:?VF_TEST_OUTPUT_DIR is required}"
: "${CXX:?CXX is required}"

expected_revision=3cf03257f219afbe7334045ff7c6a06ac68c627d
actual_revision=$(git -C "$VF_LLAMA_SOURCE_DIR" rev-parse HEAD)
[[ "$actual_revision" == "$expected_revision" ]] || die "unexpected llama.cpp revision: $actual_revision"
[[ -f "$VF_GGML_BASE_LIBRARY" ]] || die "pinned GGML base library is missing"
[[ -f "$VF_GGUF_INVENTORY_FIXTURE" ]] || die "existing synthetic GGUF fixture is missing"

test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$VF_TEST_OUTPUT_DIR"
binary="$VF_TEST_OUTPUT_DIR/r692_gguf_checkpoint_inventory"
output="$VF_TEST_OUTPUT_DIR/r692_inventory.json"
"$CXX" -std=c++17 -Wall -Wextra -Werror -O2 \
  -I"$VF_LLAMA_SOURCE_DIR/ggml/include" \
  "$test_dir/r692_gguf_checkpoint_inventory.cpp" \
  "$VF_GGML_BASE_LIBRARY" \
  -Wl,-rpath,"$(dirname -- "$VF_GGML_BASE_LIBRARY")" \
  -o "$binary"

LD_LIBRARY_PATH="$(dirname -- "$VF_GGML_BASE_LIBRARY")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$binary" "$VF_GGUF_INVENTORY_FIXTURE" > "$output"
python3 - "$output" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
assert result["status"] == "METADATA_ONLY_NO_WEIGHTS_LOADED"
assert result["gguf_version"] == 3
assert result["file_bytes"] == 13536
assert result["tensor_count"] == 3
assert result["tensor_type_counts"] == {"q8_0": 3}
assert result["metadata"]["general.architecture"] == "qwen2moe"
assert result["metadata"]["tokenizer.ggml.model"] is None
assert result["metadata"]["general.license"] is None
assert result["metadata"]["license_verification_required"] is True
assert result["architecture"]["embedding_length"] == 32
assert result["architecture"]["block_count"] == 1
assert result["architecture"]["expert_count"] == 4
assert result["compatibility_assessment"] == "NOT_ATTEMPTED"
print("R692 metadata-only GGUF inventory passed on retained synthetic fixture")
PY

invalid_fixture="$VF_TEST_OUTPUT_DIR/r692-invalid-fixture.gguf"
invalid_output="$VF_TEST_OUTPUT_DIR/r692-invalid-output.json"
printf 'not-a-GGUF' > "$invalid_fixture"
if "$binary" "$invalid_fixture" > "$invalid_output" 2>/dev/null; then
  die "invalid GGUF was accepted"
else
  invalid_status=$?
fi
[[ "$invalid_status" -eq 3 ]] || die "invalid GGUF returned status $invalid_status"
[[ ! -s "$invalid_output" ]] || die "invalid GGUF published partial JSON"
printf 'R692 invalid GGUF rejected without JSON output\n'
