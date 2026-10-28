#!/usr/bin/env bash
# Use only after the hardware gate and reservation of one V100.
set -euo pipefail
VF_BUNDLE_DIR=$(cd "$(dirname "$0")" && pwd)
: "${VF_GPU_PYTHON:?Set the existing verified CUDA12 Torch SM70 interpreter}"
: "${VF_GPU_UUID:?Set the available reserved V100 UUID}"
VF_PREFLIGHT_PYTHON=${VF_PREFLIGHT_PYTHON:-python3}
VF_H1_RUN_ID=${VF_H1_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
case "$VF_H1_RUN_ID" in
  ""|*[!A-Za-z0-9._-]*|.|..) echo "Invalid H1 run identifier" >&2; exit 2;;
esac
VF_H1_RECEIPT_DIR=${VF_H1_RECEIPT_DIR:-"$VF_BUNDLE_DIR/../h1-receipts"}
VF_H1_RECEIPT_DIR=$(mkdir -p "$VF_H1_RECEIPT_DIR" && cd "$VF_H1_RECEIPT_DIR" && pwd)
export VF_H1_RECEIPT_DIR
"$VF_PREFLIGHT_PYTHON" "$VF_BUNDLE_DIR/source/scripts/h1_preflight.py" \
  --bundle "$VF_BUNDLE_DIR" --out "$VF_H1_RECEIPT_DIR/static-$VF_H1_RUN_ID.json"
export VF_H1_LIBRARY="$VF_BUNDLE_DIR/lib/libvf_qpn2_moe_sm70.so"
export VF_H1_CUDA_PROOF="$VF_BUNDLE_DIR/provenance/CUDA_COMPILED.json"
export VF_H1_CUDART="$VF_BUNDLE_DIR/lib/libcudart.so.12"
export VF_H1_BINARY="$VF_BUNDLE_DIR/bin/vf"
export VF_H1_OUTPUT="$VF_H1_RECEIPT_DIR/hardware-$VF_H1_RUN_ID.json"
exec bash "$VF_BUNDLE_DIR/source/handoff/hardware_canary.sh"
