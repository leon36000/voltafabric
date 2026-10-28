#!/usr/bin/env bash
# Invoke only after docs/HARDWARE_GATE.md and reservation of one V100.
set -euo pipefail
: "${VF_GPU_PYTHON:?Set an existing CUDA12 Torch SM70 interpreter; the CPU venv cannot run H1}"
: "${VF_GPU_UUID:?Set the UUID of the available reserved V100}"
VF_SOURCE_DIR=$(cd "$(dirname "$0")/.." && pwd)
VF_ROOT=$(cd "$VF_SOURCE_DIR/.." && pwd)
VF_GPU_DEVICE=${VF_GPU_DEVICE:-0}
VF_H1_RUN_ID=${VF_H1_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
cd "$VF_SOURCE_DIR"
# Default to the qualified reproducible library, not an older evidence artifact.
VF_H1_LIBRARY=${VF_H1_LIBRARY:-"$VF_SOURCE_DIR/build/libvf_qpn2_moe_sm70.so"}
VF_H1_CUDA_PROOF=${VF_H1_CUDA_PROOF:-"$VF_SOURCE_DIR/evidence/continuation_a23/CUDA_COMPILED.json"}
VF_H1_CUDART=${VF_H1_CUDART:-"$VF_ROOT/.tools/cuda12/components/cuda_cudart-linux-x86_64-12.8.90-archive/lib/libcudart.so.12"}
VF_H1_BINARY=${VF_H1_BINARY:-"$VF_ROOT/target/release/vf"}
VF_H1_OUTPUT=${VF_H1_OUTPUT:-"$VF_SOURCE_DIR/evidence/hardware-H1-$VF_H1_RUN_ID.json"}
VF_FULL_ARGS=()
VF_ADAPTER_ARGS=()
case "${VF_H1_FULL_EXPERT:-0}" in
  0) ;;
  1) VF_FULL_ARGS=(--full-expert-sample "${VF_H1_FULL_EXPERT_SAMPLE:-$VF_SOURCE_DIR/fixtures/checkpoint_qwen36_expert0_full.safetensors}") ;;
  *) echo "VF_H1_FULL_EXPERT must be 0 or 1" >&2; exit 2 ;;
esac
case "${VF_H1_FULL_EXPERT_ADAPTER:-0}" in
  0) ;;
  1)
    if [[ "${VF_H1_FULL_EXPERT:-0}" != 1 ]]; then
      echo "VF_H1_FULL_EXPERT_ADAPTER requires VF_H1_FULL_EXPERT=1" >&2
      exit 2
    fi
    VF_ADAPTER_ARGS=(--full-expert-adapter)
    ;;
  *) echo "VF_H1_FULL_EXPERT_ADAPTER must be 0 or 1" >&2; exit 2 ;;
esac
exec "$VF_GPU_PYTHON" scripts/hardware_canary.py --ack-hardware-gate \
  --gpu-uuid "$VF_GPU_UUID" --device "$VF_GPU_DEVICE" \
  --library "$VF_H1_LIBRARY" --cuda-proof "$VF_H1_CUDA_PROOF" \
  --cudart "$VF_H1_CUDART" \
  --routing-binary "$VF_H1_BINARY" --out "$VF_H1_OUTPUT" \
  --resident-io "${VF_H1_RESIDENT_IO:-json}" "${VF_FULL_ARGS[@]}" \
  "${VF_ADAPTER_ARGS[@]}"
