/* VoltaFabric QPN2-MoE C ABI, v1. Copyright 2026. */
#ifndef VF_QPN2_MOE_FFI_H
#define VF_QPN2_MOE_FFI_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define VF_QPN2_ABI_VERSION 1u
enum vf_qpn2_status {
  VF_QPN2_OK = 0, VF_QPN2_ABI = -1, VF_QPN2_DIMENSION = -2,
  VF_QPN2_LENGTH = -3, VF_QPN2_POINTER = -4, VF_QPN2_INDEX = -5,
  VF_QPN2_COVERAGE = -6, VF_QPN2_RESOURCE = -7, VF_QPN2_CUDA = -8
};
typedef struct vf_qpn2_moe_params {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t experts, n, k, slots, input_rows, in_div, splitk, nacc;
  /* Borrowed contiguous DEVICE buffers. No ownership transfer or allocation.
   * codes: exact donor QPN fragment order [E,N/32,K/16,32,8] bytes;
   * scales: [E,N/32,K/16,32] fp8-e4m3 bytes; gscales: [E,N/32] fp32.
   * x/y are IEEE fp16 bit patterns. No dense weight buffer is accepted.
   * All device buffer lengths are minimum capacities in indicated units.
   * The required y span must be disjoint from every input/weight/metadata span.
   * Validation rejects overlap and address wrap without dereferencing device
   * pointers; caller also prevents aliases introduced by distinct mappings. */
  const uint8_t *codes; size_t codes_bytes;
  const uint8_t *scales; size_t scales_bytes;
  const float *gscales; size_t gscales_count;
  const uint16_t *x; size_t x_count;
  uint16_t *y; size_t y_count;
  const int32_t *block_expert; size_t block_expert_count;
  const int32_t *sorted_slots; size_t sorted_slots_count;
  const int32_t *ntpp; size_t ntpp_count;
  /* Borrowed immutable HOST mirrors, lengths exactly equal corresponding
   * device metadata counts. Caller must upload these exact values before launch
   * on the same stream and retain all buffers until GPU completion.
   * ntpp_host is padded slot count, divisible by 8, <= blocks*8.
   * Active expert IDs are physical stack indices [0,E). -1 means skipped block
   * containing padding only. Slot IDs >= slots mean padding; negative forbidden.
   * Host validation proves unique full output coverage for this metadata only.
   * Device accessibility, mirror equality, completion and numerical correctness
   * remain the integrator/GPU gate responsibility. Never dereference opaque
   * device pointers on the CPU. No output memset or synchronization is performed. */
  const int32_t *block_expert_host; size_t block_expert_host_count;
  const int32_t *sorted_slots_host; size_t sorted_slots_host_count;
  int32_t ntpp_host;
} vf_qpn2_moe_params;
int vf_qpn2_moe_validate(const vf_qpn2_moe_params *p);
/* Borrowed CUDA stream as opaque cudaStream_t. Enqueues real donor kernel.
 * Return OK means enqueue accepted, not completed/correct. cuda_error_out is an
 * optional HOST int receiving cudaPeekAtLastError(); caller handles async errors.
 * Empty routing batches/ranks must be skipped by the integrator (slots >= 1). */
int vf_qpn2_moe_launch(const vf_qpn2_moe_params *p, void *stream,
                       int32_t *cuda_error_out);
#ifdef __cplusplus
}
#endif
#endif
