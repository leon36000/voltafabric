#include "qpn2_moe_ffi.h"
#include <limits.h>
#include <stdlib.h>

static int aligned(const void *p, uintptr_t alignment) {
  return p != NULL && (uintptr_t)p % alignment == 0;
}
static int multiply(size_t a, size_t b, size_t *out) {
  if (a && b > SIZE_MAX / a) return 0;
  *out = a * b;
  return 1;
}
static int range_end(const void *p, size_t bytes, uintptr_t *end) {
  uintptr_t start = (uintptr_t)p;
  if (bytes > UINTPTR_MAX - start) return 0;
  *end = start + bytes;
  return 1;
}
static int output_disjoint(const void *output, size_t output_bytes,
                           const void *input, size_t input_bytes) {
  uintptr_t oe, ie;
  if (!range_end(output, output_bytes, &oe) ||
      !range_end(input, input_bytes, &ie)) return 0;
  return oe <= (uintptr_t)input || ie <= (uintptr_t)output;
}
int vf_qpn2_moe_validate(const vf_qpn2_moe_params *p) {
  size_t nk, scales, codes, globals, inputs, outputs, padded;
  size_t global_bytes, input_bytes, output_bytes;
  unsigned char *seen;
  if (!p) return VF_QPN2_POINTER;
  if (p->abi_version != VF_QPN2_ABI_VERSION ||
      p->struct_size != sizeof(*p)) return VF_QPN2_ABI;
  if (!p->experts || p->experts > 1024 || !p->n || p->n > 65536 ||
      p->n % 32 || !p->k || p->k > 65536 || p->k % 64 ||
      !p->slots || p->slots > 262144 || !p->in_div || p->in_div > 64 ||
      p->slots % p->in_div || p->input_rows != p->slots / p->in_div ||
      !((p->splitk == 2 && p->nacc == 1) ||
        (p->splitk == 4 && p->nacc == 1) ||
        (p->splitk == 8 && (p->nacc == 1 || p->nacc == 2)) ||
        (p->splitk == 16 && (p->nacc == 1 || p->nacc == 2))) ||
      (p->k / 16) % p->splitk ||
      !p->block_expert_count || p->block_expert_count > 65535)
    return VF_QPN2_DIMENSION;
  if (!multiply(p->n, p->k, &nk) ||
      !multiply(p->experts, nk / 16, &scales) ||
      !multiply(scales, 8, &codes) ||
      !multiply(p->experts, p->n / 32, &globals) ||
      !multiply(p->input_rows, p->k, &inputs) ||
      !multiply(p->slots, p->n, &outputs) ||
      !multiply(p->block_expert_count, 8, &padded) ||
      !multiply(globals, sizeof(float), &global_bytes) ||
      !multiply(inputs, sizeof(uint16_t), &input_bytes) ||
      !multiply(outputs, sizeof(uint16_t), &output_bytes))
    return VF_QPN2_LENGTH;
  if (p->codes_bytes < codes || p->scales_bytes < scales ||
      p->gscales_count < globals || p->x_count < inputs ||
      p->y_count < outputs || p->sorted_slots_count != padded ||
      p->ntpp_count != 1 ||
      p->block_expert_host_count != p->block_expert_count ||
      p->sorted_slots_host_count != p->sorted_slots_count)
    return VF_QPN2_LENGTH;
  if (!aligned(p->codes, 8) || !aligned(p->scales, 1) ||
      !aligned(p->gscales, 4) || !aligned(p->x, 16) || !aligned(p->y, 2) ||
      !aligned(p->block_expert, 4) || !aligned(p->sorted_slots, 4) ||
      !aligned(p->ntpp, 4) || !aligned(p->block_expert_host, 4) ||
      !aligned(p->sorted_slots_host, 4))
    return VF_QPN2_POINTER;
  /* Check required spans without dereferencing any device address. Outputs
   * must not alias readers under the donor's __restrict__ contract. */
  if (!output_disjoint(p->y, output_bytes, p->codes, codes) ||
      !output_disjoint(p->y, output_bytes, p->scales, scales) ||
      !output_disjoint(p->y, output_bytes, p->gscales, global_bytes) ||
      !output_disjoint(p->y, output_bytes, p->x, input_bytes) ||
      !output_disjoint(p->y, output_bytes, p->block_expert, p->block_expert_count * sizeof(int32_t)) ||
      !output_disjoint(p->y, output_bytes, p->sorted_slots, p->sorted_slots_count * sizeof(int32_t)) ||
      !output_disjoint(p->y, output_bytes, p->ntpp, sizeof(int32_t)) ||
      !output_disjoint(p->y, output_bytes, p->block_expert_host, p->block_expert_host_count * sizeof(int32_t)) ||
      !output_disjoint(p->y, output_bytes, p->sorted_slots_host, p->sorted_slots_host_count * sizeof(int32_t)))
    return VF_QPN2_POINTER;
  if (p->ntpp_host < 0 || p->ntpp_host % 8 ||
      (size_t)p->ntpp_host > padded) return VF_QPN2_INDEX;
  seen = (unsigned char *)calloc(p->slots, 1);
  if (!seen) return VF_QPN2_RESOURCE;
  for (size_t b = 0; b < (size_t)p->ntpp_host / 8; ++b) {
    int32_t expert = p->block_expert_host[b];
    if (expert < -1 || expert >= (int32_t)p->experts) {
      free(seen); return VF_QPN2_INDEX;
    }
    for (size_t r = 0; r < 8; ++r) {
      int32_t slot = p->sorted_slots_host[b * 8 + r];
      if (slot < 0 || (expert == -1 && slot < (int32_t)p->slots)) {
        free(seen); return VF_QPN2_INDEX;
      }
      if (slot < (int32_t)p->slots) {
        if (seen[slot]) { free(seen); return VF_QPN2_COVERAGE; }
        seen[slot] = 1;
      }
    }
  }
  for (size_t s = 0; s < p->slots; ++s)
    if (!seen[s]) { free(seen); return VF_QPN2_COVERAGE; }
  free(seen);
  return VF_QPN2_OK;
}
