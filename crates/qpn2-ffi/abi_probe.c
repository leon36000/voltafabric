#include "qpn2_moe_ffi.h"
#include <stdio.h>
_Static_assert(VF_QPN2_ABI_VERSION == 1u, "ABI version changed");
_Static_assert(sizeof(vf_qpn2_moe_params) == 208u, "v1 requires 64-bit struct208");
int main(void) {
  printf("ABI %u\n", VF_QPN2_ABI_VERSION);
  printf("SIZE %zu\n", sizeof(vf_qpn2_moe_params));
  printf("ALIGN %zu\n", _Alignof(vf_qpn2_moe_params));
#define FIELD(name, type) printf("FIELD %s %zu %zu %zu\n", #name, offsetof(vf_qpn2_moe_params, name), sizeof(((vf_qpn2_moe_params *)0)->name), _Alignof(type))
  FIELD(abi_version, uint32_t);
  FIELD(struct_size, uint32_t);
  FIELD(experts, uint32_t);
  FIELD(n, uint32_t);
  FIELD(k, uint32_t);
  FIELD(slots, uint32_t);
  FIELD(input_rows, uint32_t);
  FIELD(in_div, uint32_t);
  FIELD(splitk, uint32_t);
  FIELD(nacc, uint32_t);
  FIELD(codes, const uint8_t *);
  FIELD(codes_bytes, size_t);
  FIELD(scales, const uint8_t *);
  FIELD(scales_bytes, size_t);
  FIELD(gscales, const float *);
  FIELD(gscales_count, size_t);
  FIELD(x, const uint16_t *);
  FIELD(x_count, size_t);
  FIELD(y, uint16_t *);
  FIELD(y_count, size_t);
  FIELD(block_expert, const int32_t *);
  FIELD(block_expert_count, size_t);
  FIELD(sorted_slots, const int32_t *);
  FIELD(sorted_slots_count, size_t);
  FIELD(ntpp, const int32_t *);
  FIELD(ntpp_count, size_t);
  FIELD(block_expert_host, const int32_t *);
  FIELD(block_expert_host_count, size_t);
  FIELD(sorted_slots_host, const int32_t *);
  FIELD(sorted_slots_host_count, size_t);
  FIELD(ntpp_host, int32_t);
#undef FIELD
  printf("STATUSES %d %d %d %d %d %d %d %d %d\n", VF_QPN2_OK,
    VF_QPN2_ABI, VF_QPN2_DIMENSION, VF_QPN2_LENGTH, VF_QPN2_POINTER,
    VF_QPN2_INDEX, VF_QPN2_COVERAGE, VF_QPN2_RESOURCE, VF_QPN2_CUDA);
  return 0;
}
