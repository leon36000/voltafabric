#include "../cuda/qpn2_moe_ffi.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static int32_t experts[2] = {0, 1};
static int32_t slots[16] = {0, 1, 2, 3, 8, 8, 8, 8, 4, 5, 6, 7, 8, 8, 8, 8};
static vf_qpn2_moe_params good(void) {
  vf_qpn2_moe_params p;
  memset(&p, 0, sizeof(p));
  p.abi_version = 1; p.struct_size = sizeof(p);
  p.experts = 2; p.n = 32; p.k = 64; p.slots = 8; p.input_rows = 4;
  p.in_div = 2; p.splitk = 2; p.nacc = 1;
  /* Opaque synthetic aligned device addresses are never dereferenced. */
  p.codes = (const uint8_t *)(uintptr_t)0x10000; p.codes_bytes = 2048;
  p.scales = (const uint8_t *)(uintptr_t)0x20000; p.scales_bytes = 256;
  p.gscales = (const float *)(uintptr_t)0x30000; p.gscales_count = 2;
  p.x = (const uint16_t *)(uintptr_t)0x40000; p.x_count = 256;
  p.y = (uint16_t *)(uintptr_t)0x50000; p.y_count = 256;
  p.block_expert = (const int32_t *)(uintptr_t)0x60000; p.block_expert_count = 2;
  p.sorted_slots = (const int32_t *)(uintptr_t)0x70000; p.sorted_slots_count = 16;
  p.ntpp = (const int32_t *)(uintptr_t)0x80000; p.ntpp_count = 1;
  p.block_expert_host = experts; p.block_expert_host_count = 2;
  p.sorted_slots_host = slots; p.sorted_slots_host_count = 16; p.ntpp_host = 16;
  return p;
}
#define BAD(FIELD, VALUE, STATUS) do { vf_qpn2_moe_params p = good(); \
  p.FIELD = VALUE; assert(vf_qpn2_moe_validate(&p) == STATUS); ++checks; } while (0)
int main(void) {
  unsigned checks = 0;
  vf_qpn2_moe_params p = good();
  assert(vf_qpn2_moe_validate(&p) == VF_QPN2_OK); ++checks;
  for (uint32_t split = 2; split <= 16; split *= 2) {
    vf_qpn2_moe_params variant = good();
    variant.k = 256; variant.codes_bytes = 8192;
    variant.scales_bytes = 1024; variant.x_count = 1024;
    variant.splitk = split;
    assert(vf_qpn2_moe_validate(&variant) == VF_QPN2_OK); ++checks;
    if (split >= 8) {
      variant.nacc = 2;
      assert(vf_qpn2_moe_validate(&variant) == VF_QPN2_OK); ++checks;
    }
  }
  assert(vf_qpn2_moe_validate(NULL) == VF_QPN2_POINTER); ++checks;
  BAD(abi_version, 2, VF_QPN2_ABI);
  BAD(struct_size, 0, VF_QPN2_ABI);
  BAD(n, 31, VF_QPN2_DIMENSION);
  BAD(k, 32, VF_QPN2_DIMENSION);
  BAD(splitk, 16, VF_QPN2_DIMENSION);
  BAD(nacc, 2, VF_QPN2_DIMENSION);
  BAD(input_rows, 3, VF_QPN2_DIMENSION);
  BAD(in_div, 0, VF_QPN2_DIMENSION);
  BAD(slots, 0, VF_QPN2_DIMENSION);
  BAD(experts, 1025, VF_QPN2_DIMENSION);
  BAD(block_expert_count, 65536, VF_QPN2_DIMENSION);
  BAD(codes_bytes, 2047, VF_QPN2_LENGTH);
  BAD(scales_bytes, 255, VF_QPN2_LENGTH);
  BAD(gscales_count, 1, VF_QPN2_LENGTH);
  BAD(x_count, 255, VF_QPN2_LENGTH);
  BAD(y_count, 255, VF_QPN2_LENGTH);
  BAD(sorted_slots_count, 15, VF_QPN2_LENGTH);
  BAD(ntpp_count, 0, VF_QPN2_LENGTH);
  BAD(sorted_slots_host_count, 8, VF_QPN2_LENGTH);
  BAD(block_expert_host_count, 1, VF_QPN2_LENGTH);
  BAD(codes, NULL, VF_QPN2_POINTER);
  BAD(x, (const uint16_t *)(uintptr_t)0x40002, VF_QPN2_POINTER);
  BAD(ntpp_host, -8, VF_QPN2_INDEX);
  BAD(ntpp_host, 17, VF_QPN2_INDEX);
  BAD(ntpp_host, 24, VF_QPN2_INDEX);
  BAD(ntpp_host, 8, VF_QPN2_COVERAGE);
  experts[0] = 2; assert(vf_qpn2_moe_validate(&p) == VF_QPN2_INDEX); ++checks;
  experts[0] = -1; assert(vf_qpn2_moe_validate(&p) == VF_QPN2_INDEX); ++checks;
  experts[0] = 0;
  slots[0] = -1; assert(vf_qpn2_moe_validate(&p) == VF_QPN2_INDEX); ++checks;
  slots[0] = 1; assert(vf_qpn2_moe_validate(&p) == VF_QPN2_COVERAGE); ++checks;
  slots[0] = 0;
  slots[8] = 8; assert(vf_qpn2_moe_validate(&p) == VF_QPN2_COVERAGE); ++checks;
  slots[8] = 4;
  assert(vf_qpn2_moe_validate(&p) == VF_QPN2_OK); ++checks;
  BAD(y, (uint16_t *)(uintptr_t)0x40000, VF_QPN2_POINTER);
  BAD(y, (uint16_t *)(uintptr_t)0x40010, VF_QPN2_POINTER);
  BAD(y, (uint16_t *)(uintptr_t)0x107fe, VF_QPN2_POINTER);
  BAD(y, (uint16_t *)(uintptr_t)0x20000, VF_QPN2_POINTER);
  BAD(y, (uint16_t *)(uintptr_t)0x30000, VF_QPN2_POINTER);
  BAD(y, (uint16_t *)(uintptr_t)0x60000, VF_QPN2_POINTER);
  BAD(y, (uint16_t *)(uintptr_t)0x70000, VF_QPN2_POINTER);
  BAD(y, (uint16_t *)(uintptr_t)0x80000, VF_QPN2_POINTER);
  BAD(y, (uint16_t *)(uintptr_t)(UINTPTR_MAX - 255), VF_QPN2_POINTER);
  BAD(x, (const uint16_t *)(uintptr_t)(UINTPTR_MAX - 15), VF_QPN2_POINTER);
  p = good(); p.y = (uint16_t *)(uintptr_t)0x10800;
  assert(vf_qpn2_moe_validate(&p) == VF_QPN2_OK); ++checks;
  /* A42 C clause sweep: each case breaks one check only. With that check disabled,
   * the zero, unaligned and non-dividing cases were accepted, and the oversized ones
   * were refused only later for another reason. */
  BAD(n, 0, VF_QPN2_DIMENSION);
  BAD(n, 65568, VF_QPN2_DIMENSION);
  BAD(k, 0, VF_QPN2_DIMENSION);
  BAD(k, 65600, VF_QPN2_DIMENSION);
  p = good(); p.slots = 0; p.input_rows = 0;
  assert(vf_qpn2_moe_validate(&p) == VF_QPN2_DIMENSION); ++checks;
  p = good(); p.slots = 262146; p.input_rows = 131073;
  assert(vf_qpn2_moe_validate(&p) == VF_QPN2_DIMENSION); ++checks;
  p = good(); p.in_div = 65; p.slots = 130; p.input_rows = 2;
  assert(vf_qpn2_moe_validate(&p) == VF_QPN2_DIMENSION); ++checks;
  p = good(); p.in_div = 3; p.input_rows = 2;
  assert(vf_qpn2_moe_validate(&p) == VF_QPN2_DIMENSION); ++checks;
  p = good(); p.block_expert_count = 0; p.block_expert_host_count = 0;
  p.sorted_slots_count = 0; p.sorted_slots_host_count = 0; p.ntpp_host = 0;
  assert(vf_qpn2_moe_validate(&p) == VF_QPN2_DIMENSION); ++checks;
  /* The host table is only read up to ntpp_host, so a longer declared device
   * table is the only thing wrong here. */
  p = good(); p.sorted_slots_count = 24; p.sorted_slots_host_count = 24;
  assert(vf_qpn2_moe_validate(&p) == VF_QPN2_LENGTH); ++checks;
  BAD(block_expert_host, (const int32_t *)(uintptr_t)((uintptr_t)experts + 1),
      VF_QPN2_POINTER);
  BAD(sorted_slots_host, (const int32_t *)(uintptr_t)((uintptr_t)slots + 2),
      VF_QPN2_POINTER);
  printf("CPU_VALIDATE checks=%u PASS (no CUDA runtime/device calls)\n", checks);
  return 0;
}
