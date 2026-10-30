// Test-only AArch64 dispatch stubs: the pinned donor's x86 paths cannot run here.
// Any accidental dispatch into an unsupported x86 kernel terminates the test.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include <cstdlib>
namespace strata::kernels::cpu {
bool cpu_avx2_ok() { return false; }
bool cpu_avx512_ok() { return false; }
bool cpu_gather_fast() { return false; }
int iq256_gather_setting() { return 0; }
void q8k_quant_avx2(const float*,void*,int64_t) { std::abort(); }
void iq256_gu_rows(int,const uint8_t*,size_t,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
void iq512_gu_rows(int,const uint8_t*,size_t,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
void kq256_gu_rows(int,const uint8_t*,size_t,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
void kq256_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
void iq4nl256_down_rows(const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
}
