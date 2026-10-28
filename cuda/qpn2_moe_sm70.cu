// FFI host wrapper. Device core stays byte-identical to the attributed donor.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 700
#error "This isolated QPN2 boundary is qualified for compilation on SM70 only"
#endif
#include "qpn2_moe_ffi.h"
#include "qpn2_moe_core.cuh"
static_assert(sizeof(int) == sizeof(int32_t), "donor kernel requires 32-bit int");
static_assert(sizeof(half) == sizeof(uint16_t), "FP16 storage width");

extern "C" int vf_qpn2_moe_launch(const vf_qpn2_moe_params *p, void *stream,
                                  int32_t *cuda_error_out) {
  if (cuda_error_out) *cuda_error_out = 0;
  const int status = vf_qpn2_moe_validate(p);
  if (status != VF_QPN2_OK) return status;
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#define VF_LAUNCH(SP, NA) \
  skinny_nvfp4_qpn2_moe<SP, NA> \
    <<<dim3(p->n / 32, (unsigned)p->block_expert_count), dim3(32 * SP), \
       0, cuda_stream>>>(p->codes, p->scales, p->gscales, \
       reinterpret_cast<const half *>(p->x), \
       reinterpret_cast<const int *>(p->block_expert), \
       reinterpret_cast<const int *>(p->sorted_slots), \
       reinterpret_cast<const int *>(p->ntpp), reinterpret_cast<half *>(p->y), \
       (int)p->n, (int)p->k, (int)p->slots, (int)p->in_div)
  switch (p->splitk * 10 + p->nacc) {
    case 21: VF_LAUNCH(2, 1); break;
    case 41: VF_LAUNCH(4, 1); break;
    case 81: VF_LAUNCH(8, 1); break;
    case 82: VF_LAUNCH(8, 2); break;
    case 161: VF_LAUNCH(16, 1); break;
    case 162: VF_LAUNCH(16, 2); break;
    default: return VF_QPN2_DIMENSION;
  }
#undef VF_LAUNCH
  const cudaError_t error = cudaPeekAtLastError();
  if (cuda_error_out) *cuda_error_out = (int32_t)error;
  return error == cudaSuccess ? VF_QPN2_OK : VF_QPN2_CUDA;
}
