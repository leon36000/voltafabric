#ifndef VF_NATIVE_GGML_BRIDGE_H
#define VF_NATIVE_GGML_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// An instance owns an immutable copy of one native expert. Caller authenticates bytes.
// No function retries or applies routing weights. Each instance requires exclusive access.
void* vf_native_create(int gu, int down, const uint8_t* bytes, size_t len);
int vf_native_forward(void* instance, const float* input, size_t input_len, float* output, size_t output_len);
void vf_native_destroy(void* instance);
// Owns one bounded GGML matrix; authentication and identity remain caller responsibilities.
void* vf_native_matrix_create(int type, size_t rows, size_t columns, const uint8_t* bytes, size_t len);
int vf_native_matrix_forward(void* instance, const float* input, size_t input_len, float* output, size_t output_len);
void vf_native_matrix_destroy(void* instance);
#ifdef __cplusplus
}
#endif
#endif
