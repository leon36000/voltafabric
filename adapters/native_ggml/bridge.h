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
// Opaque, batch-local prepared activation. Input and format are bound at creation.
void* vf_native_prepare(int gu, const float* input, size_t input_len);
int vf_native_forward_prepared(void* instance, const void* prepared, float* output, size_t output_len);
void vf_native_prepared_destroy(void* prepared);
// Owns one bounded GGML matrix; authentication and identity remain caller responsibilities.
void* vf_native_matrix_create(int type, size_t rows, size_t columns, const uint8_t* bytes, size_t len);
int vf_native_matrix_forward(void* instance, const float* input, size_t input_len, float* output, size_t output_len);
void vf_native_matrix_destroy(void* instance);
// Owns one bounded stack of same-format expert matrices; weights stay resident
// and every forward call preserves slot order and returns unweighted FP32 rows.
void* vf_native_routed_matrix_create(int type, size_t experts, size_t rows, size_t columns,
                                     const uint8_t* bytes, size_t len);
int vf_native_routed_matrix_forward(void* instance, const float* slot_inputs,
                                    size_t slot_input_len, const int32_t* expert_ids,
                                    size_t slots, float* slot_outputs, size_t output_len);
void vf_native_routed_matrix_destroy(void* instance);
#ifdef __cplusplus
}
#endif
#endif
