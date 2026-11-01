#include "bridge.h"
#include "provider.hpp"
#include "ggml.h"
#include <algorithm>
#include <memory>
#include <vector>
namespace {
struct Resident { int gu; int down; std::vector<uint8_t> weights; };
struct Matrix { int type; size_t rows, columns; std::vector<uint8_t> weights; };
struct RoutedMatrix { int type; size_t experts, rows, columns; std::vector<uint8_t> weights; };
}
extern "C" void* vf_native_create(int gu, int down, const uint8_t* bytes, size_t len) {
    try {
        if (!bytes || (gu != 18 && gu != 21 && gu != 22 && gu != 23) || (down != 20 && down != 42)) return nullptr;
        const size_t expected = 2 * ggml_row_size(static_cast<ggml_type>(gu), 2560) * 640
                              + ggml_row_size(static_cast<ggml_type>(down), 640) * 2560;
        if (len != expected || len > 4 * 1024 * 1024) return nullptr;
        auto result = std::make_unique<Resident>(); result->gu=gu; result->down=down;
        result->weights.assign(bytes,bytes+len); return result.release();
    } catch (...) { return nullptr; }
}
extern "C" int vf_native_forward(void* instance, const float* input, size_t input_len, float* output, size_t output_len) {
    try {
        if (!instance || !input || !output || input_len != 2560 || output_len != 2560) return -1;
        auto* r=static_cast<Resident*>(instance);
        std::vector<float> x(input,input+input_len), staged; std::string error;
        if (!vf::native_ggml::forward(r->gu,r->down,r->weights,x,staged,error)) return -1;
        std::copy(staged.begin(),staged.end(),output); return 0;
    } catch (...) { return -1; }
}
extern "C" void vf_native_destroy(void* instance) { delete static_cast<Resident*>(instance); }
extern "C" void* vf_native_prepare(int gu, const float* input, size_t input_len) {
    try {
        if (!input || input_len != 2560) return nullptr;
        auto prepared=std::make_unique<vf::native_ggml::PreparedInput>();
        std::vector<float> x(input,input+input_len); std::string error;
        if (!vf::native_ggml::prepare_input(gu,x,*prepared,error)) return nullptr;
        return prepared.release();
    } catch (...) { return nullptr; }
}
extern "C" int vf_native_forward_prepared(void* instance, const void* prepared, float* output, size_t output_len) {
    try {
        if (!instance || !prepared || !output || output_len != 2560) return -1;
        auto* r=static_cast<Resident*>(instance);
        const auto* x=static_cast<const vf::native_ggml::PreparedInput*>(prepared);
        std::vector<float> staged; std::string error;
        if (!vf::native_ggml::forward_prepared(r->gu,r->down,r->weights,*x,staged,error)) return -1;
        std::copy(staged.begin(),staged.end(),output); return 0;
    } catch (...) { return -1; }
}
extern "C" void vf_native_prepared_destroy(void* prepared) { delete static_cast<vf::native_ggml::PreparedInput*>(prepared); }


extern "C" void* vf_native_matrix_create(int type,size_t rows,size_t columns,const uint8_t* bytes,size_t len) {
    try {
        if(!bytes || !vf::native_ggml::matrix_shape(type,rows,columns,len)) return nullptr;
        auto matrix=std::make_unique<Matrix>(); matrix->type=type; matrix->rows=rows; matrix->columns=columns;
        matrix->weights.assign(bytes,bytes+len); return matrix.release();
    } catch (...) { return nullptr; }
}
extern "C" int vf_native_matrix_forward(void* instance,const float* input,size_t input_len,float* output,size_t output_len) {
    try {
        if(!instance || !input || !output) return -1;
        auto* matrix=static_cast<Matrix*>(instance);
        if(input_len!=matrix->columns || output_len!=matrix->rows) return -1;
        std::vector<float> x(input,input+input_len),staged; std::string error;
        if(!vf::native_ggml::matrix_forward(matrix->type,matrix->rows,matrix->columns,matrix->weights,x,staged,error)) return -1;
        std::copy(staged.begin(),staged.end(),output); return 0;
    } catch (...) { return -1; }
}
extern "C" void vf_native_matrix_destroy(void* instance) { delete static_cast<Matrix*>(instance); }

extern "C" void* vf_native_routed_matrix_create(int type, size_t experts, size_t rows,
                                                  size_t columns, const uint8_t* bytes,
                                                  size_t len) {
    try {
        if (!bytes || (type != 8 && type != 12 && type != 13 && type != 14 && type != 23) ||
            experts == 0 || experts > 512 || rows == 0 || rows > 10240 ||
            columns == 0 || columns > 6144) return nullptr;
        const auto ggml_type_value = static_cast<ggml_type>(type);
        const size_t block = ggml_blck_size(ggml_type_value);
        if (block == 0 || columns % block != 0) return nullptr;
        const size_t row_bytes = ggml_row_size(ggml_type_value, columns);
        if (row_bytes == 0 || rows > SIZE_MAX / row_bytes) return nullptr;
        const size_t expert_bytes = rows * row_bytes;
        if (experts > SIZE_MAX / expert_bytes || experts * expert_bytes != len ||
            len > 1'250'000'000ULL) return nullptr;
        auto matrix = std::make_unique<RoutedMatrix>();
        matrix->type = type;
        matrix->experts = experts;
        matrix->rows = rows;
        matrix->columns = columns;
        matrix->weights.assign(bytes, bytes + len);
        return matrix.release();
    } catch (...) { return nullptr; }
}

extern "C" int vf_native_routed_matrix_forward(void* instance, const float* slot_inputs,
                                                 size_t slot_input_len,
                                                 const int32_t* expert_ids, size_t slots,
                                                 float* slot_outputs, size_t output_len) {
    try {
        if (!instance || !slot_inputs || !expert_ids || !slot_outputs || slots == 0 || slots > 512)
            return -1;
        auto* matrix = static_cast<RoutedMatrix*>(instance);
        if (slots > SIZE_MAX / matrix->columns || slot_input_len != slots * matrix->columns ||
            slots > SIZE_MAX / matrix->rows || output_len != slots * matrix->rows)
            return -1;
        std::vector<float> inputs(slot_inputs, slot_inputs + slot_input_len);
        std::vector<int32_t> ids(expert_ids, expert_ids + slots);
        std::vector<float> staged;
        std::string error;
        if (!vf::native_ggml::routed_matrix_forward(matrix->type, matrix->experts,
                matrix->rows, matrix->columns, matrix->weights, inputs, ids, staged, error) ||
            staged.size() != output_len) return -1;
        std::copy(staged.begin(), staged.end(), slot_outputs);
        return 0;
    } catch (...) { return -1; }
}

extern "C" void vf_native_routed_matrix_destroy(void* instance) {
    delete static_cast<RoutedMatrix*>(instance);
}
