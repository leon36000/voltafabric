#include "bridge.h"
#include "provider.hpp"
#include "ggml.h"
#include <algorithm>
#include <memory>
#include <vector>
namespace {
struct Resident { int gu; int down; std::vector<uint8_t> weights; };
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
