#include "scalar_reference.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Diagnostic ABI: input buffers remain caller-owned; output is copied only on success.
extern "C" int vf_native_scalar_reference(int gu,int down,const uint8_t* blob,size_t blob_len,
                                           const float* input,size_t input_len,
                                           double* output,size_t output_len) {
    try {
        if(!blob || !input || !output || blob_len==0 || blob_len>4*1024*1024 || input_len!=2560 || output_len!=2560) return -1;
        const std::vector<uint8_t> weights(blob,blob+blob_len);
        const std::vector<float> values(input,input+input_len);
        std::vector<double> staged;std::string error;
        if(!vf::native_ggml::scalar_reference(gu,down,weights,values,staged,error) || staged.size()!=output_len) return -1;
        std::copy(staged.begin(),staged.end(),output);return 0;
    } catch (...) { return -1; }
}
