// Local CPU quantizer probe. The GGML libraries and formats are provided by
// the pinned Strata/llama.cpp build (MIT); this diagnostic adds no GPU work.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    std::vector<float> x(2560);
    std::ifstream in(argv[1], std::ios::binary);
    if (!in.read(reinterpret_cast<char *>(x.data()), x.size() * sizeof(float)) || in.peek() != EOF) return 2;
    for (float v : x) if (!std::isfinite(v)) return 3;
    ggml_cpu_init();
    auto q8 = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
    auto qk = ggml_get_type_traits_cpu(GGML_TYPE_IQ2_S);
    if (!q8 || !qk || q8->vec_dot_type != GGML_TYPE_Q8_0 || qk->vec_dot_type != GGML_TYPE_Q8_K) return 4;
    auto fast8 = ggml_get_type_traits_cpu(q8->vec_dot_type);
    auto fastk = ggml_get_type_traits_cpu(qk->vec_dot_type);
    if (!fast8 || !fastk || !fast8->from_float || !fastk->from_float) return 5;
    std::vector<block_q8_0> a(x.size()/32), ar(x.size()/32);
    std::vector<block_q8_K> b(x.size()/256), br(x.size()/256);
    fast8->from_float(x.data(),a.data(),x.size());
    fastk->from_float(x.data(),b.data(),x.size());
    quantize_row_q8_0_ref(x.data(),ar.data(),x.size());
    quantize_row_q8_K_ref(x.data(),br.data(),x.size());
    auto write = [&](const char *name,const void *data,size_t n) {
        std::ofstream out(std::string(argv[2])+name,std::ios::binary | std::ios::out);
        return bool(out.write(static_cast<const char *>(data),n));
    };
    return write("-q80-fast.bin",a.data(),a.size()*sizeof(a[0])) &&
           write("-q80-ref.bin",ar.data(),ar.size()*sizeof(ar[0])) &&
           write("-q8k-fast.bin",b.data(),b.size()*sizeof(b[0])) &&
           write("-q8k-ref.bin",br.data(),br.size()*sizeof(br[0])) ? 0 : 6;
}
