// Local CPU projection probe against a pinned IQ4_NL slice. No worker session.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    constexpr int hidden = 2560, mid = 640;
    constexpr std::streamoff offset = 2390092800LL;
    constexpr size_t blob_size = 1971200;
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<uint8_t> blob(blob_size);
    if (!in.seekg(offset) || !in.read(reinterpret_cast<char*>(blob.data()),blob.size())) return 2;
    ggml_cpu_init();
    auto down = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
    auto quant = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (!down || !down->vec_dot || down->vec_dot_type != GGML_TYPE_Q8_0 || !quant || !quant->from_float) return 3;
    const size_t gate_row = ggml_row_size(GGML_TYPE_IQ2_S,hidden);
    const size_t down_row = ggml_row_size(GGML_TYPE_IQ4_NL,mid);
    const size_t down_offset = 2*gate_row*mid;
    if (down_offset + down_row*hidden != blob.size() || down_row != 360) return 4;
    auto write = [&](const std::string &name, const void *data, size_t n) {
        std::ofstream out(std::string(argv[2])+name, std::ios::binary | std::ios::out);
        return bool(out.write(static_cast<const char*>(data),n));
    };
    if (!write("-weights.bin",blob.data()+down_offset,down_row*hidden)) return 5;
    for (const std::string mode : {"smooth", "boundary"}) {
        std::vector<float> input(mid), output(hidden);
        const float values[] = {127.f,0.5f,-0.5f,1.5f,-1.5f,2.5f,-2.5f,126.5f,-126.5f};
        for (int i=0;i<mid;++i) input[i] = mode == "smooth" ? std::sin(float(i)*0.17f)/32.f : (i%32<9 ? values[i%32] : 0.f);
        std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_Q8_0,mid));
        quant->from_float(input.data(),packed.data(),mid);
        for (int row=0;row<hidden;++row)
            down->vec_dot(mid,&output[row],0,blob.data()+down_offset+row*down_row,0,packed.data(),0,1);
        if (!write("-"+mode+"-input.f32",input.data(),input.size()*sizeof(float)) ||
            !write("-"+mode+"-quant.bin",packed.data(),packed.size()) ||
            !write("-"+mode+"-output.f32",output.data(),output.size()*sizeof(float))) return 6;
    }
    return 0;
}
