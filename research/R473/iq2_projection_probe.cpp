// Local CPU IQ2_S projection probe against pinned SC117 expert 255; no worker.
#include "provider.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
int main(int argc,char **argv) {
    if (argc != 3) return 1;
    constexpr int hidden=2560, mid=640;
    constexpr std::streamoff offset=2390092800LL;
    constexpr size_t blob_size=1971200;
    std::ifstream in(argv[1],std::ios::binary);
    std::vector<uint8_t> blob(blob_size);
    if (!in.seekg(offset) || !in.read(reinterpret_cast<char*>(blob.data()),blob.size())) return 2;
    ggml_cpu_init();
    auto t=ggml_get_type_traits_cpu(GGML_TYPE_IQ2_S);
    auto q=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (!t || !t->vec_dot || t->vec_dot_type!=GGML_TYPE_Q8_K || !q || !q->from_float) return 3;
    const size_t row=ggml_row_size(GGML_TYPE_IQ2_S,hidden);
    const size_t down_row=ggml_row_size(GGML_TYPE_IQ4_NL,mid);
    if (row!=820 || 2*row*mid+down_row*hidden!=blob.size()) return 4;
    auto write=[&](const std::string &name,const void *data,size_t n) {
        std::ofstream out(std::string(argv[2])+name,std::ios::binary | std::ios::out);
        return bool(out.write(static_cast<const char*>(data),n));
    };
    for (const std::string mode : {"smooth", "boundary"}) {
        std::vector<float> input(hidden), gate(mid), up(mid);
        const float v[]={127.f,0.5f,-0.5f,1.5f,-1.5f,2.5f,-2.5f,126.5f,-126.5f};
        for (int i=0;i<hidden;++i) input[i] = mode=="smooth" ? std::sin(float(i)*0.17f)/32.f : (i%32<9 ? v[i%32] : 0.f);
        std::vector<float> expert;
        std::string error;
        if (!vf::native_ggml::forward(GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, blob, input, expert, error) || expert.size()!=hidden) return 6;
        std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_Q8_K,hidden));
        q->from_float(input.data(),packed.data(),hidden);
        for (int i=0;i<mid;++i) {
            t->vec_dot(hidden,&gate[i],0,blob.data()+i*row,0,packed.data(),0,1);
            t->vec_dot(hidden,&up[i],0,blob.data()+mid*row+i*row,0,packed.data(),0,1);
        }
        auto dt=ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
        auto mq=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if (!dt || !dt->vec_dot || !mq || !mq->from_float) return 7;
        std::vector<float> inter(mid), staged(hidden);
        for (int i=0;i<mid;++i) inter[i]=(gate[i]/(1.f+std::exp(-gate[i])))*up[i];
        std::vector<uint8_t> imq(ggml_row_size(GGML_TYPE_Q8_0,mid));
        mq->from_float(inter.data(),imq.data(),mid);
        for (int i=0;i<hidden;++i)
            dt->vec_dot(mid,&staged[i],0,blob.data()+2*row*mid+i*down_row,0,imq.data(),0,1);
        if (!write("-"+mode+"-intermediate.f32",inter.data(),inter.size()*sizeof(float)) ||
            !write("-"+mode+"-midquant.bin",imq.data(),imq.size()) ||
            !write("-"+mode+"-staged.f32",staged.data(),staged.size()*sizeof(float))) return 8;
        if (!write("-"+mode+"-input.f32",input.data(),input.size()*sizeof(float)) ||
            !write("-"+mode+"-quant.bin",packed.data(),packed.size()) ||
            !write("-"+mode+"-gate.f32",gate.data(),gate.size()*sizeof(float)) ||
            !write("-"+mode+"-up.f32",up.data(),up.size()*sizeof(float)) ||
            !write("-"+mode+"-expert.f32",expert.data(),expert.size()*sizeof(float))) return 5;
    }
    return 0;
}
