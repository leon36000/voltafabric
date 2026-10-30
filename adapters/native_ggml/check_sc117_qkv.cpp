// Diagnostic: selected scalar-dequantized rows of SC117 layer-0 Q6_K QKV.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=4) {std::fprintf(stderr,"usage: check_sc117_qkv SHARD INPUT QKV\n");return 1;}
    std::ifstream shard(argv[1],std::ios::binary|std::ios::ate),input(argv[2],std::ios::binary),output(argv[3],std::ios::binary|std::ios::ate);
    if(!shard || shard.tellg()!=std::streamoff(55063448064LL) || !input || !output || output.tellg()!=std::streamoff(10240*4)) return 2;
    std::vector<float> x(2560),actual(10240),ax(2560),row(2560); output.seekg(0);
    if(!input.read(reinterpret_cast<char*>(x.data()),x.size()*4) || !output.read(reinterpret_cast<char*>(actual.data()),actual.size()*4)) return 3;
    for(float v:actual) if(!std::isfinite(v)) return 4;
    ggml_cpu_init(); const auto* w=ggml_get_type_traits(GGML_TYPE_Q6_K); const auto* a=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if(!w || !w->to_float || !a || !a->from_float || ggml_row_size(GGML_TYPE_Q6_K,2560)!=2100) return 5;
    std::vector<uint8_t> qx(ggml_row_size(GGML_TYPE_Q8_K,2560)),packed(2100); a->from_float(x.data(),qx.data(),2560);
    for(size_t i=0;i<10;++i) {float scale;std::memcpy(&scale,qx.data()+i*292,4);for(size_t j=0;j<256;++j){int8_t q;std::memcpy(&q,qx.data()+i*292+4+j,1);ax[i*256+j]=scale*q;}}
    const int rows[]={0,1,127,2047,2048,4095,4096,8191,10239}; double max_abs=0;
    for(int index:rows) {
        shard.seekg(std::streamoff(892207104LL)+std::streamoff(index)*2100);
        if(!shard.read(reinterpret_cast<char*>(packed.data()),packed.size())) return 6;
        w->to_float(packed.data(),row.data(),2560); double expected=0;
        for(size_t j=0;j<row.size();++j) expected+=double(row[j])*ax[j];
        const double error=std::abs(double(actual[size_t(index)])-expected);max_abs=std::max(max_abs,error);
        if(!std::isfinite(expected) || error>1e-5+1e-3*std::abs(expected)) return 7;
    }
    std::printf("sampled_rows=%zu all_finite=%zu max_abs=%.9g tolerance=1e-5+1e-3*abs(reference)\n",sizeof(rows)/sizeof(rows[0]),actual.size(),max_abs);
    return 0;
}
