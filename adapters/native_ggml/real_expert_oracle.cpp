// Local CPU diagnostic: reference uses GGML quantization and dequantization,
// but double scalar dot products are independent of GGML's vector dot kernels.
#include "provider.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void decode_q8k(const std::vector<uint8_t>& packed, std::vector<float>& values) {
    for (size_t i=0;i<values.size()/256;++i) {
        float scale; std::memcpy(&scale,packed.data()+i*292,sizeof(scale));
        for (size_t j=0;j<256;++j) {
            int8_t q; std::memcpy(&q,packed.data()+i*292+4+j,1);
            values[i*256+j]=scale*q;
        }
    }
}
static double dot(const std::vector<float>& a,const std::vector<float>& b) {
    double sum=0; for(size_t i=0;i<a.size();++i) sum+=double(a[i])*b[i]; return sum;
}
int main(int argc,char** argv) {
    if(argc!=3 && argc!=4) { std::fprintf(stderr,"usage: real_expert_oracle PACK EXPERT_0_OR_1 [REFERENCE_FILE]\n"); return 1; }
    std::string id=argv[2]; if(id!="0" && id!="1") return 2;
    constexpr int H=2560,F=640; constexpr size_t BLOB=2176000;
    std::ifstream file(argv[1],std::ios::binary); std::vector<uint8_t> blob(BLOB);
    if(!file || !file.seekg(id=="0"?0:BLOB) || !file.read(reinterpret_cast<char*>(blob.data()),BLOB)) return 3;
    ggml_cpu_init();
    auto g=ggml_get_type_traits_cpu(GGML_TYPE_IQ3_XXS), d=ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
    auto gc=ggml_get_type_traits(GGML_TYPE_IQ3_XXS), dc=ggml_get_type_traits(GGML_TYPE_IQ4_NL);
    auto ga=ggml_get_type_traits_cpu(g->vec_dot_type), da=ggml_get_type_traits_cpu(d->vec_dot_type);
    auto dac=ggml_get_type_traits(d->vec_dot_type);
    if(!gc->to_float || !dc->to_float || !ga->from_float || !da->from_float || !dac->to_float || g->vec_dot_type!=GGML_TYPE_Q8_K) return 4;
    const size_t gr=ggml_row_size(GGML_TYPE_IQ3_XXS,H), dr=ggml_row_size(GGML_TYPE_IQ4_NL,F), up=gr*F, down=2*up;
    if(down+dr*H!=BLOB) return 5;
    std::vector<float> x(H), output, ax(H), row(H), mid(F), dw(F), am(F);
    std::vector<double> reference(H);
    for(int i=0;i<H;++i) x[i]=std::sin(float(i)*0.17f)/32.f;
    std::string error;
    if(!vf::native_ggml::forward(18,20,blob,x,output,error)) {std::fprintf(stderr,"provider: %s\n",error.c_str());return 6;}
    std::vector<uint8_t> qx(ggml_row_size(g->vec_dot_type,H)), qmid(ggml_row_size(d->vec_dot_type,F));
    ga->from_float(x.data(),qx.data(),H); decode_q8k(qx,ax);
    for(int i=0;i<F;++i) {
        gc->to_float(blob.data()+size_t(i)*gr,row.data(),H); float gate=float(dot(row,ax));
        gc->to_float(blob.data()+up+size_t(i)*gr,row.data(),H); float value=float(dot(row,ax));
        mid[i]=(gate/(1.f+std::exp(-gate)))*value;
        if(!std::isfinite(mid[i])) return 7;
    }
    da->from_float(mid.data(),qmid.data(),F); dac->to_float(qmid.data(),am.data(),F);
    double max_abs=0, max_expected=0, max_relative=0; int nonzero=0, violations=0;
    for(int i=0;i<H;++i) {
        dc->to_float(blob.data()+down+size_t(i)*dr,dw.data(),F); double want=dot(dw,am);
        if(!std::isfinite(want)||!std::isfinite(output[i])) return 8;
        reference[i]=want;
        double diff=std::abs(double(output[i])-want);
        max_abs=std::max(max_abs,diff); max_expected=std::max(max_expected,std::abs(want));
        max_relative=std::max(max_relative,diff/std::max(1e-12,std::abs(want)));
        violations+=diff>1e-7+1e-3*std::abs(want);
        nonzero+=output[i]!=0.f;
    }
    std::printf("expert=%s rows=%d nonzero=%d max_abs=%.9g max_expected=%.9g max_relative=%.9g violations=%d tolerance=1e-7+1e-3*abs(reference)\n",id.c_str(),H,nonzero,max_abs,max_expected,max_relative,violations);
    if(nonzero==0 || violations!=0) return 9;
    if(argc==4) {
        const uint32_t endian=1;
        if(*reinterpret_cast<const uint8_t*>(&endian)!=1) return 10;
        std::ofstream ref(argv[3],std::ios::binary|std::ios::trunc);
        if(!ref.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(float)) ||
           !ref.write(reinterpret_cast<const char*>(reference.data()),reference.size()*sizeof(double))) return 11;
    }
    return 0;
}
