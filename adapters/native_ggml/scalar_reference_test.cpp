#include "scalar_reference.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
int main() {
    constexpr size_t H=2560, BLOB=1971200;
    std::vector<uint8_t> blob(BLOB,0);
    std::vector<float> input(H,0.f);
    std::vector<double> output(3,7.0);
    std::string error;
    if (!vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output.size()!=H) return 1;
    for(double value:output) if(value!=0.0 || !std::isfinite(value)) return 2;
    const auto saved=output;
    input[0]=std::numeric_limits<float>::quiet_NaN();
    if(vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output!=saved) return 3;
    input[0]=0;input.pop_back();
    if(vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output!=saved) return 4;
    input.push_back(0);blob.pop_back();
    if(vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output!=saved) return 5;
    blob.push_back(0);
    input[0]=std::numeric_limits<float>::infinity();
    if(vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output!=saved) return 6;
    input[0]=0;
    if(std::fesetround(FE_DOWNWARD)!=0) return 7;
    bool rejected=!vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error);
    if(std::fesetround(FE_TONEAREST)!=0 || !rejected || output!=saved) return 8;
    blob[0]=0;blob[1]=0x7c; // Non-finite half scale in the first IQ2_S block.
    if(vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output!=saved) return 9;
    const int pairs[][2]={{18,20},{18,42},{21,20},{21,42},{22,20},{22,42},{23,20}};
    for (const auto& pair:pairs) {
        const auto gate=static_cast<ggml_type>(pair[0]), down=static_cast<ggml_type>(pair[1]);
        const size_t size=2*ggml_row_size(gate,H)*640+ggml_row_size(down,640)*H;
        std::vector<uint8_t> weights(size,0);
        std::vector<double> value(3,7.0);
        if(!vf::native_ggml::scalar_reference(pair[0],pair[1],weights,input,value,error) || value.size()!=H) return 10;
        for(double v:value) if(v!=0.0 || !std::isfinite(v)) return 11;
        const auto before=value;
        if(vf::native_ggml::scalar_reference(pair[0],39,weights,input,value,error) || value!=before) return 12;
        weights.pop_back();
        if(vf::native_ggml::scalar_reference(pair[0],pair[1],weights,input,value,error) || value!=before) return 13;
    }
    if(vf::native_ggml::scalar_reference(23,42,blob,input,output,error) || output!=saved) return 14;
    ggml_cpu_init();
    for (const auto& pair:pairs) {
        constexpr size_t F=640;
        const auto gt=static_cast<ggml_type>(pair[0]),dt=static_cast<ggml_type>(pair[1]);
        const auto *g=ggml_get_type_traits_cpu(gt),*d=ggml_get_type_traits_cpu(dt);
        const auto *gc=ggml_get_type_traits(gt),*dc=ggml_get_type_traits(dt);
        const auto *gq=ggml_get_type_traits_cpu(g->vec_dot_type),*dq=ggml_get_type_traits_cpu(d->vec_dot_type);
        const auto *dqc=ggml_get_type_traits(d->vec_dot_type);
        const size_t gr=ggml_row_size(gt,H),dr=ggml_row_size(dt,F),up=gr*F,down=2*up;
        std::vector<uint8_t> weights(down+dr*H,0);
        for(size_t i=0;i<down;i+=ggml_type_size(gt)) weights[i+1]=0x28;
        for(size_t i=down;i<weights.size();i+=ggml_type_size(dt)) weights[i+1]=0x28;
        std::vector<float> x(H),decoded(H),activation(H),mid(F),out_act(F),down_row(F);
        for(size_t i=0;i<H;++i) x[i]=std::sin(float(i)*0.17f)/16.f;
        std::vector<uint8_t> qx(ggml_row_size(g->vec_dot_type,H)),qm(ggml_row_size(d->vec_dot_type,F));
        gq->from_float(x.data(),qx.data(),H);
        if(g->vec_dot_type!=GGML_TYPE_Q8_K || qx.size()!=2920) return 15;
        for(size_t b=0;b<10;++b) {
            float scale;std::memcpy(&scale,qx.data()+b*292,4);
            for(size_t j=0;j<256;++j) {int8_t code;std::memcpy(&code,qx.data()+b*292+4+j,1);activation[b*256+j]=scale*code;}
        }
        gc->to_float(weights.data(),decoded.data(),H);
        double dot=0;for(size_t i=0;i<H;++i) dot+=double(decoded[i])*double(activation[i]);
        const float gate=float(dot),activated=(gate/(1.f+std::exp(-gate)))*gate;
        std::fill(mid.begin(),mid.end(),activated);
        dq->from_float(mid.data(),qm.data(),F);dqc->to_float(qm.data(),out_act.data(),F);
        dc->to_float(weights.data()+down,down_row.data(),F);
        double expected=0;for(size_t i=0;i<F;++i) expected+=double(down_row[i])*double(out_act[i]);
        std::vector<double> actual;
        if(!std::isfinite(expected) || std::abs(expected)<1e-9 ||
           !vf::native_ggml::scalar_reference(pair[0],pair[1],weights,x,actual,error)) return 16;
        for(double v:actual) if(v!=expected) return 17;
    }
    return 0;
}
