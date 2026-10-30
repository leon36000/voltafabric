#include "provider.hpp"
#include "bridge.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static bool close(float a, double b) { return std::abs(a-b) <= 1e-3 * std::max(1.0, std::abs(b)); }
static void decode_q8k(const std::vector<uint8_t>& packed, std::vector<float>& values) {
    constexpr size_t count=256, block=292;
    for (size_t i=0;i<values.size()/count;++i) {
        float scale; std::memcpy(&scale,packed.data()+i*block,sizeof(scale));
        for (size_t j=0;j<count;++j) {
            int8_t q; std::memcpy(&q,packed.data()+i*block+4+j,1);
            values[i*count+j]=scale*q;
        }
    }
}
static bool check(int gu, int down) {
    constexpr int H=2560,F=640;
    auto gt=static_cast<ggml_type>(gu), dt=static_cast<ggml_type>(down);
    auto g=ggml_get_type_traits_cpu(gt), d=ggml_get_type_traits_cpu(dt);
    auto ga=ggml_get_type_traits_cpu(g->vec_dot_type), da=ggml_get_type_traits_cpu(d->vec_dot_type);
    const auto* gconv=ggml_get_type_traits(gt); const auto* dconv=ggml_get_type_traits(dt);
    const auto* gac=ggml_get_type_traits(g->vec_dot_type); const auto* dac=ggml_get_type_traits(d->vec_dot_type);
    if (!gconv->to_float || !dconv->to_float || (!gac->to_float && g->vec_dot_type != GGML_TYPE_Q8_K) || !dac->to_float) return false;
    size_t gr=ggml_row_size(gt,H), dr=ggml_row_size(dt,F), up=gr*F, dn=2*up;
    std::vector<uint8_t> blob(dn+dr*H,0);
    size_t gs=ggml_type_size(gt), ds=ggml_type_size(dt);
    // Nonzero finite half scale 1/32 for every synthetic quant block.
    for (size_t at=0;at<dn;at+=gs) { blob[at]=0; blob[at+1]=0x28; }
    for (size_t at=dn;at<blob.size();at+=ds) { blob[at]=0; blob[at+1]=0x28; }
    std::vector<float> x(H), out;
    for (int i=0;i<H;++i) x[i]=std::sin(i*0.17f)/16.f;
    std::string err;
    if (!vf::native_ggml::forward(gu,down,blob,x,out,err)) { std::fprintf(stderr,"forward: %s\n",err.c_str());return false; }
    std::vector<uint8_t> qx(ggml_row_size(g->vec_dot_type,H)); ga->from_float(x.data(),qx.data(),H);
    std::vector<float> gw(H), ax(H); gconv->to_float(blob.data(),gw.data(),H);
    if (g->vec_dot_type == GGML_TYPE_Q8_K) decode_q8k(qx,ax); else gac->to_float(qx.data(),ax.data(),H);
    double gate=0; for (int i=0;i<H;++i) gate+=(double)gw[i]*ax[i];
    const float gg=(float)gate; const float mid=(gg/(1.f+std::exp(-gg)))*gg;
    std::vector<float> midrow(F,mid); std::vector<uint8_t> qmid(ggml_row_size(d->vec_dot_type,F)); da->from_float(midrow.data(),qmid.data(),F);
    std::vector<float> dw(F), am(F); dconv->to_float(blob.data()+dn,dw.data(),F); dac->to_float(qmid.data(),am.data(),F);
    double expected=0; for (int i=0;i<F;++i) expected+=(double)dw[i]*am[i];
    if (!std::isfinite(expected) || std::abs(expected) < 1e-9 || out.size()!=H) return false;
    for (float value:out) if (!close(value,expected)) { std::fprintf(stderr,"types %d/%d got %.9g expected %.9g\n",gu,down,value,expected);return false; }
    void* resident = vf_native_create(gu, down, blob.data(), blob.size());
    if (!resident) return false;
    std::vector<float> bridged(H,99.f);
    if (vf_native_forward(resident,x.data(),x.size(),bridged.data(),bridged.size()) || bridged != out) { vf_native_destroy(resident); return false; }
    std::fill(blob.begin(),blob.end(),0);
    std::fill(bridged.begin(),bridged.end(),99.f);
    if (vf_native_forward(resident,x.data(),x.size(),bridged.data(),bridged.size()) || bridged != out) { vf_native_destroy(resident); return false; }
    vf_native_destroy(resident);
    auto original=out; blob.pop_back();
    if (vf::native_ggml::forward(gu,down,blob,x,out,err) || out!=original) return false;
    return true;
}
int main() {
    ggml_cpu_init();
    std::vector<uint8_t> empty;
    std::vector<float> x(2560,0.f), output(5,7.f);
    std::string error;
    if (vf::native_ggml::forward(39,20,empty,x,output,error) || output!=std::vector<float>(5,7.f)) return 1;
    x.pop_back();
    if (vf::native_ggml::forward(18,20,empty,x,output,error) || output!=std::vector<float>(5,7.f)) return 1;
    x.push_back(0.f);
    x[0]=std::numeric_limits<float>::quiet_NaN();
    if (vf::native_ggml::forward(18,20,empty,x,output,error) || output!=std::vector<float>(5,7.f)) return 1;
    const int pairs[][2]={{18,20},{18,42},{21,20},{21,42},{22,20},{22,42},{23,20}};
    for (auto& pair:pairs) if (!check(pair[0],pair[1])) { std::fprintf(stderr,"failed pair %d/%d\n",pair[0],pair[1]);return 1; }
    std::puts("all seven native format pairs passed"); return 0;
}
