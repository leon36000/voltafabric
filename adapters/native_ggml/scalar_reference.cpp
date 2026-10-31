// Diagnostic adaptation of R590's scalar oracle; GGML formats and quantizers from
// llama.cpp GGML 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT).
#include "scalar_reference.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>

namespace vf::native_ggml {
namespace {
constexpr int H=2560, F=640;
std::once_flag init;
bool fail(std::string& error, const char* message) { error=message; return false; }
bool bounded(float value) {
    return std::isfinite(value) && std::abs(value)<=1e12f &&
        (value==0.f || std::abs(value)>=std::numeric_limits<float>::min());
}
double scalar_dot(const std::vector<float>& a, const std::vector<float>& b) {
    double sum=0; for(size_t i=0;i<a.size();++i) sum+=double(a[i])*double(b[i]); return sum;
}
}
bool scalar_reference_view(int gate_up_type, int down_type, const uint8_t* blob, size_t blob_size,
                           const float* input, size_t input_size, std::vector<double>& output,
                           std::string& error) {
    error.clear();
    const uint32_t endian=1;
    if (*reinterpret_cast<const uint8_t*>(&endian)!=1 || sizeof(float)!=4 || sizeof(double)!=8 ||
        !std::numeric_limits<float>::is_iec559 || !std::numeric_limits<double>::is_iec559 ||
        std::fegetround()!=FE_TONEAREST)
        return fail(error,"unsupported scalar arithmetic environment");
    if ((gate_up_type!=18 && gate_up_type!=21 && gate_up_type!=22 && gate_up_type!=23) ||
        (down_type!=20 && down_type!=42) || (gate_up_type==23 && down_type==42))
        return fail(error,"unsupported scalar format pair");
    if (!blob || !input || input_size!=H) return fail(error,"invalid scalar input geometry");
    for(size_t i=0;i<input_size;++i) if(!bounded(input[i])) return fail(error,"invalid scalar input value");
    std::call_once(init, []{ggml_cpu_init();});
    const auto gt=static_cast<ggml_type>(gate_up_type), dt=static_cast<ggml_type>(down_type);
    const auto *g=ggml_get_type_traits_cpu(gt), *d=ggml_get_type_traits_cpu(dt);
    const auto *gc=ggml_get_type_traits(gt), *dc=ggml_get_type_traits(dt);
    if(!g||!d||!gc||!dc||!gc->to_float||!dc->to_float||g->vec_dot_type!=GGML_TYPE_Q8_K||d->vec_dot_type!=GGML_TYPE_Q8_0)
        return fail(error,"unsupported scalar GGML traits");
    const auto *gq=ggml_get_type_traits_cpu(g->vec_dot_type), *dq=ggml_get_type_traits_cpu(d->vec_dot_type);
    const auto *dqc=ggml_get_type_traits(d->vec_dot_type);
    if(!gq||!dq||!dqc||!gq->from_float||!dq->from_float||!dqc->to_float)
        return fail(error,"missing scalar GGML quantizer");
    const size_t gr=ggml_row_size(gt,H), dr=ggml_row_size(dt,F), up=gr*F, down=2*up;
    if(down+dr*H!=blob_size) return fail(error,"invalid scalar blob geometry");
    std::vector<uint8_t> qx(ggml_row_size(g->vec_dot_type,H)), qm(ggml_row_size(d->vec_dot_type,F));
    if(qx.size()!=2920 || qm.size()!=680) return fail(error,"invalid scalar Q8 geometry");
    std::vector<float> act(H),row(H),mid(F),weights(F),mact(F);
    std::vector<double> staged(H);
    gq->from_float(input,qx.data(),H);
    for(size_t b=0;b<10;++b) {
        float scale; std::memcpy(&scale,qx.data()+b*292,4);
        if(!std::isfinite(scale)) return fail(error,"invalid scalar activation scale");
        for(size_t j=0;j<256;++j) {int8_t code;std::memcpy(&code,qx.data()+b*292+4+j,1);act[b*256+j]=scale*code;}
    }
    for(int i=0;i<F;++i) {
        gc->to_float(blob+size_t(i)*gr,row.data(),H);
        for(float v:row) if(!std::isfinite(v)) return fail(error,"nonfinite scalar gate weight");
        const float gate=float(scalar_dot(row,act));
        gc->to_float(blob+up+size_t(i)*gr,row.data(),H);
        for(float v:row) if(!std::isfinite(v)) return fail(error,"nonfinite scalar up weight");
        const float value=float(scalar_dot(row,act));
        mid[i]=(gate/(1.f+std::exp(-gate)))*value;
        if(!bounded(mid[i])) return fail(error,"invalid scalar intermediate");
    }
    dq->from_float(mid.data(),qm.data(),F); dqc->to_float(qm.data(),mact.data(),F);
    for(float v:mact) if(!std::isfinite(v)) return fail(error,"nonfinite scalar intermediate decode");
    for(int i=0;i<H;++i) {
        dc->to_float(blob+down+size_t(i)*dr,weights.data(),F);
        for(float v:weights) if(!std::isfinite(v)) return fail(error,"nonfinite scalar down weight");
        staged[i]=scalar_dot(weights,mact);
        if(!std::isfinite(staged[i])) return fail(error,"nonfinite scalar output");
    }
    output.swap(staged); return true;
}
bool scalar_reference(int gate_up_type, int down_type, const std::vector<uint8_t>& blob,
                      const std::vector<float>& input, std::vector<double>& output,
                      std::string& error) {
    return scalar_reference_view(gate_up_type,down_type,blob.data(),blob.size(),
                                 input.data(),input.size(),output,error);
}
bool scalar_reference_iq2s_iq4nl(const std::vector<uint8_t>& blob,
                                  const std::vector<float>& input,
                                  std::vector<double>& output,
                                  std::string& error) {
    return scalar_reference(22,20,blob,input,output,error);
}
}
