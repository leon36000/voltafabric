// Native SC117 expert CPU boundary, using the pinned ggml CPU type traits.
// Reference: Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf,
// src/kernels/cpu/native_expert.cpp (MIT); llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT).
#include "provider.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <mutex>

namespace vf::native_ggml {
namespace {
constexpr int H = 2560, F = 640;
std::once_flag init;
bool fail(std::string& error, const char* text) { error = text; return false; }
}
bool forward(int gu, int down, const std::vector<uint8_t>& blob,
             const std::vector<float>& input, std::vector<float>& output, std::string& error) {
    error.clear();
    if ((gu != 18 && gu != 21 && gu != 22 && gu != 23) || (down != 20 && down != 42))
        return fail(error, "unsupported native format");
    if (input.size() != H) return fail(error, "invalid activation shape");
    for (float x : input) if (!std::isfinite(x)) return fail(error, "non-finite activation");
    std::call_once(init, [] { ggml_cpu_init(); });
    auto gt = static_cast<ggml_type>(gu), dt = static_cast<ggml_type>(down);
    const auto* g = ggml_get_type_traits_cpu(gt);
    const auto* d = ggml_get_type_traits_cpu(dt);
    if (!g || !d || !g->vec_dot || !d->vec_dot) return fail(error, "missing ggml dot trait");
    const auto* ga = ggml_get_type_traits_cpu(g->vec_dot_type);
    const auto* da = ggml_get_type_traits_cpu(d->vec_dot_type);
    if (!ga || !da || !ga->from_float || !da->from_float) return fail(error, "missing ggml quantizer");
    if (H % ggml_blck_size(gt) || F % ggml_blck_size(dt) ||
        H % ggml_blck_size(g->vec_dot_type) || F % ggml_blck_size(d->vec_dot_type))
        return fail(error, "invalid ggml geometry");
    const size_t grow = ggml_row_size(gt, H), drow = ggml_row_size(dt, F);
    const size_t up = grow * F, dn = 2 * up, size = dn + drow * H;
    if (blob.size() != size) return fail(error, "invalid native blob length");
    std::vector<uint8_t> qx(ggml_row_size(g->vec_dot_type, H));
    std::vector<uint8_t> qmid(ggml_row_size(d->vec_dot_type, F));
    std::vector<float> mid(F), staged(H);
    ga->from_float(input.data(), qx.data(), H);
    for (int i = 0; i < F; ++i) {
        float gate = 0.f, v = 0.f;
        g->vec_dot(H, &gate, 0, blob.data() + i * grow, 0, qx.data(), 0, 1);
        g->vec_dot(H, &v, 0, blob.data() + up + i * grow, 0, qx.data(), 0, 1);
        mid[i] = (gate / (1.f + std::exp(-gate))) * v;
        if (!std::isfinite(mid[i])) return fail(error, "non-finite intermediate");
    }
    da->from_float(mid.data(), qmid.data(), F);
    for (int i = 0; i < H; ++i) {
        d->vec_dot(F, &staged[i], 0, blob.data() + dn + i * drow, 0, qmid.data(), 0, 1);
        if (!std::isfinite(staged[i])) return fail(error, "non-finite expert output");
    }
    output.swap(staged);
    return true;
}
bool matrix_shape(int type,size_t rows,size_t columns,size_t length) {
    if((type!=8 && type!=12 && type!=14) || rows==0 || rows>10240 || columns==0 || columns>6144 || length>32*1024*1024) return false;
    auto t=static_cast<ggml_type>(type);
    if(columns % size_t(ggml_blck_size(t))!=0) return false;
    const size_t row=ggml_row_size(t,columns);
    return row>0 && rows<=length/row && rows*row==length;
}
bool matrix_forward(int type,size_t rows,size_t columns,const std::vector<uint8_t>& weights,
                    const std::vector<float>& input,std::vector<float>& output,std::string& error) {
    error.clear();
    if(!matrix_shape(type,rows,columns,weights.size()) || input.size()!=columns) return fail(error,"invalid native matrix geometry");
    for(float x:input) if(!std::isfinite(x)) return fail(error,"non-finite matrix input");
    std::call_once(init, [] { ggml_cpu_init(); });
    auto t=static_cast<ggml_type>(type); const auto* traits=ggml_get_type_traits_cpu(t);
    if(!traits || !traits->vec_dot) return fail(error,"missing matrix dot trait");
    const auto* activation=ggml_get_type_traits_cpu(traits->vec_dot_type);
    if(!activation || !activation->from_float || columns%size_t(ggml_blck_size(traits->vec_dot_type))) return fail(error,"missing matrix activation quantizer");
    std::vector<uint8_t> quantized(ggml_row_size(traits->vec_dot_type,columns));
    std::vector<float> staged(rows); activation->from_float(input.data(),quantized.data(),columns);
    const size_t stride=ggml_row_size(t,columns);
    for(size_t i=0;i<rows;++i) {
        traits->vec_dot(int(columns),&staged[i],0,weights.data()+i*stride,0,quantized.data(),0,1);
        if(!std::isfinite(staged[i])) return fail(error,"non-finite matrix output");
    }
    output.swap(staged); return true;
}

}
