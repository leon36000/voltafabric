// Native SC117 expert CPU boundary, using the pinned ggml CPU type traits.
// Reference: Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf,
// src/kernels/cpu/native_expert.cpp (MIT); llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT).
#include "provider.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <mutex>

namespace vf::native_ggml {
namespace {
constexpr int H = 2560, F = 640;
std::once_flag init;
bool fail(std::string& error, const char* text) { error = text; return false; }
}
bool prepare_input(int gu, const std::vector<float>& input,
                   PreparedInput& prepared, std::string& error) {
    error.clear();
    if (gu != 18 && gu != 21 && gu != 22 && gu != 23) return fail(error, "unsupported native format");
    if (input.size() != H) return fail(error, "invalid activation shape");
    for (float x : input) if (!std::isfinite(x)) return fail(error, "non-finite activation");
    std::call_once(init, [] { ggml_cpu_init(); });
    auto gt=static_cast<ggml_type>(gu);
    const auto* g=ggml_get_type_traits_cpu(gt);
    if (!g || !g->vec_dot) return fail(error, "missing ggml dot trait");
    const auto* ga=ggml_get_type_traits_cpu(g->vec_dot_type);
    if (!ga || !ga->from_float || H % ggml_blck_size(gt) || H % ggml_blck_size(g->vec_dot_type))
        return fail(error, "missing ggml activation quantizer or geometry");
    PreparedInput staged;
    staged.gate_up_type=gu;
    staged.quantized.resize(ggml_row_size(g->vec_dot_type,H));
    ga->from_float(input.data(),staged.quantized.data(),H);
    prepared=std::move(staged);
    return true;
}
bool forward_prepared(int gu, int down, const std::vector<uint8_t>& blob,
                      const PreparedInput& prepared, std::vector<float>& output, std::string& error) {
    error.clear();
    if ((gu != 18 && gu != 21 && gu != 22 && gu != 23) || (down != 20 && down != 42))
        return fail(error, "unsupported native format");
    std::call_once(init, [] { ggml_cpu_init(); });
    auto gt = static_cast<ggml_type>(gu), dt = static_cast<ggml_type>(down);
    const auto* g = ggml_get_type_traits_cpu(gt);
    const auto* d = ggml_get_type_traits_cpu(dt);
    if (!g || !d || !g->vec_dot || !d->vec_dot) return fail(error, "missing ggml dot trait");
    const auto* da = ggml_get_type_traits_cpu(d->vec_dot_type);
    if (!da || !da->from_float) return fail(error, "missing ggml quantizer");
    if (prepared.gate_up_type != gu || prepared.quantized.size() != ggml_row_size(g->vec_dot_type,H) ||
        H % ggml_blck_size(gt) || F % ggml_blck_size(dt) ||
        H % ggml_blck_size(g->vec_dot_type) || F % ggml_blck_size(d->vec_dot_type))
        return fail(error, "invalid prepared input or ggml geometry");
    const size_t grow = ggml_row_size(gt, H), drow = ggml_row_size(dt, F);
    const size_t up = grow * F, dn = 2 * up, size = dn + drow * H;
    if (blob.size() != size) return fail(error, "invalid native blob length");
    std::vector<uint8_t> qmid(ggml_row_size(d->vec_dot_type, F));
    std::vector<float> mid(F), staged(H);
    for (int i = 0; i < F; ++i) {
        float gate = 0.f, v = 0.f;
        g->vec_dot(H, &gate, 0, blob.data() + i * grow, 0, prepared.quantized.data(), 0, 1);
        g->vec_dot(H, &v, 0, blob.data() + up + i * grow, 0, prepared.quantized.data(), 0, 1);
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
bool forward(int gu, int down, const std::vector<uint8_t>& blob,
             const std::vector<float>& input, std::vector<float>& output, std::string& error) {
    PreparedInput prepared;
    return prepare_input(gu,input,prepared,error) && forward_prepared(gu,down,blob,prepared,output,error);
}
bool matrix_shape(int type,size_t rows,size_t columns,size_t length) {
    if((type!=8 && type!=12 && type!=13 && type!=14 && type!=23) || rows==0 || rows>10240 || columns==0 || columns>6144 || length>32*1024*1024) return false;
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

bool routed_matrix_forward(int type, size_t experts, size_t rows, size_t columns,
                           const std::vector<uint8_t>& weights,
                           const std::vector<float>& slot_inputs,
                           const std::vector<int32_t>& expert_ids,
                           std::vector<float>& slot_outputs, std::string& error) {
    error.clear();
    if ((type != 8 && type != 12 && type != 13 && type != 14 && type != 23) ||
        experts == 0 || experts > 512 || rows == 0 || rows > 10240 ||
        columns == 0 || columns > 6144 || expert_ids.empty() || expert_ids.size() > 512 ||
        columns % size_t(ggml_blck_size(static_cast<ggml_type>(type))) != 0)
        return fail(error, "invalid routed expert geometry");

    const auto tensor_type = static_cast<ggml_type>(type);
    const size_t row_bytes = ggml_row_size(tensor_type, columns);
    if (row_bytes == 0 || rows > SIZE_MAX / row_bytes)
        return fail(error, "routed expert size overflow");
    const size_t expert_bytes = rows * row_bytes;
    if (experts > SIZE_MAX / expert_bytes || experts * expert_bytes != weights.size() ||
        weights.size() > 32 * 1024 * 1024)
        return fail(error, "invalid routed expert payload length");
    if (expert_ids.size() > SIZE_MAX / columns ||
        expert_ids.size() * columns != slot_inputs.size())
        return fail(error, "invalid routed activation shape");
    if (expert_ids.size() > SIZE_MAX / rows)
        return fail(error, "routed output size overflow");
    for (int32_t id : expert_ids) {
        if (id < 0 || size_t(id) >= experts)
            return fail(error, "expert id outside resident set");
    }
    for (float x : slot_inputs) {
        if (!std::isfinite(x)) return fail(error, "non-finite routed activation");
    }

    std::call_once(init, [] { ggml_cpu_init(); });
    const auto* traits = ggml_get_type_traits_cpu(tensor_type);
    if (!traits || !traits->vec_dot)
        return fail(error, "missing routed GGML dot trait");
    const auto* activation = ggml_get_type_traits_cpu(traits->vec_dot_type);
    if (!activation || !activation->from_float ||
        columns % size_t(ggml_blck_size(traits->vec_dot_type)) != 0)
        return fail(error, "missing routed GGML activation quantizer");

    const size_t output_count = rows * expert_ids.size();
    const size_t output_bytes = output_count * sizeof(float);
    constexpr size_t context_workspace = 16 * 1024 * 1024;

    struct context_owner {
        ggml_context* value;
        ~context_owner() { if (value) ggml_free(value); }
    };
    ggml_init_params params{};
    params.mem_size = context_workspace;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    context_owner context{ggml_init(params)};
    if (!context.value) return fail(error, "could not allocate routed GGML context");

    ggml_tensor* matrix = ggml_new_tensor_3d(context.value, tensor_type,
                                              int64_t(columns), int64_t(rows), int64_t(experts));
    ggml_tensor* input = ggml_new_tensor_3d(context.value, GGML_TYPE_F32,
                                             int64_t(columns), int64_t(expert_ids.size()), 1);
    ggml_tensor* ids = ggml_new_tensor_2d(context.value, GGML_TYPE_I32,
                                           int64_t(expert_ids.size()), 1);
    if (!matrix || !input || !ids)
        return fail(error, "could not allocate routed GGML tensors");
    // The call is synchronous. Borrow the caller's resident read-only payload and activations.
    matrix->data = const_cast<uint8_t*>(weights.data());
    input->data = const_cast<float*>(slot_inputs.data());
    ids->data = const_cast<int32_t*>(expert_ids.data());

    ggml_tensor* result = ggml_mul_mat_id(context.value, matrix, input, ids);
    ggml_cgraph* graph = ggml_new_graph(context.value);
    if (!result || !graph) return fail(error, "could not construct routed GGML graph");
    std::vector<float> staged(output_count);
    result->data = staged.data();
    ggml_build_forward_expand(graph, result);
    if (ggml_graph_compute_with_ctx(context.value, graph, 1) != GGML_STATUS_SUCCESS ||
        !result->data)
        return fail(error, "routed GGML graph failed");

    if (result->data != staged.data() || output_bytes != staged.size() * sizeof(float))
        return fail(error, "routed GGML output binding changed");
    for (float x : staged) {
        if (!std::isfinite(x)) return fail(error, "non-finite routed expert output");
    }
    slot_outputs.swap(staged);
    return true;
}

}
