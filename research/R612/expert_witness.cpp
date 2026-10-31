// Local CPU diagnostic: reference uses GGML quantization and dequantization,
// but double scalar dot products are independent of GGML's vector dot kernels.
#include "provider.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <charconv>
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
    if(argc<3 || argc>6) { std::fprintf(stderr,"usage: real_expert_oracle PACK EXPERT_0_TO_511 [REFERENCE_FILE [INPUT_F32 [LAYER]]]\n"); return 1; }
    std::string id=argv[2]; int expert=-1;
    auto parsed=std::from_chars(id.data(),id.data()+id.size(),expert);
    if(parsed.ec!=std::errc{} || parsed.ptr!=id.data()+id.size() || expert<0 || expert>=512) return 2;
    constexpr int H=2560,F=640;
    int layer=0;
    if(argc==6) { std::string s=argv[5]; auto p=std::from_chars(s.data(),s.data()+s.size(),layer); if(p.ec!=std::errc{} || p.ptr!=s.data()+s.size() || (layer!=1 && layer!=2)) return 14; }
    const size_t BLOB=layer==0?2176000:(layer==1?1510400:1971200);
    const int gu=layer==0?18:22, down_type=layer==1?42:20;
    std::ifstream file(argv[1],std::ios::binary); std::vector<uint8_t> blob(BLOB);
    if(!file || !file.read(reinterpret_cast<char*>(blob.data()),BLOB) || file.peek() != std::char_traits<char>::eof()) return 3;
    ggml_cpu_init();
    auto gt=static_cast<ggml_type>(gu), dt=static_cast<ggml_type>(down_type);
    auto g=ggml_get_type_traits_cpu(gt), d=ggml_get_type_traits_cpu(dt);
    auto gc=ggml_get_type_traits(gt), dc=ggml_get_type_traits(dt);
    if(!g || !d || !gc || !dc) return 4;
    auto ga=ggml_get_type_traits_cpu(g->vec_dot_type), da=ggml_get_type_traits_cpu(d->vec_dot_type);
    auto dac=ggml_get_type_traits(d->vec_dot_type);
    if(!gc->to_float || !dc->to_float || !ga->from_float || !da->from_float || !dac->to_float || g->vec_dot_type!=GGML_TYPE_Q8_K) return 4;
    const size_t gr=ggml_row_size(gt,H), dr=ggml_row_size(dt,F), up=gr*F, down=2*up;
    if(down+dr*H!=BLOB) return 5;
    std::vector<float> x(H), output, ax(H), row(H), mid(F), dw(F), am(F);
    std::vector<double> reference(H);
    if(argc>=5) {
        std::ifstream input(argv[4],std::ios::binary);
        if(!input || !input.read(reinterpret_cast<char*>(x.data()),x.size()*sizeof(float))) return 12;
        for(float v:x) if(!std::isfinite(v)) return 13;
    } else {
        for(int i=0;i<H;++i) x[i]=std::sin(float(i)*0.17f)/32.f;
    }
    std::string error;
    if(!vf::native_ggml::forward(gu,down_type,blob,x,output,error)) {std::fprintf(stderr,"provider: %s\n",error.c_str());return 6;}
    std::vector<uint8_t> qx(ggml_row_size(g->vec_dot_type,H)), qmid(ggml_row_size(d->vec_dot_type,F));
    ga->from_float(x.data(),qx.data(),H); decode_q8k(qx,ax);
    for(int i=0;i<F;++i) {
        gc->to_float(blob.data()+size_t(i)*gr,row.data(),H); float gate=float(dot(row,ax));
        gc->to_float(blob.data()+up+size_t(i)*gr,row.data(),H); float value=float(dot(row,ax));
        mid[i]=(gate/(1.f+std::exp(-gate)))*value;
        if(!std::isfinite(mid[i])) return 7;
    }
    std::vector<float> mid_vec(F), am_vec(F);std::vector<uint8_t> qmid_vec(qmid.size());double mid_delta=0;
    for(int i=0;i<F;++i){float g0=0,u0=0;g->vec_dot(H,&g0,0,blob.data()+size_t(i)*gr,0,qx.data(),0,1);g->vec_dot(H,&u0,0,blob.data()+up+size_t(i)*gr,0,qx.data(),0,1);mid_vec[i]=(g0/(1.f+std::exp(-g0)))*u0;mid_delta=std::max(mid_delta,std::abs(double(mid_vec[i])-mid[i]));}
    da->from_float(mid_vec.data(),qmid_vec.data(),F);dac->to_float(qmid_vec.data(),am_vec.data(),F);
    da->from_float(mid.data(),qmid.data(),F); dac->to_float(qmid.data(),am.data(),F);
    int byte_diff=0;double act_delta=0;for(size_t i=0;i<qmid.size();++i)byte_diff+=qmid[i]!=qmid_vec[i];for(int i=0;i<F;++i)act_delta=std::max(act_delta,std::abs(double(am[i])-am_vec[i]));
    std::printf("mid_max_abs=%.9g quant_bytes_differ=%d dequant_max_abs=%.9g\n",mid_delta,byte_diff,act_delta);
    if (d->vec_dot_type != GGML_TYPE_Q8_0 || qmid.size() != 20 * 34) return 15;
    float max_scalar = 0, max_vector = 0;
    for (size_t j=512; j<544; ++j) {
        max_scalar=std::max(max_scalar,std::abs(mid[j]));
        max_vector=std::max(max_vector,std::abs(mid_vec[j]));
    }
    int8_t scalar_code, vector_code;
    std::memcpy(&scalar_code,&qmid[16*34+2+29],1);
    std::memcpy(&vector_code,&qmid_vec[16*34+2+29],1);
    std::printf("witness_index=541 scalar_mid=%.17g vector_mid=%.17g scalar_max=%.17g vector_max=%.17g scalar_scaled=%.17g vector_scaled=%.17g scalar_code=%d vector_code=%d\n",
        double(mid[541]),double(mid_vec[541]),double(max_scalar),double(max_vector),
        double(mid[541]*(127.0f/max_scalar)),double(mid_vec[541]*(127.0f/max_vector)),
        int(scalar_code),int(vector_code));
    for (size_t b = 0; b < qmid.size(); ++b) {
        if (qmid[b] == qmid_vec[b]) continue;
        const size_t block = b / 34, within = b % 34;
        if (within < 2) {
            std::printf("difference_kind=scale block=%zu byte=%zu scalar=%u vector=%u\n", block, within, unsigned(qmid[b]), unsigned(qmid_vec[b]));
            continue;
        }
        const size_t index = block * 32 + within - 2;
        float max_s = 0, max_v = 0;
        for (size_t j = block*32; j < block*32+32; ++j) {
            max_s = std::max(max_s, std::abs(mid[j])); max_v = std::max(max_v, std::abs(mid_vec[j]));
        }
        const float scaled_s = mid[index] * (127.0f / max_s);
        const float scaled_v = mid_vec[index] * (127.0f / max_v);
        int8_t qs, qv; std::memcpy(&qs, &qmid[b], 1); std::memcpy(&qv, &qmid_vec[b], 1);
        std::printf("difference_kind=code block=%zu index=%zu scalar_mid=%.17g vector_mid=%.17g scalar_max=%.17g vector_max=%.17g scalar_scaled=%.17g vector_scaled=%.17g scalar_code=%d vector_code=%d\n", block, index, double(mid[index]), double(mid_vec[index]), double(max_s), double(max_v), double(scaled_s), double(scaled_v), int(qs), int(qv));
    }
    int changed = 0, changed_index = -1, original_violations = 0;
    for (int j=0; j<F; ++j) if (am[j] != am_vec[j]) { ++changed; changed_index = j; }
    double effect_max = 0, linearity_error = 0;
    for (int row_index=0; row_index<H; ++row_index) {
        dc->to_float(blob.data()+down+size_t(row_index)*dr,dw.data(),F);
        const double scalar=dot(dw,am), staged=dot(dw,am_vec);
        const double delta=staged-scalar;
        effect_max=std::max(effect_max,std::abs(delta));
        if (changed == 1) linearity_error=std::max(linearity_error,std::abs(delta-double(dw[changed_index])*(double(am_vec[changed_index])-am[changed_index])));
        original_violations += std::abs(double(output[row_index])-scalar)>1e-7+1e-3*std::abs(scalar);
    }
    std::printf("changed_activations=%d changed_index=%d scalar_vs_staged_max_abs=%.17g single_delta_linearity_max_error=%.17g original_scalar_violations=%d\n", changed, changed_index, effect_max, linearity_error, original_violations);
    qmid=qmid_vec; am=am_vec; std::printf("down_reference_uses_vector_intermediate\n");
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
    if(false) {
        const uint32_t endian=1;
        if(*reinterpret_cast<const uint8_t*>(&endian)!=1) return 10;
        std::ofstream ref(argv[3],std::ios::binary|std::ios::trunc);
        if(!ref.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(float)) ||
           !ref.write(reinterpret_cast<const char*>(reference.data()),reference.size()*sizeof(double))) return 11;
    }
    return 0;
}
