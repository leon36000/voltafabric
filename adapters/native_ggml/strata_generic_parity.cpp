// Direct comparison against the pinned Strata generic CPU functions, with synthetic weights.
#include "provider.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
namespace cpu = strata::kernels::cpu;
static bool check(int gu, int down) {
    constexpr int H=2560, F=640, N=2;
    cpu::NativeFmt fmt{}; std::string err;
    if (!cpu::native_fmt(gu,down,H,F,fmt,err)) { std::fprintf(stderr,"native_fmt %s\n",err.c_str()); return false; }
    std::vector<uint8_t> blob(fmt.bytes,0);
    const size_t gs=ggml_type_size((ggml_type)gu), ds=ggml_type_size((ggml_type)down);
    for (size_t at=0;at<fmt.down_off;at+=gs)
        blob[at+1]=uint8_t(0x24 + ((3*(at/fmt.gu_row)+2*(at/fmt.up_off)+(at%fmt.gu_row)/gs)%5));
    for (size_t at=fmt.down_off;at<blob.size();at+=ds) {
        const size_t rel=at-fmt.down_off;
        blob[at+1]=uint8_t(0x24 + ((3*(rel/fmt.d_row)+(rel%fmt.d_row)/ds)%5));
    }
    std::vector<float> x[N], mid[N], donor[N], provider;
    std::vector<uint8_t> a[N], h[N]; const void* ap[N]; const void* hp[N]; float* mp[N]; float* op[N];
    for (int t=0;t<N;++t) {
        x[t].resize(H); mid[t].resize(F); donor[t].resize(H);
        a[t].resize(fmt.act_bytes); h[t].resize(fmt.h_bytes);
        for (int i=0;i<H;++i) x[t][i]=std::sin(float(i)*0.17f + float(t)*0.31f)/32.f;
        cpu::native_quant_act(fmt,x[t].data(),a[t].data()); ap[t]=a[t].data(); mp[t]=mid[t].data();
    }
    cpu::native_gu_rows(fmt,blob.data(),ap,N,mp,0,F);
    for (int t=0;t<N;++t) { cpu::native_quant_h(fmt,mid[t].data(),h[t].data()); hp[t]=h[t].data(); op[t]=donor[t].data(); }
    cpu::native_down_rows(fmt,blob.data(),hp,N,op,0,H);
    bool nonzero=false, varied=false;
    for (int t=0;t<N;++t) {
        if (!vf::native_ggml::forward(gu,down,blob,x[t],provider,err)) { std::fprintf(stderr,"provider %s\n",err.c_str()); return false; }
        for (int i=0;i<H;++i) {
            if (!std::isfinite(donor[t][i]) || std::memcmp(&donor[t][i],&provider[i],sizeof(float))!=0) {
                std::fprintf(stderr,"mismatch %d/%d token %d row %d donor %.9g provider %.9g\n",gu,down,t,i,donor[t][i],provider[i]); return false;
            }
            nonzero |= donor[t][i] != 0.f;
            varied |= i > 0 && donor[t][i] != donor[t][0];
        }
    }
    return nonzero && varied;
}
int main() {
    const int pairs[][2]={{18,20},{18,42},{21,20},{21,42},{22,20},{22,42},{23,20}};
    for (const auto& pair:pairs) if (!check(pair[0],pair[1])) return 1;
    std::puts("Strata pinned generic CPU: all seven pairs, two tokens, bit-identical"); return 0;
}
