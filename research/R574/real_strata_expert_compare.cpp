// Pinned Strata generic CPU arithmetic versus VoltaFabric on the R428 witness.
// The input reference stores 2560 f32 input values followed by 2560 f64 staged outputs.
#include "provider.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
namespace cpu = strata::kernels::cpu;
static bool read_exact(const char* path, void* data, size_t size) {
    std::ifstream in(path,std::ios::binary | std::ios::ate);
    if (!in || in.tellg()!=std::streampos(size)) return false;
    in.seekg(0); return bool(in.read(static_cast<char*>(data),size));
}
int main(int argc,char** argv) {
    if (argc != 4) return 1;
    constexpr int H=2560,F=640;
    cpu::NativeFmt fmt{}; std::string err;
    if (!cpu::native_fmt(22,20,H,F,fmt,err) || fmt.bytes!=1971200) return 2;
    std::vector<uint8_t> blob(fmt.bytes), aq(fmt.act_bytes), hq(fmt.h_bytes);
    std::vector<float> x(H), mid(F), donor(H), provider, ref_input(H);
    std::vector<double> ref(H);
    if (!read_exact(argv[1],blob.data(),blob.size()) || !read_exact(argv[2],x.data(),x.size()*4)) return 3;
    std::ifstream in(argv[3],std::ios::binary | std::ios::ate);
    if (!in || in.tellg()!=std::streampos(H*4+H*8)) return 4;
    in.seekg(0); if (!in.read(reinterpret_cast<char*>(ref_input.data()),H*4) || !in.read(reinterpret_cast<char*>(ref.data()),H*8)) return 4;
    if (std::memcmp(x.data(),ref_input.data(),H*4)!=0) return 5;
    for (float v:x) if (!std::isfinite(v)) return 6;
    cpu::native_quant_act(fmt,x.data(),aq.data());
    const void* acts[]={aq.data()}; float* mids[]={mid.data()};
    cpu::native_gu_rows(fmt,blob.data(),acts,1,mids,0,F);
    for (float v:mid) if (!std::isfinite(v)) return 7;
    cpu::native_quant_h(fmt,mid.data(),hq.data());
    float block_max=0;
    for (int i=512;i<544;++i) block_max=std::max(block_max,std::abs(mid[i]));
    int8_t code; std::memcpy(&code,hq.data()+16*34+2+(541-512),1);
    std::printf("mid541=%.17g block_max=%.17g normalized=%.17g q8_code=%d\n",double(mid[541]),double(block_max),double(mid[541]*(127.0f/block_max)),int(code));
    const void* hs[]={hq.data()}; float* outs[]={donor.data()};
    cpu::native_down_rows(fmt,blob.data(),hs,1,outs,0,H);
    if (!vf::native_ggml::forward(22,20,blob,x,provider,err) || provider.size()!=H) return 8;
    int donor_diff=0, ref_fail=0; double max_donor=0,max_ref=0;
    for(int i=0;i<H;++i) {
        if (!std::isfinite(donor[i]) || !std::isfinite(provider[i]) || !std::isfinite(ref[i])) return 9;
        donor_diff += std::memcmp(&donor[i],&provider[i],sizeof(float))!=0;
        max_donor=std::max(max_donor,std::abs(double(donor[i])-provider[i]));
        const double delta=std::abs(double(provider[i])-ref[i]); max_ref=std::max(max_ref,delta);
        ref_fail += delta > 1e-7+1e-3*std::abs(ref[i]);
    }
    std::printf("donor_bit_differences=%d donor_max_abs=%.17g x86_staged_reference_violations=%d x86_staged_reference_max_abs=%.17g\n",donor_diff,max_donor,ref_fail,max_ref);
    return donor_diff || ref_fail ? 10 : 0;
}
