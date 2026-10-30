// Diagnostic only: bounded Q8-independent IQ4_XS embedding row from observed SC117 shard.
#include "ggml.h"
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=3) { std::fprintf(stderr,"usage: sc117_embedding SHARD TOKEN_ID\n"); return 1; }
    std::string token=argv[2]; int id=-1; auto p=std::from_chars(token.data(),token.data()+token.size(),id);
    if(p.ec!=std::errc{} || p.ptr!=token.data()+token.size() || id<0 || id>=248320) return 2;
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=std::streamoff(55063448064LL)) return 3;
    constexpr size_t row_bytes=1360;
    if(ggml_row_size(GGML_TYPE_IQ4_XS,2560)!=row_bytes) return 4;
    const auto* traits=ggml_get_type_traits(GGML_TYPE_IQ4_XS); if(!traits || !traits->to_float) return 5;
    std::vector<uint8_t> row(row_bytes); std::vector<float> out(2560);
    if(!file.seekg(std::streamoff(545644544LL)+std::streamoff(id)*row_bytes) ||
       !file.read(reinterpret_cast<char*>(row.data()),row.size())) return 6;
    traits->to_float(row.data(),out.data(),2560);
    for(float v:out) if(!std::isfinite(v)) return 7;
    const uint32_t little=1; if(*reinterpret_cast<const uint8_t*>(&little)!=1) return 8;
    return std::fwrite(out.data(),sizeof(float),out.size(),stdout)==out.size()?0:9;
}
