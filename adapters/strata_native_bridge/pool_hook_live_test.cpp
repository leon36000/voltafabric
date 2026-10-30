#include "pool_hook.hpp"
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=2) return 1;
    std::ifstream file(argv[1],std::ios::binary); std::vector<float> expected(2560);
    if(!file.read(reinterpret_cast<char*>(expected.data()),expected.size()*sizeof(float))) return 2;
    vf::strata_bridge::Client client(0,1,30000,0);
    vf::strata_bridge::PoolContext context{client,0,30,false,{}, {}};
    std::vector<float> x(2560,0.001f),out(2*2560,9.f); int32_t ids[2]={0,1};
    for(int i=0;i<2;++i) {
        if(!vf::strata_bridge::pool_cpu_only(&context,x.data(),ids,1,2,out.data(),0)) return 3;
        for(size_t j=0;j<2560;++j) if(std::memcmp(&out[j],&expected[j],4)||std::memcmp(&out[2560+j],&expected[j],4)) return 4;
    }
    std::fill(out.begin(),out.end(),9.f);
    if(vf::strata_bridge::pool_cpu_only(&context,x.data(),ids,1,2,out.data(),1)) return 5;
    if(!context.terminal || vf::strata_bridge::pool_cpu_only(&context,x.data(),ids,1,2,out.data(),0)) return 6;
    for(float v:out) if(v!=9.f) return 7;
    std::string error; return client.shutdown(error)?0:8;
}
