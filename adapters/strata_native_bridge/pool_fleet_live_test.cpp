#include "pool_hook.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=2) return 1;
    std::ifstream file(argv[1],std::ios::binary); std::vector<float> expected(2560);
    if(!file.read(reinterpret_cast<char*>(expected.data()),expected.size()*sizeof(float))) return 2;
    if(std::all_of(expected.begin(),expected.end(),[](float x){return x==0.f;})) return 3;
    vf::strata_bridge::Client c0(0,1,30000,0), c7(2,2,30000,7);
    vf::strata_bridge::PoolContext a{c0,0,60,false,{}, {}}, b{c7,7,60,false,{}, {}};
    std::array<vf::strata_bridge::PoolContext*,48> contexts{}; contexts[0]=&a; contexts[7]=&b;
    vf::strata_bridge::PoolFleet fleet(contexts);
    std::vector<float> x(2560,0.001f), out(2*2560,9.f); int32_t ids[2]={0,1};
    auto wrong_contexts=contexts; wrong_contexts[0]=&b;
    vf::strata_bridge::PoolFleet miswired(wrong_contexts);
    if(vf::strata_bridge::pool_cpu_by_layer(&miswired,x.data(),ids,1,2,out.data(),0)) return 11;
    if(!std::all_of(out.begin(),out.end(),[](float v){return v==9.f;})) return 12;
    if(a.next_batch!=60 || b.next_batch!=60) return 13;
    for(int layer : {0,7,0,7}) {
        std::fill(out.begin(),out.end(),9.f);
        if(!vf::strata_bridge::pool_cpu_by_layer(&fleet,x.data(),ids,1,2,out.data(),layer)) return 4;
        for(size_t i=0;i<out.size();++i) {
            float target=layer==0?expected[i%2560]:0.f;
            if(std::memcmp(&out[i],&target,4)) return 5;
        }
    }
    if(a.next_batch!=62 || b.next_batch!=62) return 6;
    std::fill(out.begin(),out.end(),9.f);
    if(vf::strata_bridge::pool_cpu_by_layer(&fleet,x.data(),ids,1,2,out.data(),1)) return 7;
    if(vf::strata_bridge::pool_cpu_by_layer(&fleet,x.data(),ids,1,2,out.data(),0)) return 8;
    if(!std::all_of(out.begin(),out.end(),[](float v){return v==9.f;})) return 9;
    std::string e0,e7; return c0.shutdown(e0)&&c7.shutdown(e7)?0:10;
}
