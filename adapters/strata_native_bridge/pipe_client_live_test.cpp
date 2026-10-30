// Test process: stdin/stdout are the same connected AF_UNIX socket after handshake.
#include "pipe_client.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=2) return 1;
    std::ifstream file(argv[1],std::ios::binary); std::vector<float> expected(2560);
    if(!file.read(reinterpret_cast<char*>(expected.data()),expected.size()*sizeof(float))) return 2;
    vf::strata_bridge::Client client(0,1,30000,0);
    std::vector<float> x(2560,0.001f),out(3*2560,9.f); std::vector<int32_t> ids{0,2,1}, kinds{-1,0,-1}; std::string error;
    for(uint64_t batch: {20u,21u}) {
        if(!client.run(batch,1,3,x,ids,kinds,out,error)) return 3;
        for(size_t i=0;i<2560;++i) {
            if(std::memcmp(&out[i],&expected[i],sizeof(float)) || out[2560+i]!=9.f || std::memcmp(&out[5120+i],&expected[i],sizeof(float))) return 4;
        }
    }
    return client.shutdown(error)?0:5;
}
