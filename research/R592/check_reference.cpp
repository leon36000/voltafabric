#include "scalar_reference.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
static bool read_exact(const char* path, void* dst, size_t size) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()!=std::streampos(size)) return false;
    f.seekg(0);return bool(f.read(static_cast<char*>(dst),size));
}
int main(int argc,char** argv) {
    if(argc!=4) return 1;
    std::vector<uint8_t> blob(1971200),capture(31760);
    std::vector<float> x(2560);std::vector<double> out;std::string error;
    if(!read_exact(argv[1],blob.data(),blob.size())||!read_exact(argv[2],x.data(),x.size()*4)||!read_exact(argv[3],capture.data(),capture.size())) return 2;
    if(!vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,x,out,error)||out.size()!=2560) return 3;
    for(double v:out) if(!std::isfinite(v)) return 4;
    return std::memcmp(out.data(),capture.data()+11280,out.size()*8)==0 ? 0 : 5;
}
