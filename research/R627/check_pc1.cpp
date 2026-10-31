#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
extern "C" int vf_native_scalar_prepare(const float*,size_t,float*,size_t);
extern "C" int vf_native_scalar_reference_prepared(int,int,const uint8_t*,size_t,const float*,size_t,double*,size_t);
static bool read_exact(const char* path,void* out,size_t n) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=std::streampos(n)) return false;
    file.seekg(0);return bool(file.read(static_cast<char*>(out),n));
}
int main(int argc,char** argv) {
    if(argc!=4) return 1;
    std::vector<uint8_t> blob(2176000),capture(30720);
    std::vector<float> input(2560);std::vector<double> output(2560,7.0);
    if(!read_exact(argv[1],blob.data(),blob.size()) || !read_exact(argv[2],input.data(),input.size()*sizeof(float)) ||
       !read_exact(argv[3],capture.data(),capture.size())) return 2;
    std::vector<float> prepared(2560);
    if(vf_native_scalar_prepare(input.data(),input.size(),prepared.data(),prepared.size())!=0) return 3;
    if(vf_native_scalar_reference_prepared(18,20,blob.data(),blob.size(),prepared.data(),prepared.size(),output.data(),output.size())!=0) return 3;
    if(std::memcmp(output.data(),capture.data()+10240,output.size()*sizeof(double))!=0) return 4;
    const auto saved=output;
    if(vf_native_scalar_reference_prepared(18,20,blob.data(),blob.size()-1,prepared.data(),prepared.size(),output.data(),output.size())==0 || output!=saved) return 5;
    return 0;
}
