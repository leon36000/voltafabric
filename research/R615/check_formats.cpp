#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
extern "C" int vf_native_scalar_reference(int,int,const uint8_t*,size_t,const float*,size_t,double*,size_t);
static bool read(const char* path,std::vector<uint8_t>& out,size_t limit) {
    std::ifstream f(path,std::ios::binary|std::ios::ate); if(!f) return false;
    auto n=f.tellg();if(n<=0 || n>std::streampos(limit)) return false;
    out.resize(size_t(n));f.seekg(0);return bool(f.read(reinterpret_cast<char*>(out.data()),n));
}
static bool number(const char* s,int& n) {auto end=s+std::strlen(s);auto r=std::from_chars(s,end,n);return r.ec==std::errc{} && r.ptr==end;}
int main(int argc,char** argv) {
    if(argc!=6) return 1;
    int gu=0,down=0;
    if(!number(argv[4],gu)||!number(argv[5],down))return 2;
    std::vector<uint8_t> blob,input,expected;
    if(!read(argv[1],blob,4*1024*1024)||!read(argv[2],input,10240)||input.size()!=10240||!read(argv[3],expected,20480)||expected.size()!=20480) return 3;
    std::vector<float> x(2560);std::memcpy(x.data(),input.data(),input.size());std::vector<double> out(2560,7.0);
    if(vf_native_scalar_reference(gu,down,blob.data(),blob.size(),x.data(),x.size(),out.data(),out.size())!=0) return 4;
    if(std::memcmp(out.data(),expected.data(),expected.size())!=0) return 5;
    const auto previous=out;
    if(vf_native_scalar_reference(gu,down,blob.data(),blob.size()-1,x.data(),x.size(),out.data(),out.size())==0 || out!=previous) return 6;
    return 0;
}
