#include "scalar_reference.hpp"
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>
static bool read_bounded(const char* path,std::vector<uint8_t>& dst,size_t limit) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()<0 || uint64_t(f.tellg())>limit) return false;
    dst.resize(size_t(f.tellg()));f.seekg(0);return bool(f.read(reinterpret_cast<char*>(dst.data()),std::streamsize(dst.size())));
}
int main(int argc,char** argv) {
    if(argc!=6) return 1;
    int gu=-1,down=-1;const std::string a=argv[1],b=argv[2];
    auto pa=std::from_chars(a.data(),a.data()+a.size(),gu),pb=std::from_chars(b.data(),b.data()+b.size(),down);
    if(pa.ec!=std::errc{}||pa.ptr!=a.data()+a.size()||pb.ec!=std::errc{}||pb.ptr!=b.data()+b.size()) return 2;
    std::vector<uint8_t> blob,raw;std::vector<double> out;std::string error;
    if(!read_bounded(argv[3],blob,4*1024*1024)||!read_bounded(argv[4],raw,10240)||raw.size()!=10240) return 3;
    std::vector<float> input(2560);std::memcpy(input.data(),raw.data(),raw.size());
    if(!vf::native_ggml::scalar_reference(gu,down,blob,input,out,error)||out.size()!=2560) return 4;
    int fd=::open(argv[5],O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW,0600);if(fd<0) return 5;
    const auto* data=reinterpret_cast<const uint8_t*>(out.data());size_t left=out.size()*8;
    while(left){ssize_t n=::write(fd,data,left);if(n<=0){::close(fd);return 6;}data+=n;left-=size_t(n);}
    const bool ok=::fsync(fd)==0;return (::close(fd)==0 && ok)?0:7;
}
