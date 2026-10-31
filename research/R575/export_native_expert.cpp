// Fresh local CPU provider output for one authenticated SC117 expert/input pair.
#include "provider.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
static bool read_exact(const char* path, void* data, size_t size) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if (!in || in.tellg()!=std::streampos(size)) return false;
    in.seekg(0); return bool(in.read(static_cast<char*>(data),size));
}
int main(int argc,char** argv) {
    if (argc!=4) return 1;
    const uint32_t endian=1;
    if (*reinterpret_cast<const uint8_t*>(&endian)!=1) return 2;
    std::vector<uint8_t> blob(1971200); std::vector<float> input(2560),output;
    if (!read_exact(argv[1],blob.data(),blob.size()) || !read_exact(argv[2],input.data(),input.size()*4)) return 3;
    std::string err;
    if (!vf::native_ggml::forward(22,20,blob,input,output,err) || output.size()!=2560) return 4;
    for(float v:output) if(!std::isfinite(v)) return 5;
    int fd=::open(argv[3],O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0) return 6;
    const auto* bytes=reinterpret_cast<const uint8_t*>(output.data()); size_t done=0, size=output.size()*4;
    while(done<size) { const ssize_t n=::write(fd,bytes+done,size-done); if(n<=0){::close(fd);return 7;} done+=size_t(n); }
    if(::fsync(fd)!=0 || ::close(fd)!=0) return 8;
    return 0;
}
