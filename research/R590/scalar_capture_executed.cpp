// Diagnostic SC117 scalar reference for one IQ3_S gate/up + IQ4_NL down expert.
// GGML dequantizers and Q8 quantizers: llama.cpp GGML 3cf03257 (MIT).
// Scalar double dot order is explicit; std::exp and float casts remain platform dependent.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <unistd.h>
#include <vector>
static bool read_exact(const char* path, void* dst, size_t size) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != std::streampos(size)) return false;
    f.seekg(0); return bool(f.read(static_cast<char*>(dst), size));
}
static double dot(const std::vector<float>& a, const std::vector<float>& b) {
    double sum=0; for(size_t i=0;i<a.size();++i) sum += double(a[i])*double(b[i]); return sum;
}
static bool write_all(int fd, const void* data, size_t size) {
    const auto* bytes=static_cast<const uint8_t*>(data);
    while(size) { const auto n=::write(fd,bytes,size); if(n<=0) return false; bytes+=n; size-=size_t(n); }
    return true;
}
int main(int argc,char** argv) {
    if(argc!=4) return 1;
    const uint32_t endian=1; if(*reinterpret_cast<const uint8_t*>(&endian)!=1) return 2;
    constexpr int H=2560,F=640;
    std::vector<uint8_t> blob(1971200); std::vector<float> x(H),act(H),row(H),gate(F),up(F),mid(F),down(F),mact(F);
    std::vector<double> out(H);
    if(!read_exact(argv[1],blob.data(),blob.size()) || !read_exact(argv[2],x.data(),x.size()*4)) return 3;
    for(float v:x) if(!std::isfinite(v)) return 4;
    ggml_cpu_init(); auto gt=static_cast<ggml_type>(22),dt=static_cast<ggml_type>(20);
    const auto *g=ggml_get_type_traits_cpu(gt),*d=ggml_get_type_traits_cpu(dt);
    const auto *gc=ggml_get_type_traits(gt),*dc=ggml_get_type_traits(dt);
    if(!g||!d||!gc||!dc||!gc->to_float||!dc->to_float||g->vec_dot_type!=GGML_TYPE_Q8_K||d->vec_dot_type!=GGML_TYPE_Q8_0) return 5;
    const auto *gq=ggml_get_type_traits_cpu(g->vec_dot_type),*dq=ggml_get_type_traits_cpu(d->vec_dot_type);
    const auto *dqc=ggml_get_type_traits(d->vec_dot_type);
    if(!gq||!dq||!dqc||!gq->from_float||!dq->from_float||!dqc->to_float) return 6;
    const size_t gr=ggml_row_size(gt,H),dr=ggml_row_size(dt,F),up_offset=gr*F,down_offset=2*up_offset;
    if(down_offset+dr*H!=blob.size()) return 7;
    std::vector<uint8_t> qx(ggml_row_size(g->vec_dot_type,H)),qm(ggml_row_size(d->vec_dot_type,F));
    gq->from_float(x.data(),qx.data(),H);
    if(qx.size()!=2920 || qm.size()!=680) return 8;
    for(size_t b=0;b<10;++b) { float s; std::memcpy(&s,qx.data()+b*292,4); for(size_t j=0;j<256;++j) {int8_t v;std::memcpy(&v,qx.data()+b*292+4+j,1);act[b*256+j]=s*v;} }
    for(int i=0;i<F;++i) {
        gc->to_float(blob.data()+size_t(i)*gr,row.data(),H); gate[i]=float(dot(row,act));
        gc->to_float(blob.data()+up_offset+size_t(i)*gr,row.data(),H); up[i]=float(dot(row,act));
        mid[i]=(gate[i]/(1.f+std::exp(-gate[i])))*up[i];
        if(!std::isfinite(mid[i])) return 9;
    }
    dq->from_float(mid.data(),qm.data(),F); dqc->to_float(qm.data(),mact.data(),F);
    for(int i=0;i<H;++i) {dc->to_float(blob.data()+down_offset+size_t(i)*dr,down.data(),F);out[i]=dot(down,mact);if(!std::isfinite(out[i])) return 10;}
    int fd=::open(argv[3],O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600); if(fd<0) return 11;
    const bool ok=write_all(fd,qx.data(),qx.size()) && write_all(fd,gate.data(),gate.size()*4) && write_all(fd,up.data(),up.size()*4) && write_all(fd,mid.data(),mid.size()*4) && write_all(fd,qm.data(),qm.size()) && write_all(fd,out.data(),out.size()*8) && ::fsync(fd)==0;
    const int status=::close(fd); return ok && status==0 ? 0 : 12;
}
