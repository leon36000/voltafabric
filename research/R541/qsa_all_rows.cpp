// Full-row scalar projection check for the bounded R537 live layer-3 trace.
// Independent sums, with the same GGML activation quantizers and decoders.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static bool same(const struct stat &a, const struct stat &b) {
    return a.st_dev==b.st_dev && a.st_ino==b.st_ino && a.st_size==b.st_size &&
        a.st_mtim.tv_sec==b.st_mtim.tv_sec && a.st_mtim.tv_nsec==b.st_mtim.tv_nsec &&
        a.st_ctim.tv_sec==b.st_ctim.tv_sec && a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;
}
static bool read_exact(int fd, void *data, size_t n, off_t off) {
    auto *p=static_cast<unsigned char*>(data);
    while(n) { const ssize_t r=pread(fd,p,n,off); if(r<=0)return false; p+=r;n-=size_t(r);off+=r; }
    return true;
}
int main(int argc,char **argv) {
    if(argc!=3 && !(argc==4 && std::strcmp(argv[3],"--negative")==0))return 1;
    const bool negative=argc==4;
    const int shard=open(argv[1],O_RDONLY|O_NOFOLLOW|O_CLOEXEC), trace=open(argv[2],O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
    struct stat s0{},t0{},s1{},t1{};
    if(shard<0||trace<0||fstat(shard,&s0)||fstat(trace,&t0)||!S_ISREG(s0.st_mode)||!S_ISREG(t0.st_mode)||
       s0.st_size!=55063448064LL||t0.st_size!=92LL*37380*4)return 2;
    std::vector<float> observed(92*37380);
    if(!read_exact(trace,observed.data(),observed.size()*sizeof(float),0))return 3;
    if(negative)observed[91*37380+2564]+=0.5f;
    ggml_cpu_init();
    for(int p=0;p<4;++p) {
        const auto type=p==0?GGML_TYPE_IQ4_XS:(p==3?GGML_TYPE_Q8_0:GGML_TYPE_Q6_K);
        const int cols=p==3?6144:2560, rows=p==0?12288:(p==3?2560:512);
        const int base=p==0?2564:(p==1?27140:(p==2?28164:34820));
        const int64_t offset=p==0?4105151616LL:(p==1?4087363712LL:(p==2?4121864320LL:4088439936LL));
        const auto *w=ggml_get_type_traits(type); const auto *c=ggml_get_type_traits_cpu(type);
        if(!w||!w->to_float||!c)return 4;
        const auto *aq=ggml_get_type_traits_cpu(c->vec_dot_type);const auto *ad=ggml_get_type_traits(c->vec_dot_type);
        if(!aq||!aq->from_float)return 4;
        std::vector<float> inputs(size_t(92)*cols),wd(cols);
        std::vector<uint8_t> quant(ggml_row_size(c->vec_dot_type,cols)),weights(ggml_row_size(type,cols));
        for(int pos=0;pos<92;++pos) {
            const float *in=observed.data()+size_t(pos)*37380+(p==3?28676:0);
            for(int j=0;j<cols;++j)if(!std::isfinite(in[j]))return 5;
            aq->from_float(in,quant.data(),cols);
            float *out=inputs.data()+size_t(pos)*cols;
            if(c->vec_dot_type==GGML_TYPE_Q8_K) {
                for(int n=0;n<cols/256;++n) { float scale;std::memcpy(&scale,quant.data()+n*292,4);
                    for(int j=0;j<256;++j){int8_t v;std::memcpy(&v,quant.data()+n*292+4+j,1);out[n*256+j]=scale*v;}}
            } else {if(!ad||!ad->to_float)return 5;ad->to_float(quant.data(),out,cols);}
        }
        double maximum=0,maxratio=0;uint64_t count=0;
        for(int row=0;row<rows;++row) {
            if(!read_exact(shard,weights.data(),weights.size(),offset+off_t(row)*weights.size()))return 6;
            w->to_float(weights.data(),wd.data(),cols);
            for(int pos=0;pos<92;++pos) {
                const float *x=inputs.data()+size_t(pos)*cols;
                double sum=0;for(int j=0;j<cols;++j)sum+=double(wd[j])*x[j];
                const double actual=observed[size_t(pos)*37380+base+row];
                const double error=std::abs(actual-sum),tol=1e-5+1e-3*std::abs(sum);
                if(!std::isfinite(sum)||!std::isfinite(actual)||!std::isfinite(error)||error>tol) {
                    std::fprintf(stderr,"mismatch position=%d projection=%d row=%d error=%.9g tolerance=%.9g\n",pos,p,row,error,tol);return 7;
                }
                maximum=std::max(maximum,error);maxratio=std::max(maxratio,error/tol);++count;
            }
        }
        std::printf("{\"projection\":%d,\"rows\":%d,\"positions\":92,\"comparisons\":%llu,\"max_abs\":%.17g,\"max_tolerance_fraction\":%.17g}\n",p,rows,(unsigned long long)count,maximum,maxratio);
    }
    if(fstat(shard,&s1)||fstat(trace,&t1)||!same(s0,s1)||!same(t0,t1))return 8;
    close(shard);close(trace);return 0;
}
