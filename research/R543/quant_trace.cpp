// Capture the actual GGML activation quantization for the R537 layer-3 trace.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
static bool read_exact(int fd,void*data,size_t n,off_t off){auto*p=static_cast<unsigned char*>(data);while(n){ssize_t r=pread(fd,p,n,off);if(r<=0)return false;p+=r;n-=size_t(r);off+=r;}return true;}
static bool write_exact(const void*data,size_t n){auto*p=static_cast<const unsigned char*>(data);while(n){ssize_t r=write(STDOUT_FILENO,p,n);if(r<=0)return false;p+=r;n-=size_t(r);}return true;}
static bool same(const struct stat&a,const struct stat&b){return a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_size==b.st_size&&a.st_mtim.tv_sec==b.st_mtim.tv_sec&&a.st_mtim.tv_nsec==b.st_mtim.tv_nsec&&a.st_ctim.tv_sec==b.st_ctim.tv_sec&&a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;}
int main(int argc,char**argv){
 if(argc!=2&&!(argc==3&&std::strcmp(argv[2],"--negative")==0))return 1;
 const bool negative=argc==3; const uint32_t endian=1;if(*reinterpret_cast<const unsigned char*>(&endian)!=1)return 2;
 const int fd=open(argv[1],O_RDONLY|O_NOFOLLOW|O_CLOEXEC);struct stat a{},b{};
 if(fd<0||fstat(fd,&a)||!S_ISREG(a.st_mode)||a.st_size!=92LL*37380*4)return 3;
 ggml_cpu_init();
 const auto *q=ggml_get_type_traits_cpu(GGML_TYPE_IQ4_XS),*k=ggml_get_type_traits_cpu(GGML_TYPE_Q6_K),*o=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
 if(!q||!k||!o||q->vec_dot_type!=GGML_TYPE_Q8_K||k->vec_dot_type!=GGML_TYPE_Q8_K||o->vec_dot_type!=GGML_TYPE_Q8_0)return 4;
 const auto *q8k=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K),*q8=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
 if(!q8k||!q8||!q8k->from_float||!q8->from_float||ggml_row_size(GGML_TYPE_Q8_K,2560)!=2920||ggml_row_size(GGML_TYPE_Q8_0,6144)!=6528)return 5;
 std::vector<float> row(37380);std::vector<uint8_t> a8(2920),b8(6528);
 for(int pos=0;pos<92;++pos){
  if(!read_exact(fd,row.data(),row.size()*4,off_t(pos)*37380*4))return 6;
  for(int j=0;j<2560;++j)if(!std::isfinite(row[j]))return 7;
  for(int j=28676;j<34820;++j)if(!std::isfinite(row[j]))return 7;
  q8k->from_float(row.data(),a8.data(),2560);q8->from_float(row.data()+28676,b8.data(),6144);
  if(negative&&pos==91)a8[4]^=1;
  if(!write_exact(a8.data(),a8.size())||!write_exact(b8.data(),b8.size()))return 8;
 }
 if(fstat(fd,&b)||!same(a,b))return 9;
 close(fd);return 0;
}
