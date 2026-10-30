// Stream layer-3 GGML-decoded rows for an external decoder comparison.
#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
static bool same(const struct stat&a,const struct stat&b){return a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_size==b.st_size&&a.st_mtim.tv_sec==b.st_mtim.tv_sec&&a.st_mtim.tv_nsec==b.st_mtim.tv_nsec&&a.st_ctim.tv_sec==b.st_ctim.tv_sec&&a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;}
static bool read_exact(int fd,void*data,size_t n,off_t off){auto*p=static_cast<unsigned char*>(data);while(n){ssize_t r=pread(fd,p,n,off);if(r<=0)return false;p+=r;n-=size_t(r);off+=r;}return true;}
static bool write_exact(const void*data,size_t n){auto*p=static_cast<const unsigned char*>(data);while(n){ssize_t r=write(STDOUT_FILENO,p,n);if(r<=0)return false;p+=r;n-=size_t(r);}return true;}
int main(int argc,char**argv){
 if(argc!=2&&!(argc==3&&std::strcmp(argv[2],"--negative")==0))return 1;
 const bool negative=argc==3;const uint32_t endian=1;if(*reinterpret_cast<const unsigned char*>(&endian)!=1)return 2;
 const int fd=open(argv[1],O_RDONLY|O_NOFOLLOW|O_CLOEXEC);struct stat a{},b{};
 if(fd<0||fstat(fd,&a)||!S_ISREG(a.st_mode)||a.st_size!=55063448064LL)return 3;
 for(int p=0;p<4;++p){
  const auto type=p==0?GGML_TYPE_IQ4_XS:(p==3?GGML_TYPE_Q8_0:GGML_TYPE_Q6_K);
  const int cols=p==3?6144:2560,rows=p==0?12288:(p==3?2560:512);
  const int64_t offset=p==0?4105151616LL:(p==1?4087363712LL:(p==2?4121864320LL:4088439936LL));
  const auto *t=ggml_get_type_traits(type);if(!t||!t->to_float)return 4;
  std::vector<uint8_t> packed(ggml_row_size(type,cols));std::vector<float> out(cols);
  for(int row=0;row<rows;++row){
   if(!read_exact(fd,packed.data(),packed.size(),offset+off_t(row)*packed.size()))return 5;
   t->to_float(packed.data(),out.data(),cols);
   if(negative&&p==0&&row==0)out[0]+=0.5f;
   for(float v:out)if(!std::isfinite(v))return 6;
   if(!write_exact(out.data(),out.size()*sizeof(float)))return 7;
  }
 }
 if(fstat(fd,&b)||!same(a,b))return 8;
 close(fd);
 return 0;
}
