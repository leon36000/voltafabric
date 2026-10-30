// Sample real Q/K/V/output projection rows using scalar dots and GGML decoders.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <vector>
int main(int argc,char**argv){
 if(argc!=11)return 1;std::ifstream f(argv[1],std::ios::binary|std::ios::ate),i(argv[2],std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(55063448064LL)||!i||i.tellg()!=std::streamoff(37380*4))return 2;
 std::vector<float> o(37380);i.seekg(0);if(!i.read(reinterpret_cast<char*>(o.data()),o.size()*4))return 3;ggml_cpu_init();
 for(int p=0;p<4;++p){int kind=std::atoi(argv[3+2*p]);if((p==3 && kind!=8)||(p!=3 && kind!=12 && kind!=13 && kind!=14 && kind!=23))return 8;ggml_type type=static_cast<ggml_type>(kind);int cols=p==3?6144:2560;int rows=p==0?12288:(p==3?2560:512);int base=p==0?2564:(p==1?27140:(p==2?28164:34820));char* end=nullptr;int64_t offset=std::strtoll(argv[4+2*p],&end,10);if(!end || *end || offset<0)return 9;
  auto w=ggml_get_type_traits(type);auto c=ggml_get_type_traits_cpu(type);auto aq=ggml_get_type_traits_cpu(c->vec_dot_type);auto ad=ggml_get_type_traits(c->vec_dot_type);if(!w||!w->to_float||!c||!aq||!aq->from_float)return 4;
  std::vector<uint8_t> x(ggml_row_size(c->vec_dot_type,cols)),b(ggml_row_size(type,cols));std::vector<float> xd(cols),wd(cols);aq->from_float(o.data()+(p==3?28676:0),x.data(),cols);
  if(c->vec_dot_type==GGML_TYPE_Q8_K){for(int n=0;n<cols/256;++n){float scale;std::memcpy(&scale,x.data()+n*292,4);for(int j=0;j<256;++j){int8_t v;std::memcpy(&v,x.data()+n*292+4+j,1);xd[n*256+j]=scale*v;}}}else{if(!ad||!ad->to_float)return 5;ad->to_float(x.data(),xd.data(),cols);}
  int ids[]={0,1,127,255,511,1023,2047,4095,6143,8191,12287};int count=0;double maxerr=0;for(int row:ids){if(row>=rows)continue;++count;f.seekg(offset+std::streamoff(row)*b.size());if(!f.read(reinterpret_cast<char*>(b.data()),b.size()))return 6;w->to_float(b.data(),wd.data(),cols);double sum=0;for(int j=0;j<cols;++j)sum+=double(wd[j])*xd[j];double e=std::abs(double(o[base+row])-sum);maxerr=std::max(maxerr,e);if(!std::isfinite(sum)||e>1e-5+1e-3*std::abs(sum))return 7;}
  std::printf("projection=%d type=%d sampled=%d max_abs=%.9g\n",p,kind,count,maxerr);
 }
 return 0;
}
