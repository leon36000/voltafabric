// Sampled SC117 shared projections: scalar dot, shared GGML decoding/quantization.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
int main(int argc,char** argv){
 if(argc!=4)return 1;
 std::ifstream shard(argv[1],std::ios::binary|std::ios::ate),input(argv[2],std::ios::binary|std::ios::ate),result(argv[3],std::ios::binary|std::ios::ate);
 if(!shard||shard.tellg()!=std::streamoff(55063448064LL)||!input||input.tellg()!=std::streamoff(110728)||!result||result.tellg()!=std::streamoff(19841*4))return 2;
 std::vector<float>x(27682),out(19841);input.seekg(0);result.seekg(0);if(!input.read(reinterpret_cast<char*>(x.data()),x.size()*4)||!result.read(reinterpret_cast<char*>(out.data()),out.size()*4))return 3;
 for(float f:out){if(!std::isfinite(f))return 4;}
 ggml_cpu_init();double maxerr[3]={0,0,0};
 for(int part=0;part<3;++part){
  const auto wt=part==0?GGML_TYPE_IQ4_XS:(part==1?GGML_TYPE_Q4_K:GGML_TYPE_Q8_0);
  const int cols=part==2?640:2560,rows=part==2?2560:640;
  const int64_t offset=part==0?2617729920LL:(part==1?2887297920LL:2344664960LL);
  const int outbase=part==0?0:(part==1?640:1920);const float* in=part==2?out.data()+1280:x.data()+25088;
  const auto*w=ggml_get_type_traits(wt);const auto*t=ggml_get_type_traits_cpu(wt);if(!w||!w->to_float||!t||!t->vec_dot)return 5;
  const auto*a=ggml_get_type_traits_cpu(t->vec_dot_type);const auto*d=ggml_get_type_traits(t->vec_dot_type);if(!a||!a->from_float)return 6;
  std::vector<uint8_t> quant(ggml_row_size(t->vec_dot_type,cols)),packed(ggml_row_size(wt,cols));std::vector<float> aq(cols),row(cols);a->from_float(in,quant.data(),cols);
  if(t->vec_dot_type==GGML_TYPE_Q8_K){for(int i=0;i<cols/256;++i){float scale;std::memcpy(&scale,quant.data()+i*292,4);for(int j=0;j<256;++j){int8_t q;std::memcpy(&q,quant.data()+i*292+4+j,1);aq[i*256+j]=scale*q;}}}
  else {if(!d||!d->to_float)return 7;d->to_float(quant.data(),aq.data(),cols);}
  int selected[]={0,1,127,319,639,1023,2047,2559};int count=0;
  for(int index:selected){if(index>=rows)continue;++count;shard.seekg(offset+std::streamoff(index)*packed.size());if(!shard.read(reinterpret_cast<char*>(packed.data()),packed.size()))return 8;w->to_float(packed.data(),row.data(),cols);double expected=0;for(int j=0;j<cols;++j)expected+=double(row[j])*aq[j];double err=std::abs(double(out[outbase+index])-expected);maxerr[part]=std::max(maxerr[part],err);if(!std::isfinite(expected)||err>1e-5+1e-3*std::abs(expected))return 9;}
  std::printf("part=%d sampled=%d max_abs=%.9g\n",part,count,maxerr[part]);
 }
 return 0;
}
