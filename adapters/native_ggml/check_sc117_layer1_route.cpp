// Sample layer-1 gate and output projection rows with shared GGML quantization/decoding.
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
 std::ifstream shard(argv[1],std::ios::binary|std::ios::ate),gdn(argv[2],std::ios::binary|std::ios::ate),result(argv[3],std::ios::binary|std::ios::ate);
 if(!shard||shard.tellg()!=std::streamoff(55063448064LL)||!gdn||gdn.tellg()!=std::streamoff(76176)||!result||result.tellg()!=std::streamoff(110728))return 2;
 std::vector<float> input(19044),out(27652);gdn.seekg(0);result.seekg(0);
 if(!gdn.read(reinterpret_cast<char*>(input.data()),input.size()*4)||!result.read(reinterpret_cast<char*>(out.data()),out.size()*4))return 3;
 for(float f:out){if(!std::isfinite(f))return 4;}ggml_cpu_init();
 for(int kind=0;kind<2;++kind){
  const auto wt=kind?GGML_TYPE_Q8_0:GGML_TYPE_Q6_K;const auto at=kind?GGML_TYPE_Q8_0:GGML_TYPE_Q8_K;const int cols=kind?6144:2560,rows=kind?2560:6144;
  const auto*w=ggml_get_type_traits(wt);const auto*a=ggml_get_type_traits_cpu(at);const auto*dec=ggml_get_type_traits(at);if(!w||!w->to_float||!a||!a->from_float)return 5;
  std::vector<uint8_t> qx(ggml_row_size(at,cols)),packed(ggml_row_size(wt,cols));std::vector<float> ax(cols),row(cols);
  a->from_float(kind?out.data()+6144:input.data(),qx.data(),cols);
  if(kind){if(!dec||!dec->to_float)return 6;dec->to_float(qx.data(),ax.data(),cols);}
  else {for(int i=0;i<cols/256;++i){float scale;std::memcpy(&scale,qx.data()+i*292,4);for(int j=0;j<256;++j){int8_t q;std::memcpy(&q,qx.data()+i*292+4+j,1);ax[i*256+j]=scale*q;}}}
  const int ids[]={0,1,127,1023,2047,3071,4095,5119,6143};double maxerr=0;int count=0;
  for(int i:ids){if(i>=rows)continue;++count;shard.seekg((kind?2981076736LL:2077810560LL)+std::streamoff(i)*packed.size());if(!shard.read(reinterpret_cast<char*>(packed.data()),packed.size()))return 7;w->to_float(packed.data(),row.data(),cols);double expected=0;for(int j=0;j<cols;++j)expected+=double(row[j])*ax[j];double err=std::abs(double(out[(kind?12288:0)+i])-expected);maxerr=std::max(maxerr,err);if(!std::isfinite(expected)||err>1e-5+1e-3*std::abs(expected))return 8;}
  std::printf("part=%d sampled=%d max_abs=%.9g\n",kind,count,maxerr);
 }
 return 0;
}
