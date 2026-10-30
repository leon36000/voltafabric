// Sampled CPU reference. Shares GGML quantization/dequantization, not vec-dot.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
int main(int argc,char** argv) {
 if(argc!=4){std::fprintf(stderr,"usage: check SHARD MIXED RESULT\n");return 1;}
 std::ifstream shard(argv[1],std::ios::binary|std::ios::ate),input(argv[2],std::ios::binary|std::ios::ate),result(argv[3],std::ios::binary|std::ios::ate);
 if(!shard||shard.tellg()!=std::streamoff(55063448064LL)||!input||input.tellg()!=10240||!result||result.tellg()!=std::streamoff(14848*4))return 2;
 std::vector<float>x(2560),all(14848);input.seekg(0);result.seekg(0);
 if(!input.read(reinterpret_cast<char*>(x.data()),10240)||!result.read(reinterpret_cast<char*>(all.data()),all.size()*4))return 3;
 for(float v:all)if(!std::isfinite(v))return 4;
 ggml_cpu_init();double maxerr[2]={0,0};
 for(int kind=0;kind<2;++kind){
  const int cols=kind?6144:2560, rows=kind?2560:6144; const auto wt=kind?GGML_TYPE_Q8_0:GGML_TYPE_Q4_K; const auto at=kind?GGML_TYPE_Q8_0:GGML_TYPE_Q8_K;
  const auto*w=ggml_get_type_traits(wt);const auto*a=ggml_get_type_traits_cpu(at);if(!w||!w->to_float||!a||!a->from_float)return 5;
  std::vector<uint8_t> packed(ggml_row_size(wt,cols)),qx(ggml_row_size(at,cols));std::vector<float> row(cols),ax(cols);
  const float* activation=kind?all.data()+6144:x.data();a->from_float(activation,qx.data(),cols);
  if(kind){const auto*d=ggml_get_type_traits(at);if(!d||!d->to_float)return 6;d->to_float(qx.data(),ax.data(),cols);}
  else {for(int i=0;i<cols/256;++i){float scale;std::memcpy(&scale,qx.data()+i*292,4);for(int j=0;j<256;++j){int8_t q;std::memcpy(&q,qx.data()+i*292+4+j,1);ax[i*256+j]=scale*q;}}}
  const int selected[]={0,1,127,1023,2047,3071,4095,5119,6143};
  for(int index:selected){if(index>=rows)continue;shard.seekg(std::streamoff(kind?2061098880LL:883359744LL)+std::streamoff(index)*packed.size());if(!shard.read(reinterpret_cast<char*>(packed.data()),packed.size()))return 7;
   w->to_float(packed.data(),row.data(),cols);double expected=0;for(int j=0;j<cols;++j)expected+=double(row[j])*ax[j];
   double actual=all[(kind?12288:0)+index],err=std::abs(actual-expected);maxerr[kind]=std::max(maxerr[kind],err);if(!std::isfinite(expected)||err>1e-5+1e-3*std::abs(expected))return 8;
  }
 }
 std::printf("all_finite=14848 gate_max_abs=%.9g output_max_abs=%.9g (sampled rows, shared GGML quant/dequant)\n",maxerr[0],maxerr[1]);return 0;
}
