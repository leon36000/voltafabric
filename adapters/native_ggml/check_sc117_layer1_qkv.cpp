// Sample Q5_K QKV rows using a scalar dot and GGML quantization/decoding.
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
 if(argc!=3)return 1;
 std::ifstream shard(argv[1],std::ios::binary|std::ios::ate),result(argv[2],std::ios::binary|std::ios::ate);
 if(!shard||shard.tellg()!=std::streamoff(55063448064LL)||!result||result.tellg()!=std::streamoff(19044*4))return 2;
 std::vector<float> out(19044),ax(2560),row(2560);result.seekg(0);if(!result.read(reinterpret_cast<char*>(out.data()),out.size()*4))return 3;
 for(float f:out){if(!std::isfinite(f))return 4;}ggml_cpu_init();
 const auto*w=ggml_get_type_traits(GGML_TYPE_Q5_K);const auto*a=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);if(!w||!w->to_float||!a||!a->from_float||ggml_row_size(GGML_TYPE_Q5_K,2560)!=1760)return 5;
 std::vector<uint8_t> qx(ggml_row_size(GGML_TYPE_Q8_K,2560)),packed(1760);a->from_float(out.data(),qx.data(),2560);
 for(int i=0;i<10;++i){float scale;std::memcpy(&scale,qx.data()+i*292,4);for(int j=0;j<256;++j){int8_t q;std::memcpy(&q,qx.data()+i*292+4+j,1);ax[i*256+j]=scale*q;}}
 const int ids[]={0,1,127,2047,2048,4095,4096,8191,10239};double maxerr=0;
 for(int i:ids){shard.seekg(2090712960LL+std::streamoff(i)*1760);if(!shard.read(reinterpret_cast<char*>(packed.data()),packed.size()))return 6;w->to_float(packed.data(),row.data(),2560);double expected=0;for(int j=0;j<2560;++j)expected+=double(row[j])*ax[j];double err=std::abs(double(out[2564+i])-expected);maxerr=std::max(maxerr,err);if(!std::isfinite(expected)||err>1e-5+1e-3*std::abs(expected))return 7;}
 std::printf("sampled=9 qkv_values=10240 all_finite=19044 max_abs=%.9g shared_ggml_quantization_and_decoding\n",maxerr);return 0;
}
