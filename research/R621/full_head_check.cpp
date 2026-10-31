// All Q6_K output rows by scalar f64 dot; shared GGML quantization and decoding.
// Adapted from VoltaFabric R429, GGML llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT).
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
int main(int argc,char**argv){
 if(argc!=3)return 1;std::ifstream s(argv[1],std::ios::binary|std::ios::ate),o(argv[2],std::ios::binary|std::ios::ate);
 if(!s||s.tellg()!=std::streamoff(55063448064LL)||!o||o.tellg()!=std::streamoff(1003524))return 2;
 std::vector<float> x(2560),logits(248320);uint32_t greedy=0;o.seekg(0);
 if(!o.read(reinterpret_cast<char*>(x.data()),10240)||!o.read(reinterpret_cast<char*>(logits.data()),logits.size()*4)||!o.read(reinterpret_cast<char*>(&greedy),4)||greedy>=logits.size())return 3;
 size_t best=0;for(size_t i=0;i<logits.size();++i){if(!std::isfinite(logits[i]))return 4;if(logits[i]>logits[best])best=i;}if(best!=greedy)return 5;
 ggml_cpu_init();auto wt=ggml_get_type_traits(GGML_TYPE_Q6_K);auto at=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);if(!wt||!wt->to_float||!at||!at->from_float)return 6;
 std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_K,2560)),row(ggml_row_size(GGML_TYPE_Q6_K,2560));std::vector<float> ax(2560),w(2560);at->from_float(x.data(),q.data(),2560);
 for(int i=0;i<10;++i){float scale;std::memcpy(&scale,q.data()+i*292,4);for(int j=0;j<256;++j){int8_t z;std::memcpy(&z,q.data()+i*292+4+j,1);ax[i*256+j]=scale*z;}}
 double max_error=0,largest_relative=0,best_ref=-INFINITY;uint32_t ref_id=0,violations=0;
 s.seekg(11024384LL);
 for(uint32_t id=0;id<logits.size();++id){
  if(!s.read(reinterpret_cast<char*>(row.data()),row.size()))return 7;
  wt->to_float(row.data(),w.data(),2560);double sum=0;
  for(int j=0;j<2560;++j)sum+=double(w[j])*ax[j];
  if(!std::isfinite(sum))return 8;
  double e=std::abs(double(logits[id])-sum);max_error=std::max(max_error,e);
  largest_relative=std::max(largest_relative,e/(1e-5+1e-3*std::abs(sum)));
  if(e>1e-5+1e-3*std::abs(sum))++violations;
  if(sum>best_ref){best_ref=sum;ref_id=id;}
 }
 std::printf("{\"rows\":%zu,\"observed_argmax\":%u,\"reference_argmax\":%u,\"violations\":%u,\"max_abs\":%.17g,\"max_tolerance_fraction\":%.17g}\n",logits.size(),greedy,ref_id,violations,max_error,largest_relative);
 return violations||greedy!=ref_id?9:0;
}
