// Independent scalar dot products with shared GGML decoders/activation quantizers.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
static bool mv(std::ifstream &f, int64_t offset, ggml_type type, const std::vector<float>& x, int rows, std::vector<float>& y){
 const auto *w=ggml_get_type_traits(type);const auto *cpu=ggml_get_type_traits_cpu(type);if(!w||!w->to_float||!cpu)return false;
 const auto *act=ggml_get_type_traits_cpu(cpu->vec_dot_type);const auto *dec=ggml_get_type_traits(cpu->vec_dot_type);if(!act||!act->from_float)return false;
 std::vector<uint8_t> q(ggml_row_size(cpu->vec_dot_type,x.size())), row(ggml_row_size(type,x.size()));std::vector<float> a(x.size()), d(x.size());
 act->from_float(x.data(),q.data(),x.size());
 if(cpu->vec_dot_type==GGML_TYPE_Q8_K){for(size_t i=0;i<x.size()/256;++i){float scale;std::memcpy(&scale,q.data()+i*292,4);for(int j=0;j<256;++j){int8_t v;std::memcpy(&v,q.data()+i*292+4+j,1);a[i*256+j]=scale*v;}}}
 else {if(!dec||!dec->to_float)return false;dec->to_float(q.data(),a.data(),x.size());}
 y.resize(rows);f.seekg(offset);for(int i=0;i<rows;++i){if(!f.read(reinterpret_cast<char*>(row.data()),row.size()))return false;w->to_float(row.data(),d.data(),x.size());double sum=0;for(size_t j=0;j<x.size();++j)sum+=double(a[j])*d[j];y[i]=float(sum);}return true;
}
int main(int argc,char**argv){
 if(argc!=4)return 1;std::ifstream f(argv[1],std::ios::binary|std::ios::ate), in(argv[2],std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(55063448064LL)||!in||in.tellg()!=10240)return 2;
 std::vector<float>x(2560),gate,up,down;in.seekg(0);if(!in.read(reinterpret_cast<char*>(x.data()),10240))return 3;ggml_cpu_init();
 if(!mv(f,5947421824LL,GGML_TYPE_Q6_K,x,640,gate)||!mv(f,6217463424LL,GGML_TYPE_Q5_K,x,640,up))return 4;
 std::vector<float> mid(640);for(int i=0;i<640;++i)mid[i]=gate[i]/(1.f+std::exp(-gate[i]))*up[i];
 if(!mv(f,5674356864LL,GGML_TYPE_Q8_0,mid,2560,down))return 5;
 f.seekg(5947416704LL);double dot=0;for(float v:x){uint16_t w; if(!f.read(reinterpret_cast<char*>(&w),2))return 6;uint32_t xb;std::memcpy(&xb,&v,4);xb=(xb+0x7fffU+((xb>>16)&1U))&0xffff0000U;float xx,ww;std::memcpy(&xx,&xb,4);uint32_t wb=uint32_t(w)<<16;std::memcpy(&ww,&wb,4);dot+=double(xx)*ww;}
 float scalar=float(1.0/(1.0+std::exp(-dot)));std::ofstream out(argv[3],std::ios::binary|std::ios::out);for(float v:down){float y=v*scalar;if(!std::isfinite(y)||!out.write(reinterpret_cast<char*>(&y),4))return 7;}return 0;
}
