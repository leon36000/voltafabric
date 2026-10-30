#include "bridge.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=4) return 1;
    std::ifstream pack(argv[1],std::ios::binary), input(argv[2],std::ios::binary);
    std::vector<float> x(2560), output(2560); std::vector<uint8_t> blob(1971200);
    if(!pack || !input.read(reinterpret_cast<char*>(x.data()),10240)) return 2;
#ifdef VF_PREPARED
    void* prepared=vf_native_prepare(22,x.data(),x.size()); if(!prepared) return 3;
#endif
    std::ofstream out(argv[3],std::ios::binary|std::ios::out|std::ios::trunc); if(!out) return 4;
    const int ids[]={12,64,75,159,181,209,255,257,272,302,480,492};
    for(int e:ids) {
        pack.seekg(1887436800LL+int64_t(e)*1971200);
        if(!pack.read(reinterpret_cast<char*>(blob.data()),blob.size())) return 5;
        void* resident=vf_native_create(22,20,blob.data(),blob.size()); if(!resident) return 6;
#ifdef VF_PREPARED
        int rc=vf_native_forward_prepared(resident,prepared,output.data(),output.size());
#else
        int rc=vf_native_forward(resident,x.data(),x.size(),output.data(),output.size());
#endif
        vf_native_destroy(resident); if(rc) return 7;
        out.write(reinterpret_cast<const char*>(output.data()),output.size()*sizeof(float)); if(!out) return 8;
    }
#ifdef VF_PREPARED
    vf_native_prepared_destroy(prepared);
#endif
    out.close(); return out.fail()?9:0;
}
