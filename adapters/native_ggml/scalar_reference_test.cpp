#include "scalar_reference.hpp"
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
int main() {
    constexpr size_t H=2560, BLOB=1971200;
    std::vector<uint8_t> blob(BLOB,0);
    std::vector<float> input(H,0.f);
    std::vector<double> output(3,7.0);
    std::string error;
    if (!vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output.size()!=H) return 1;
    for(double value:output) if(value!=0.0 || !std::isfinite(value)) return 2;
    const auto saved=output;
    input[0]=std::numeric_limits<float>::quiet_NaN();
    if(vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output!=saved) return 3;
    input[0]=0;input.pop_back();
    if(vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output!=saved) return 4;
    input.push_back(0);blob.pop_back();
    if(vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output!=saved) return 5;
    blob.push_back(0);
    input[0]=std::numeric_limits<float>::infinity();
    if(vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output!=saved) return 6;
    input[0]=0;
    if(std::fesetround(FE_DOWNWARD)!=0) return 7;
    bool rejected=!vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error);
    if(std::fesetround(FE_TONEAREST)!=0 || !rejected || output!=saved) return 8;
    blob[0]=0;blob[1]=0x7c; // Non-finite half scale in the first IQ2_S block.
    if(vf::native_ggml::scalar_reference_iq2s_iq4nl(blob,input,output,error) || output!=saved) return 9;
    return 0;
}
