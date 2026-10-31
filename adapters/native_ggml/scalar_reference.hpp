#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vf::native_ggml {
// Opt-in SC117 IQ2_S/IQ4_NL scalar diagnostic; caller authenticates the blob.
// Unweighted f64 outputs; invalid input leaves output unchanged. Not GPU qualified.
bool scalar_reference_iq2s_iq4nl(const std::vector<uint8_t>& blob,
                                  const std::vector<float>& input,
                                  std::vector<double>& output,
                                  std::string& error);
}
