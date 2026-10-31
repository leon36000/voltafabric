#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vf::native_ggml {
// Only the seven explicit SC117 gate/up and down format pairs are admitted.
// This reference is unweighted; it is not a production worker or GPU kernel.
bool scalar_reference(int gate_up_type, int down_type,
                      const std::vector<uint8_t>& blob, const std::vector<float>& input,
                      std::vector<double>& output, std::string& error);
// Opt-in SC117 IQ2_S/IQ4_NL scalar diagnostic; caller authenticates the blob.
// Unweighted f64 outputs; invalid input leaves output unchanged. Not GPU qualified.
bool scalar_reference_iq2s_iq4nl(const std::vector<uint8_t>& blob,
                                  const std::vector<float>& input,
                                  std::vector<double>& output,
                                  std::string& error);
}
