#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vf::native_ggml {
// Experimental single-token CPU reference for the observed SC117 geometry.
// The caller owns and authenticates the GGML blob and applies route weights.
// On failure output is unchanged; no GPU or remote execution is performed.
bool forward(int gate_up_type, int down_type, const std::vector<uint8_t>& blob,
             const std::vector<float>& input, std::vector<float>& output, std::string& error);
}
