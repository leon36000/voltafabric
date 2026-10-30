#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vf::native_ggml {
// Scoped activation preparation: caller retains this only for one input in one batch.
// Format remains explicit; it is not interchangeable with other quantization formats.
struct PreparedInput { int gate_up_type = -1; std::vector<uint8_t> quantized; };
bool prepare_input(int gate_up_type, const std::vector<float>& input,
                   PreparedInput& prepared, std::string& error);
bool forward_prepared(int gate_up_type, int down_type, const std::vector<uint8_t>& blob,
                      const PreparedInput& prepared, std::vector<float>& output, std::string& error);
// Experimental single-token CPU reference for the observed SC117 geometry.
// The caller owns and authenticates the GGML blob and applies route weights.
// On failure output is unchanged; no GPU or remote execution is performed.
bool forward(int gate_up_type, int down_type, const std::vector<uint8_t>& blob,
             const std::vector<float>& input, std::vector<float>& output, std::string& error);
// GGML Q4_K, Q6_K or Q8_0 matrix; caller authenticates the resident bytes.
// Output is staged atomically; this is a CPU reference, not a GPU claim.
bool matrix_forward(int type, size_t rows, size_t columns, const std::vector<uint8_t>& weights,
                    const std::vector<float>& input, std::vector<float>& output, std::string& error);
bool matrix_shape(int type, size_t rows, size_t columns, size_t length);

}
