#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vf::native_ggml {
// An activation is prepared once per token and reused only for that token.
// Invalid input leaves the output unchanged; this is not a model identity.
bool scalar_prepare_view(const float* input, size_t input_size,
                         std::vector<float>& output, std::string& error);
bool scalar_reference_prepared_view(int gate_up_type, int down_type,
                                    const uint8_t* blob, size_t blob_size,
                                    const float* prepared, size_t prepared_size,
                                    std::vector<double>& output, std::string& error);
// Borrowed buffers must remain valid and immutable for the whole call. Output
// changes only after successful calculation; the caller owns all buffers.
bool scalar_reference_view(int gate_up_type, int down_type,
                           const uint8_t* blob, size_t blob_size,
                           const float* input, size_t input_size,
                           std::vector<double>& output, std::string& error);
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
