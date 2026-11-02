#include "llama-context.h"
#include "gguf.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "qwen2moe_llama_route_bridge.hpp"
#include "r688/rust-loader/r688_session_loader.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace vf::ggml;
constexpr size_t kHidden = 32;
constexpr size_t kExperts = 2;
constexpr size_t kSourceExperts = 4;
constexpr size_t kQ8RowBytes = 34;
constexpr size_t kExpertBytes = kHidden * kQ8RowBytes;

struct SourceTensorSpan {
    uint64_t data_offset;
    uint64_t relative_offset;
    uint64_t bytes;
};

struct ExpertCheckpointSlice {
    int32_t resident_expert;
    size_t resident_local;
    size_t checkpoint_expert;
};

// This is the exact R659 rank-local resident ordering [40, 3] -> [1, 0].
constexpr std::array<ExpertCheckpointSlice, kExperts> kRankLocalSlices{{
    {40, 0, 1},
    {3, 1, 0},
}};

std::vector<uint8_t> g_gate;
std::vector<uint8_t> g_up;
std::vector<uint8_t> g_down;
std::string g_source_path;
size_t g_rank_local_source_bytes = 0;
size_t g_source_tensor_bytes = 0;
std::array<SourceTensorSpan, 3> g_source_spans{};

void append_u32(std::vector<uint8_t> &out, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<uint8_t> &out, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_string(std::vector<uint8_t> &out, const std::string &value) {
    append_u64(out, value.size());
    out.insert(out.end(), value.begin(), value.end());
}

// Byte-for-byte layout recipe from R659's runtime fixture(false, 32): F16
// scale 1.0 and a single row-indexed signed Q8_0 code in each row.
std::vector<uint8_t> r659_q8_payload(size_t rows, size_t columns,
                                     size_t experts, int8_t first_code) {
    const size_t row_bytes = columns / 32 * kQ8RowBytes;
    std::vector<uint8_t> payload(rows * experts * row_bytes, 0);
    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t row = 0; row < rows; ++row) {
            const size_t row_offset = (expert * rows + row) * row_bytes;
            for (size_t block = 0; block < columns / 32; ++block) {
                const size_t block_offset = row_offset + block * kQ8RowBytes;
                payload[block_offset] = 0x00;
                payload[block_offset + 1] = 0x3c; // F16 1.0
                if (row < columns && row / 32 == block) {
                    payload[block_offset + 2 + row % 32] =
                        static_cast<uint8_t>(first_code + static_cast<int8_t>(expert));
                }
            }
        }
    }
    return payload;
}

struct FixtureTensor {
    std::string name;
    std::array<uint64_t, 3> dims;
    std::vector<uint8_t> payload;
};

std::vector<uint8_t> make_gguf_bytes(
        size_t expert_count,
        const std::array<std::vector<uint8_t>, 3> & payloads) {
    const std::array<std::string, 3> names{
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    };
    std::vector<FixtureTensor> tensors;
    tensors.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        tensors.push_back(FixtureTensor{
            names[i], {32, 32, expert_count}, payloads[i]});
    }
    std::vector<uint8_t> out{'G', 'G', 'U', 'F'};
    append_u32(out, 3);
    append_u64(out, tensors.size());
    append_u64(out, 5);

    append_string(out, "general.architecture");
    append_u32(out, 8); // GGUF_TYPE_STRING
    append_string(out, "qwen2moe");
    append_string(out, "qwen2moe.block_count");
    append_u32(out, 4); // GGUF_TYPE_UINT32
    append_u32(out, 1);
    append_string(out, "qwen2moe.embedding_length");
    append_u32(out, 4);
    append_u32(out, 32);
    append_string(out, "qwen2moe.expert_count");
    append_u32(out, 4);
    append_u32(out, static_cast<uint32_t>(expert_count));
    append_string(out, "qwen2moe.expert_feed_forward_length");
    append_u32(out, 4);
    append_u32(out, 32);

    uint64_t relative = 0;
    for (const auto & tensor : tensors) {
        append_string(out, tensor.name);
        append_u32(out, 3);
        for (const uint64_t dim : tensor.dims) append_u64(out, dim);
        append_u32(out, 8); // GGML_TYPE_Q8_0
        append_u64(out, relative);
        relative += tensor.payload.size();
    }
    while (out.size() % 32 != 0) out.push_back(0);
    for (const auto & tensor : tensors) {
        out.insert(out.end(), tensor.payload.begin(), tensor.payload.end());
    }
    return out;
}

std::array<std::vector<uint8_t>, 3> make_source_payloads() {
    return {
        r659_q8_payload(32, 32, kSourceExperts, 1),
        r659_q8_payload(32, 32, kSourceExperts, 2),
        r659_q8_payload(32, 32, kSourceExperts, 1),
    };
}

std::vector<uint8_t> make_r659_source_fixture_bytes() {
    return make_gguf_bytes(kSourceExperts, make_source_payloads());
}

bool write_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !file.write(reinterpret_cast<const char *>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size()))) return false;
    file.flush();
    return static_cast<bool>(file);
}

std::array<std::vector<uint8_t>, 3> load_rank_local_slices(
        const std::string & source_path,
        const std::array<uint64_t, 3> & relative_offsets,
        uint64_t data_offset,
        size_t expert_count) {
    std::array<std::vector<uint8_t>, 3> payloads;
    std::ifstream file(source_path, std::ios::binary);
    if (!file) throw std::runtime_error("could not reopen source GGUF for rank-local slices");
    for (auto & payload : payloads) payload.reserve(kExperts * kExpertBytes);
    for (size_t tensor = 0; tensor < payloads.size(); ++tensor) {
        std::array<bool, kExperts> seen_local{};
        for (const auto & slice : kRankLocalSlices) {
            if (slice.resident_local >= kExperts || slice.checkpoint_expert >= expert_count) {
                throw std::runtime_error("R659 slice descriptor is outside the source layout");
            }
            if (seen_local[slice.resident_local] ||
                payloads[tensor].size() != slice.resident_local * kExpertBytes) {
                throw std::runtime_error("R659 resident-local order is duplicate or incomplete");
            }
            seen_local[slice.resident_local] = true;
            const uint64_t offset = data_offset + relative_offsets[tensor] +
                static_cast<uint64_t>(slice.checkpoint_expert * kExpertBytes);
            file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
            if (!file) throw std::runtime_error("R659 slice seek failed");
            const size_t begin = payloads[tensor].size();
            payloads[tensor].resize(begin + kExpertBytes);
            file.read(reinterpret_cast<char *>(payloads[tensor].data() + begin),
                      static_cast<std::streamsize>(kExpertBytes));
            if (file.gcount() != static_cast<std::streamsize>(kExpertBytes)) {
                throw std::runtime_error("R659 slice read was incomplete");
            }
            g_rank_local_source_bytes += kExpertBytes;
        }
    }
    return payloads;
}

bool load_r659_rank_local_fixture(const std::string & directory) {
    if (directory.empty()) return false;
    std::filesystem::create_directories(directory);
    g_source_path = (std::filesystem::path(directory) / "r659-qwen2moe-source-4expert-v3.gguf").string();
    const auto compact_path = (std::filesystem::path(directory) / "r659-qwen2moe-rank-local-v3.gguf").string();
    auto source_bytes = make_r659_source_fixture_bytes();
    const size_t expected_source_file_bytes = source_bytes.size();
    if (!write_bytes(g_source_path, source_bytes)) return false;
    std::vector<uint8_t>().swap(source_bytes);

    ggml_context * unused_context = nullptr;
    gguf_init_params metadata_params{true, &unused_context};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> source_metadata(
        gguf_init_from_file(g_source_path.c_str(), metadata_params), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> unused_tensors(unused_context, ggml_free);
    if (!source_metadata || gguf_get_version(source_metadata.get()) != 3 ||
        gguf_get_n_tensors(source_metadata.get()) != 3 ||
        std::filesystem::file_size(g_source_path) != expected_source_file_bytes) {
        std::cerr << "R659 source fixture was not parsed as the expected GGUF v3\n";
        return false;
    }
    const int64_t arch = gguf_find_key(source_metadata.get(), "general.architecture");
    const int64_t hidden = gguf_find_key(source_metadata.get(), "qwen2moe.embedding_length");
    const int64_t experts = gguf_find_key(source_metadata.get(), "qwen2moe.expert_count");
    if (arch < 0 || hidden < 0 || experts < 0 ||
        std::string(gguf_get_val_str(source_metadata.get(), arch)) != "qwen2moe" ||
        gguf_get_val_u32(source_metadata.get(), hidden) != kHidden ||
        gguf_get_val_u32(source_metadata.get(), experts) != kSourceExperts) {
        std::cerr << "R659 source metadata did not round-trip through the donor parser\n";
        return false;
    }

    const std::array<const char *, 3> names{
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    };
    std::array<uint64_t, 3> offsets{};
    const uint64_t source_data_offset = gguf_get_data_offset(source_metadata.get());
    for (size_t i = 0; i < names.size(); ++i) {
        const int64_t id = gguf_find_tensor(source_metadata.get(), names[i]);
        if (id < 0 || gguf_get_tensor_type(source_metadata.get(), id) != GGML_TYPE_Q8_0 ||
            gguf_get_tensor_size(source_metadata.get(), id) != kExpertBytes * kSourceExperts) {
            std::cerr << "R659 source Q8_0 matrix span was not validated\n";
            return false;
        }
        const int64_t * dims = gguf_get_tensor_ne(source_metadata.get(), id);
        if (dims == nullptr || dims[0] != 32 || dims[1] != 32 || dims[2] != kSourceExperts) {
            std::cerr << "R659 source matrix shape was not validated\n";
            return false;
        }
        offsets[i] = gguf_get_tensor_offset(source_metadata.get(), id);
        if (offsets[i] != i * kExpertBytes * kSourceExperts) {
            std::cerr << "R659 source matrix order/span changed\n";
            return false;
        }
        g_source_tensor_bytes += gguf_get_tensor_size(source_metadata.get(), id);
        g_source_spans[i] = SourceTensorSpan{
            source_data_offset, offsets[i], static_cast<uint64_t>(gguf_get_tensor_size(source_metadata.get(), id))};
    }

    const auto local_payloads = load_rank_local_slices(
        g_source_path, offsets, source_data_offset, kSourceExperts);
    for (const auto & payload : local_payloads) {
        if (payload.size() != kExperts * kExpertBytes) return false;
    }
    if (g_rank_local_source_bytes * 2 != g_source_tensor_bytes) {
        std::cerr << "rank-local read did not stay below full checkpoint tensor coverage\n";
        return false;
    }
    const auto compact_bytes = make_gguf_bytes(kExperts, local_payloads);
    if (!write_bytes(compact_path, compact_bytes)) return false;

    ggml_context * tensor_context = nullptr;
    gguf_init_params compact_params{false, &tensor_context};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> compact_metadata(
        gguf_init_from_file(compact_path.c_str(), compact_params), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> compact_tensors(tensor_context, ggml_free);
    if (!compact_metadata || !compact_tensors ||
        gguf_get_version(compact_metadata.get()) != 3 ||
        gguf_get_n_tensors(compact_metadata.get()) != 3 ||
        std::filesystem::file_size(compact_path) != compact_bytes.size()) {
        std::cerr << "rank-local GGUF was not parsed as the expected compact GGUF v3\n";
        return false;
    }
    const int64_t local_experts = gguf_find_key(compact_metadata.get(), "qwen2moe.expert_count");
    if (local_experts < 0 || gguf_get_val_u32(compact_metadata.get(), local_experts) != kExperts) {
        std::cerr << "rank-local GGUF expert count did not match the resident slices\n";
        return false;
    }
    auto load_tensor = [&](const char * name, std::vector<uint8_t> & destination,
                           size_t tensor_index) {
        const int64_t id = gguf_find_tensor(compact_metadata.get(), name);
        if (id < 0 || gguf_get_tensor_type(compact_metadata.get(), id) != GGML_TYPE_Q8_0 ||
            gguf_get_tensor_size(compact_metadata.get(), id) != kExpertBytes * kExperts) return false;
        const int64_t * dims = gguf_get_tensor_ne(compact_metadata.get(), id);
        if (dims == nullptr || dims[0] != 32 || dims[1] != 32 || dims[2] != kExperts) return false;
        const ggml_tensor * tensor = ggml_get_tensor(compact_tensors.get(), name);
        if (tensor == nullptr || tensor->type != GGML_TYPE_Q8_0 ||
            !ggml_is_contiguous(tensor) || ggml_nbytes(tensor) != kExpertBytes * kExperts) return false;
        const auto * begin = static_cast<const uint8_t *>(tensor->data);
        destination.assign(begin, begin + ggml_nbytes(tensor));
        return destination == local_payloads[tensor_index];
    };
    if (!load_tensor(names[0], g_gate, 0) || !load_tensor(names[1], g_up, 1) ||
        !load_tensor(names[2], g_down, 2)) {
        std::cerr << "rank-local Q8_0 slices did not round-trip through donor GGUF parser\n";
        return false;
    }
    std::cout << "R659 rank-local slice load: read " << g_rank_local_source_bytes
              << " of " << g_source_tensor_bytes << " checkpoint tensor bytes; donor GGUF has "
              << kExperts << " resident experts\n";
    return true;
}
uint16_t float_to_fp16_rne(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t exponent = (bits >> 23) & 0xffU;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        return static_cast<uint16_t>(sign | (mantissa == 0 ? 0x7c00U : 0x7e00U));
    }
    int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const uint32_t midpoint = 1U << (shift - 1U);
        if (remainder > midpoint || (remainder == midpoint && (half_mantissa & 1U) != 0)) {
            ++half_mantissa;
        }
        return static_cast<uint16_t>(sign | half_mantissa);
    }
    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0)) {
        ++half_mantissa;
        if (half_mantissa == 0x400U) {
            half_mantissa = 0;
            ++half_exponent;
            if (half_exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(half_exponent) << 10) |
                                 static_cast<uint16_t>(half_mantissa));
}

float fp16_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exponent = (value >> 10) & 0x1fU;
    uint32_t mantissa = value & 0x03ffU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t e = -14;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x3ffU;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

double fp16_roundtrip(double value) {
    return static_cast<double>(fp16_to_float(float_to_fp16_rne(static_cast<float>(value))));
}

std::vector<double> scalar_q8_matvec(const std::string & source_path,
                                     const SourceTensorSpan & span,
                                     size_t expert_count,
                                     size_t checkpoint_expert,
                                     size_t rows,
                                     size_t columns,
                                     const std::vector<float> & input) {
    if (input.size() != columns || expert_count == 0 || checkpoint_expert >= expert_count ||
        columns % 32 != 0 || rows * expert_count * (columns / 32) * kQ8RowBytes != span.bytes) {
        throw std::runtime_error("scalar Q8_0 oracle received an invalid source span");
    }
    const size_t expert_bytes = rows * (columns / 32) * kQ8RowBytes;
    const uint64_t offset = span.data_offset + span.relative_offset +
                            static_cast<uint64_t>(checkpoint_expert * expert_bytes);
    std::ifstream file(source_path, std::ios::binary);
    if (!file) throw std::runtime_error("scalar Q8_0 oracle could not open source fixture");
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::vector<uint8_t> bytes(expert_bytes);
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (file.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("scalar Q8_0 oracle read an incomplete expert slice");
    }
    const size_t row_bytes = columns / 32 * kQ8RowBytes;
    std::vector<double> output(rows, 0.0);
    for (size_t row = 0; row < rows; ++row) {
        const size_t row_start = row * row_bytes;
        for (size_t block = 0; block < columns / 32; ++block) {
            const size_t block_start = row_start + block * kQ8RowBytes;
            const uint16_t scale_bits = static_cast<uint16_t>(bytes[block_start]) |
                static_cast<uint16_t>(bytes[block_start + 1] << 8);
            const double scale = static_cast<double>(fp16_to_float(scale_bits));
            for (size_t lane = 0; lane < 32; ++lane) {
                const int8_t quant = static_cast<int8_t>(bytes[block_start + 2 + lane]);
                output[row] += static_cast<double>(input[block * 32 + lane]) * scale * quant;
            }
        }
    }
    return output;
}

struct Q8ActivationBlock {
    float scale = 0.0f;
    std::array<int8_t, 32> codes{};
};

std::vector<Q8ActivationBlock> quantize_activation_q8_0(const std::vector<float> & input) {
    if (input.size() % 32 != 0) throw std::runtime_error("Q8_0 activation width is not block aligned");
    std::vector<Q8ActivationBlock> blocks(input.size() / 32);
    for (size_t block = 0; block < blocks.size(); ++block) {
        float amax = 0.0f;
        for (size_t lane = 0; lane < 32; ++lane) {
            amax = std::max(amax, std::fabs(input[block * 32 + lane]));
        }
        const float scale = amax / 127.0f;
        const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
        blocks[block].scale = static_cast<float>(fp16_roundtrip(scale));
        for (size_t lane = 0; lane < 32; ++lane) {
            const float quantized = std::round(input[block * 32 + lane] * inverse);
            if (quantized < -127.0f || quantized > 127.0f) {
                throw std::runtime_error("Q8_0 activation quantizer exceeded signed code range");
            }
            blocks[block].codes[lane] = static_cast<int8_t>(quantized);
        }
    }
    return blocks;
}

std::vector<double> scalar_q8_matvec_with_q8_0_activation(
        const std::string & source_path, const SourceTensorSpan & span,
        size_t expert_count, size_t checkpoint_expert, size_t rows, size_t columns,
        const std::vector<float> & input) {
    if (input.size() != columns || columns % 32 != 0) {
        throw std::runtime_error("Q8_0 dot oracle received an invalid activation shape");
    }
    const size_t expert_bytes = rows * (columns / 32) * kQ8RowBytes;
    if (expert_count == 0 || checkpoint_expert >= expert_count ||
        rows * expert_count * (columns / 32) * kQ8RowBytes != span.bytes) {
        throw std::runtime_error("Q8_0 dot oracle received an invalid expert span");
    }
    const uint64_t offset = span.data_offset + span.relative_offset +
                            static_cast<uint64_t>(checkpoint_expert * expert_bytes);
    std::ifstream file(source_path, std::ios::binary);
    if (!file) throw std::runtime_error("Q8_0 dot oracle could not open source fixture");
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::vector<uint8_t> bytes(expert_bytes);
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (file.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("Q8_0 dot oracle read an incomplete weight slice");
    }
    const auto activation_blocks = quantize_activation_q8_0(input);
    const size_t row_bytes = columns / 32 * kQ8RowBytes;
    std::vector<double> output(rows, 0.0);
    for (size_t row = 0; row < rows; ++row) {
        float sumf = 0.0f;
        for (size_t block = 0; block < columns / 32; ++block) {
            const size_t block_start = row * row_bytes + block * kQ8RowBytes;
            const uint16_t weight_scale_bits = static_cast<uint16_t>(bytes[block_start]) |
                static_cast<uint16_t>(bytes[block_start + 1] << 8);
            int32_t integer_sum = 0;
            for (size_t lane = 0; lane < 32; ++lane) {
                const int8_t weight_code = static_cast<int8_t>(bytes[block_start + 2 + lane]);
                integer_sum += static_cast<int32_t>(weight_code) *
                               static_cast<int32_t>(activation_blocks[block].codes[lane]);
            }
            const float weight_scale = fp16_to_float(weight_scale_bits);
            sumf += static_cast<float>(integer_sum) *
                    (weight_scale * activation_blocks[block].scale);
        }
        output[row] = static_cast<double>(sumf);
    }
    return output;
}

struct ScalarExpertResult {
    std::vector<double> gate;
    std::vector<double> up;
    std::vector<float> activation;
    std::vector<double> down;
    std::vector<double> fp16_slot;
};

ScalarExpertResult scalar_expert_ffn(const size_t local_expert, const float route_weight,
                                     const std::array<SourceTensorSpan, 3> & spans) {
    if (local_expert >= kRankLocalSlices.size()) {
        throw std::runtime_error("scalar oracle local expert is out of range");
    }
    const auto & resident = kRankLocalSlices[local_expert];
    std::vector<float> input(kHidden, 0.0f);
    input[0] = 1.0f;
    ScalarExpertResult result;
    result.gate = scalar_q8_matvec(g_source_path, spans[0], kSourceExperts,
                                   resident.checkpoint_expert, kHidden, kHidden, input);
    result.up = scalar_q8_matvec(g_source_path, spans[1], kSourceExperts,
                                 resident.checkpoint_expert, kHidden, kHidden, input);
    result.activation.resize(kHidden);
    for (size_t i = 0; i < kHidden; ++i) {
        const float gate = static_cast<float>(fp16_roundtrip(result.gate[i]));
        const float up = static_cast<float>(fp16_roundtrip(result.up[i]));
        result.activation[i] = (gate / (1.0f + std::exp(-gate))) * up;
    }
    result.down = scalar_q8_matvec(g_source_path, spans[2], kSourceExperts,
                                   resident.checkpoint_expert, kHidden, kHidden,
                                   result.activation);
    result.fp16_slot.resize(kHidden);
    for (size_t i = 0; i < kHidden; ++i) {
        result.fp16_slot[i] = fp16_roundtrip(result.down[i]) * static_cast<double>(route_weight);
    }
    return result;
}

struct ScalarRoundControl {
    const char * name;
    bool half_gate;
    bool half_up;
    bool half_activation;
    bool q8_0_linear_inputs;
};

constexpr std::array<ScalarRoundControl, 7> kRoundControls{{
    {"fp32_intermediates", false, false, false, false},
    {"fp16_gate_only", true, false, false, false},
    {"fp16_up_only", false, true, false, false},
    {"fp16_gate_and_up", true, true, false, false},
    {"fp16_activation_only", false, false, true, false},
    {"fp16_gate_up_activation", true, true, true, false},
    {"fp32_intermediates_q8_0_linear_inputs", false, false, false, true},
}};

std::vector<double> scalar_collector_control(size_t local_expert, float route_weight,
                                             const ScalarRoundControl & control,
                                             std::array<std::vector<float>, 3> * stages = nullptr) {
    const auto & resident = kRankLocalSlices.at(local_expert);
    std::vector<float> input(kHidden, 0.0f);
    input[0] = 1.0f;
    const auto matvec = control.q8_0_linear_inputs
        ? scalar_q8_matvec_with_q8_0_activation : scalar_q8_matvec;
    const auto gate = matvec(g_source_path, g_source_spans[0], kSourceExperts,
                             resident.checkpoint_expert, kHidden, kHidden, input);
    const auto up = matvec(g_source_path, g_source_spans[1], kSourceExperts,
                           resident.checkpoint_expert, kHidden, kHidden, input);
    std::vector<float> gate_f32(kHidden), up_f32(kHidden), activation(kHidden);
    for (size_t i = 0; i < kHidden; ++i) {
        gate_f32[i] = static_cast<float>(gate[i]);
        up_f32[i] = static_cast<float>(up[i]);
        if (control.half_gate) gate_f32[i] = static_cast<float>(fp16_roundtrip(gate_f32[i]));
        if (control.half_up) up_f32[i] = static_cast<float>(fp16_roundtrip(up_f32[i]));
        activation[i] = (gate_f32[i] / (1.0f + ::expf(-gate_f32[i]))) * up_f32[i];
        if (control.half_activation) {
            activation[i] = static_cast<float>(fp16_roundtrip(activation[i]));
        }
    }
    const auto down = matvec(g_source_path, g_source_spans[2], kSourceExperts,
                             resident.checkpoint_expert, kHidden, kHidden, activation);
    std::vector<double> slots(kHidden);
    for (size_t i = 0; i < kHidden; ++i) {
        slots[i] = fp16_roundtrip(down[i]) * static_cast<double>(route_weight);
    }
    if (stages != nullptr) {
        (*stages)[0] = std::move(gate_f32);
        (*stages)[1] = std::move(up_f32);
        (*stages)[2] = std::move(activation);
    }
    return slots;
}

struct EvalTensorSnapshot {
    std::string label;
    std::string op;
    std::vector<float> values;
};

struct RouteCapture {
    uint64_t batch_id = 0;
    uint64_t attempt_id = 0;
    int32_t checkpoint_expert = -1;
    int32_t registry_expert = -1;
    float route_weight = 0.0f;
    bool reject = false;
    bool duplicate_projection = false;
    size_t calls = 0;
    BatchIdentity identity;
    std::vector<int32_t> experts;
    std::vector<int32_t> owners;
    std::vector<float> activations;
    std::vector<float> weights;
    std::vector<float> raw_slots;
    std::vector<float> output;
    std::vector<double> fp16_collector_output;
    std::vector<double> session_fp32_output;
    bool capture_intermediates = false;
    bool use_session_collector = false;
    void *session_context = nullptr;
    int32_t session_status = -1;
    std::vector<EvalTensorSnapshot> intermediates;
    bool scalar_collector_match = true;
    double scalar_collector_max_abs_error = 0.0;
};

std::vector<float> scalar_donor_fp32_top2_oracle(const RouteCapture &capture) {
    if (capture.experts.size() != 2 || capture.weights.size() != 2) {
        throw std::runtime_error("FP32 scalar oracle needs the complete top-2 route");
    }
    std::vector<float> output(kHidden, 0.0f);
    std::vector<float> input(kHidden, 0.0f);
    input[0] = 1.0f;
    for (size_t slot = 0; slot < capture.experts.size(); ++slot) {
        const size_t local_expert = capture.experts[slot] == 40 ? 0
            : capture.experts[slot] == 3 ? 1
            : throw std::runtime_error("FP32 scalar oracle received an unmapped expert");
        const size_t checkpoint_expert = kRankLocalSlices[local_expert].checkpoint_expert;
        const auto gate = scalar_q8_matvec_with_q8_0_activation(g_source_path, g_source_spans[0],
            kSourceExperts, checkpoint_expert, kHidden, kHidden, input);
        const auto up = scalar_q8_matvec_with_q8_0_activation(g_source_path, g_source_spans[1],
            kSourceExperts, checkpoint_expert, kHidden, kHidden, input);
        std::vector<float> activation(kHidden);
        for (size_t i = 0; i < kHidden; ++i) {
            const float gate_value = static_cast<float>(gate[i]);
            const float up_value = static_cast<float>(up[i]);
            activation[i] = (gate_value / (1.0f + ::expf(-gate_value))) * up_value;
        }
        const auto down = scalar_q8_matvec_with_q8_0_activation(g_source_path, g_source_spans[2],
            kSourceExperts, checkpoint_expert, kHidden, kHidden, activation);
        for (size_t i = 0; i < kHidden; ++i) {
            const float weighted = static_cast<float>(down[i]) * capture.weights[slot];
            output[i] = output[i] + weighted;
        }
    }
    return output;
}

bool capture_graph_tensor(ggml_tensor * tensor, bool ask, void * userdata) {
    if (tensor == nullptr || userdata == nullptr) return false;
    auto & capture = *static_cast<RouteCapture *>(userdata);
    if (!capture.capture_intermediates) return false;
    const bool glu = tensor->op == GGML_OP_GLU;
    std::string weight_name;
    if (tensor->op == GGML_OP_MUL_MAT_ID && tensor->src[0] != nullptr) {
        weight_name = tensor->src[0]->name;
    }
    const bool matmul_id = tensor->op == GGML_OP_MUL_MAT_ID;
    if (!glu && !matmul_id) return false;
    if (ask) return true;
    if (tensor->type != GGML_TYPE_F32 || tensor->data == nullptr ||
        !ggml_is_contiguous(tensor) || ggml_nbytes(tensor) % sizeof(float) != 0) {
        return false;
    }
    EvalTensorSnapshot snapshot{};
    snapshot.label = glu ? "swiglu" : (weight_name.empty() ? "mul_mat_id" : weight_name);
    snapshot.op = ggml_op_name(tensor->op);
    const size_t bytes = ggml_nbytes(tensor);
    snapshot.values.resize(bytes / sizeof(float));
    std::memcpy(snapshot.values.data(), tensor->data, bytes);
    capture.intermediates.push_back(std::move(snapshot));
    return true;
}

class R659MoEInputs final : public llm_graph_input_i {
public:
    R659MoEInputs(ggml_tensor *activations, ggml_tensor *logits,
                  ggml_tensor *gate, ggml_tensor *up, ggml_tensor *down,
                  int32_t checkpoint_expert, bool two_routes)
        : activations_(activations), logits_(logits), gate_(gate),
          up_(up), down_(down), checkpoint_expert_(checkpoint_expert),
          two_routes_(two_routes) {}

    void set_input(const llama_ubatch *ubatch) override {
        if (ubatch == nullptr || ubatch->n_tokens != 1) return;
        std::array<float, kHidden> activations{};
        activations[0] = 1.0f;
        const std::array<float, 2> logits = two_routes_
            ? std::array<float, 2>{std::log(0.25f / 0.75f), std::log(0.75f / 0.25f)}
            : checkpoint_expert_ == 0
                ? std::array<float, 2>{std::log(0.25f / 0.75f), std::log(0.20f / 0.80f)}
                : std::array<float, 2>{std::log(0.20f / 0.80f), std::log(0.75f / 0.25f)};
        ggml_backend_tensor_set(activations_, activations.data(), 0, sizeof(activations));
        ggml_backend_tensor_set(logits_, logits.data(), 0, sizeof(logits));
        ggml_backend_tensor_set(gate_, g_gate.data(), 0, g_gate.size());
        ggml_backend_tensor_set(up_, g_up.data(), 0, g_up.size());
        ggml_backend_tensor_set(down_, g_down.data(), 0, g_down.size());
    }

private:
    ggml_tensor *activations_;
    ggml_tensor *logits_;
    ggml_tensor *gate_;
    ggml_tensor *up_;
    ggml_tensor *down_;
    int32_t checkpoint_expert_;
    bool two_routes_;
};

class R659Q8MoEGraph final : public llm_graph_context {
public:
    R659Q8MoEGraph(const llm_graph_params &params, int32_t checkpoint_expert,
                   bool two_routes)
        : llm_graph_context(params) {
        auto *activations = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kHidden, 1);
        auto *logits = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 1);
        auto *gate = ggml_new_tensor_3d(ctx0, GGML_TYPE_Q8_0, 32, 32, 2);
        auto *up = ggml_new_tensor_3d(ctx0, GGML_TYPE_Q8_0, 32, 32, 2);
        auto *down = ggml_new_tensor_3d(ctx0, GGML_TYPE_Q8_0, 32, 32, 2);
        if (!activations || !logits || !gate || !up || !down) {
            throw std::runtime_error("could not allocate Q8_0 donor graph tensors");
        }
        ggml_set_name(gate, "r687_gate_weights");
        ggml_set_name(up, "r687_up_weights");
        ggml_set_name(down, "r687_down_weights");
        for (auto *input : {activations, logits, gate, up, down}) ggml_set_input(input);
        res->add_input(std::make_unique<R659MoEInputs>(
            activations, logits, gate, up, down, checkpoint_expert, two_routes));
        ggml_tensor *routed = build_moe_ffn(
            activations, nullptr, up, gate, down, nullptr,
            2, two_routes ? 2 : 1, LLM_FFN_SILU, false, 1.0f,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, 0,
            logits, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (routed == nullptr) throw std::runtime_error("donor build_moe_ffn returned null");
        res->t_embd = routed;
        ggml_build_forward_expand(gf, routed);
    }
};

class R659Qwen2MoEModel final : public llama_model {
public:
    explicit R659Qwen2MoEModel(int32_t checkpoint_expert, bool two_routes = false)
        : llama_model(llama_model_default_params()), checkpoint_expert_(checkpoint_expert),
          two_routes_(two_routes) {
        arch = LLM_ARCH_QWEN2MOE;
        hparams.vocab_only = false;
        hparams.n_ctx_train = 256;
        hparams.n_embd = kHidden;
        hparams.n_embd_out_impl = kHidden;
        hparams.n_layer_all = 1;
        hparams.n_layer_nextn = 0;
        hparams.n_expert = 2;
        hparams.n_expert_used_arr[0] = two_routes_ ? 2 : 1;
        hparams.n_head_arr[0] = 1;
        hparams.n_head_kv_arr[0] = 1;
        hparams.n_layer_kv_from_start = 0;
        hparams.causal_attn = true;
        layers.resize(1);
    }

    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return false; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(
            const llm_graph_params &params) const override {
        return std::make_unique<R659Q8MoEGraph>(params, checkpoint_expert_, two_routes_);
    }

private:
    int32_t checkpoint_expert_;
    bool two_routes_;
};

bool make_batch(void *userdata, const llama_ubatch &ubatch, llm_graph_type graph_type,
                uint32_t expected_layers, RouteBatch &batch) {
    auto &capture = *static_cast<RouteCapture *>(userdata);
    if (ubatch.n_tokens != 1 || graph_type != LLM_GRAPH_TYPE_DEFAULT || expected_layers != 1) {
        return false;
    }
    if (capture.use_session_collector) {
        batch.identity = BatchIdentity{"r688-same-process", "r659-qwen2moe-cpu-fixture",
                                       "r688-r659-rank-local-plan", 0, 9,
                                       capture.batch_id, capture.attempt_id};
    } else {
        batch.identity = BatchIdentity{"r687-r659-rank-local", "r659-qwen2moe-rank-local-q8-fixture",
                                       "resident-index-to-registry-cpu", 0, 687,
                                       capture.batch_id, capture.batch_id};
    }
    batch.checkpoint_expert_to_registry_expert = capture.duplicate_projection
        ? std::vector<int32_t>{40, 40} : std::vector<int32_t>{40, 3};
    batch.owner_by_expert.assign(41, -1);
    batch.owner_by_expert[3] = 0;
    batch.owner_by_expert[40] = 0;
    return true;
}

RouteStatus collect(const RouteBatch &batch, const float *activations, size_t activation_count,
                    const float *slots, size_t slot_count, float *output, size_t output_count,
                    void *userdata) {
    auto &capture = *static_cast<RouteCapture *>(userdata);
    ++capture.calls;
    capture.identity = batch.identity;
    capture.experts = batch.expert_ids;
    capture.owners = batch.owner_kinds;
    capture.weights = batch.weights;
    if (activations) capture.activations.assign(activations, activations + activation_count);
    if (slots) capture.raw_slots.assign(slots, slots + slot_count);
    if (capture.use_session_collector) {
        if (activations == nullptr || slots == nullptr || output == nullptr ||
            capture.session_context == nullptr || batch.hidden != kHidden ||
            batch.slots != 2 || batch.tokens != 1 || activation_count != kHidden ||
            slot_count != kHidden * 2 || output_count != kHidden ||
            batch.expert_ids.size() != 2 || batch.weights.size() != 2 ||
            batch.owner_kinds != std::vector<int32_t>{0, 0} ||
            capture.activations[0] != 1.0f ||
            std::any_of(capture.activations.begin() + 1, capture.activations.end(),
                        [](float value) { return value != 0.0f; })) {
            return RouteStatus::invalid_batch;
        }
        if (capture.reject) return RouteStatus::provider_failure;
        std::vector<double> staged(output_count, std::numeric_limits<double>::quiet_NaN());
        const int32_t result = vf_r688_run_routed_layer(
            capture.session_context, batch.identity.run.c_str(), batch.identity.model.c_str(),
            batch.identity.plan.c_str(), batch.identity.layer, batch.identity.epoch,
            batch.identity.batch, batch.identity.attempt, batch.hidden, batch.slots,
            batch.tokens, batch.expert_ids.data(), batch.expert_ids.size(),
            batch.weights.data(), batch.weights.size(), activations, activation_count,
            staged.data(), staged.size());
        capture.session_status = result;
        if (result != 0 ||
            std::any_of(staged.begin(), staged.end(), [](double value) { return !std::isfinite(value); })) {
            return RouteStatus::provider_failure;
        }
        for (size_t i = 0; i < output_count; ++i) {
            const float value = static_cast<float>(staged[i]);
            if (!std::isfinite(value)) return RouteStatus::invalid_result;
            output[i] = value;
        }
        capture.session_fp32_output = staged;
        capture.output.assign(output, output + output_count);
        return RouteStatus::ready;
    }
    if (activations == nullptr || slots == nullptr || output == nullptr ||
        batch.hidden != kHidden || batch.slots != 1 || batch.tokens != 1 ||
        activation_count != kHidden || slot_count != kHidden || output_count != kHidden ||
        batch.expert_ids != std::vector<int32_t>{capture.registry_expert} ||
        batch.owner_kinds != std::vector<int32_t>{0} || batch.weights.size() != 1 ||
        std::fabs(batch.weights[0] - capture.route_weight) > 1.0e-6f ||
        capture.activations[0] != 1.0f ||
        std::any_of(capture.activations.begin() + 1, capture.activations.end(),
                    [](float value) { return value != 0.0f; })) {
        return RouteStatus::invalid_batch;
    }
    if (capture.reject) return RouteStatus::provider_failure;
    capture.fp16_collector_output.resize(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        capture.fp16_collector_output[i] = fp16_roundtrip(slots[i]) * batch.weights[0];
        output[i] = static_cast<float>(capture.fp16_collector_output[i]);
    }
    capture.output.assign(output, output + output_count);
    return RouteStatus::ready;
}

bool near_oracle(float actual, float expected) {
    const float tolerance = 1.0e-6f + 1.0e-3f * std::fabs(expected);
    return std::isfinite(actual) && std::fabs(actual - expected) <= tolerance;
}

bool run_graph_case(uint64_t batch_id, int32_t checkpoint_expert, int32_t registry_expert,
                    float route_weight, bool reject, bool duplicate_projection = false,
                    RouteCapture * completed = nullptr, bool * graph_completed = nullptr,
                    bool capture_intermediates = false) {
    if (graph_completed != nullptr) *graph_completed = false;
    R659Qwen2MoEModel model(checkpoint_expert);
    RouteCapture capture;
    capture.batch_id = batch_id;
    capture.checkpoint_expert = checkpoint_expert;
    capture.registry_expert = registry_expert;
    capture.route_weight = route_weight;
    capture.reject = reject;
    capture.duplicate_projection = duplicate_projection;
    capture.capture_intermediates = capture_intermediates;
    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 1;
    params.n_ubatch = 1;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.embeddings = true;
    params.offload_kqv = false;
    params.op_offload = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.cb_eval = capture_intermediates ? capture_graph_tensor : nullptr;
    params.cb_eval_user_data = capture_intermediates ? &capture : nullptr;
    llama_context context(model, params);

    Qwen2MoELlamaRouteBridge bridge(make_batch, &capture, collect, &capture);
    if (!bridge.attach(context) || context.get_vf_route_hooks() != bridge.hooks()) return false;

    llama_ubatch ubatch{};
    ubatch.n_tokens = 1;
    ggml_status status = GGML_STATUS_FAILED;
    llm_graph_result *result = context.process_ubatch(
        ubatch, LLM_GRAPH_TYPE_DEFAULT, nullptr, status);
    const bool detached = bridge.detach();
    if (duplicate_projection) {
        return result == nullptr && status == GGML_STATUS_FAILED && detached &&
               context.get_vf_route_hooks() == nullptr && capture.calls == 0 &&
               capture.output.empty();
    }
    if (reject) {
        const bool refused = result == nullptr && status == GGML_STATUS_FAILED && detached &&
               context.get_vf_route_hooks() == nullptr && capture.calls == 1 &&
               capture.experts == std::vector<int32_t>{registry_expert} && capture.output.empty() &&
               capture.fp16_collector_output.empty();
        if (!refused) {
            std::cerr << "R659 rejected-route mismatch: result=" << (result != nullptr)
                      << " status=" << status << " detached=" << detached
                      << " callbacks=" << capture.calls << " published="
                      << !capture.output.empty() << '\n';
        }
        return refused;
    }
    if (result == nullptr || status != GGML_STATUS_SUCCESS || !detached ||
        context.get_vf_route_hooks() != nullptr || capture.calls != 1 ||
        capture.identity.run != "r687-r659-rank-local" || capture.identity.model != "r659-qwen2moe-rank-local-q8-fixture" ||
        capture.identity.plan != "resident-index-to-registry-cpu" || capture.identity.layer != 0 ||
        capture.identity.epoch != 687 || capture.identity.batch != batch_id ||
        capture.identity.attempt != batch_id ||
        capture.experts != std::vector<int32_t>{registry_expert} ||
        capture.weights.size() != 1 || std::fabs(capture.weights[0] - route_weight) > 1.0e-6f ||
        capture.raw_slots.size() != kHidden || capture.output.size() != kHidden) {
        std::cerr << "R659 route mismatch: result=" << (result != nullptr) << " status=" << status
                  << " detached=" << detached << " callbacks=" << capture.calls
                  << " checkpoint=" << checkpoint_expert << " mapped=" << registry_expert
                  << " captured_ids=" << (capture.experts.empty() ? -1 : capture.experts[0])
                  << " weight=" << (capture.weights.empty() ? -1.0f : capture.weights[0])
                  << " raw_n=" << capture.raw_slots.size()
                  << " out_n=" << capture.output.size() << '\n';
        return false;
    }

    const auto oracle = scalar_expert_ffn(static_cast<size_t>(checkpoint_expert), route_weight,
                                          g_source_spans);
    for (size_t i = 0; i < kHidden; ++i) {
        const float expected_raw = static_cast<float>(oracle.down[i]);
        const double expected_weighted = oracle.fp16_slot[i];
        if (!near_oracle(capture.raw_slots[i], expected_raw) ||
            std::fabs(capture.output[i] - static_cast<float>(capture.fp16_collector_output[i])) > 1.0e-6f) {
            std::cerr << "R659 numeric mismatch at " << i << ": raw=" << capture.raw_slots[i]
                      << " expected_raw=" << expected_raw << " out=" << capture.output[i]
                      << " callback_collector=" << capture.fp16_collector_output[i] << '\n';
            return false;
        }
        const double error = std::fabs(capture.fp16_collector_output[i] - expected_weighted);
        capture.scalar_collector_max_abs_error =
            std::max(capture.scalar_collector_max_abs_error, error);
        if (error > 1.0e-6) capture.scalar_collector_match = false;
    }
    if (graph_completed != nullptr) *graph_completed = true;
    if (completed != nullptr) *completed = capture;
    return true;
}

bool read_graph_embedding(llm_graph_result *result, std::vector<float> &values) {
    if (result == nullptr) return false;
    ggml_tensor *embedding = result->get_embd();
    if (embedding == nullptr || embedding->type != GGML_TYPE_F32 ||
        embedding->ne[0] != static_cast<int64_t>(kHidden) || embedding->ne[1] != 1 ||
        embedding->ne[2] != 1 || embedding->ne[3] != 1 ||
        static_cast<size_t>(ggml_nelements(embedding)) != kHidden) {
        return false;
    }
    values.resize(kHidden);
    ggml_backend_tensor_get(embedding, values.data(), 0, ggml_nbytes(embedding));
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

uint64_t fresh_session_attempt() {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (ns <= 0) throw std::runtime_error("system clock cannot create a positive Session attempt");
    return static_cast<uint64_t>(ns);
}

bool run_unhooked_top2_graph(std::vector<float> &output) {
    R659Qwen2MoEModel model(0, true);
    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 1;
    params.n_ubatch = 1;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.embeddings = true;
    params.offload_kqv = false;
    params.op_offload = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context context(model, params);
    llama_ubatch ubatch{};
    ubatch.n_tokens = 1;
    ggml_status status = GGML_STATUS_FAILED;
    llm_graph_result *result = context.process_ubatch(
        ubatch, LLM_GRAPH_TYPE_DEFAULT, nullptr, status);
    return status == GGML_STATUS_SUCCESS && read_graph_embedding(result, output);
}

bool run_session_hook_top2_graph(void *session_context, uint64_t batch_id,
                                 uint64_t attempt_id, bool reject,
                                 std::vector<float> &output, RouteCapture *completed = nullptr,
                                 bool expect_session_refusal = false) {
    R659Qwen2MoEModel model(0, true);
    RouteCapture capture;
    capture.batch_id = batch_id;
    capture.attempt_id = attempt_id;
    capture.checkpoint_expert = 0;
    capture.registry_expert = -1;
    capture.reject = reject;
    capture.use_session_collector = true;
    capture.session_context = session_context;
    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 1;
    params.n_ubatch = 1;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.embeddings = true;
    params.offload_kqv = false;
    params.op_offload = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context context(model, params);
    Qwen2MoELlamaRouteBridge bridge(make_batch, &capture, collect, &capture);
    if (!bridge.attach(context) || context.get_vf_route_hooks() != bridge.hooks()) return false;
    llama_ubatch ubatch{};
    ubatch.n_tokens = 1;
    ggml_status status = GGML_STATUS_FAILED;
    llm_graph_result *result = context.process_ubatch(
        ubatch, LLM_GRAPH_TYPE_DEFAULT, nullptr, status);
    const bool detached = bridge.detach();
    if (reject || expect_session_refusal) {
        return result == nullptr && status == GGML_STATUS_FAILED && detached &&
               context.get_vf_route_hooks() == nullptr && capture.calls == 1 &&
               capture.output.empty() && capture.fp16_collector_output.empty() &&
               (reject ? capture.session_status == -1 : capture.session_status == 2);
    }
    if (result == nullptr || status != GGML_STATUS_SUCCESS || !detached ||
        context.get_vf_route_hooks() != nullptr || capture.calls != 1 ||
        capture.identity.run != "r688-same-process" ||
        capture.identity.model != "r659-qwen2moe-cpu-fixture" ||
        capture.identity.plan != "r688-r659-rank-local-plan" ||
        capture.identity.layer != 0 || capture.identity.epoch != 9 ||
        capture.identity.batch != batch_id || capture.identity.attempt != attempt_id ||
        capture.experts.size() != 2 || capture.weights.size() != 2 ||
        capture.raw_slots.size() != 2 * kHidden || capture.output.size() != kHidden) {
        return false;
    }
    std::map<int32_t, float> route_weights;
    for (size_t i = 0; i < capture.experts.size(); ++i) {
        if (!route_weights.emplace(capture.experts[i], capture.weights[i]).second) return false;
    }
    if (route_weights.size() != 2 || std::fabs(route_weights[40] - 0.25f) > 1.0e-6f ||
        std::fabs(route_weights[3] - 0.75f) > 1.0e-6f) {
        return false;
    }
    if (!read_graph_embedding(result, output) || output.size() != capture.output.size()) return false;
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - capture.output[i]) > 1.0e-7f) return false;
    }
    if (completed != nullptr) *completed = std::move(capture);
    return true;
}

bool run_combined_layer(bool reject_second, std::vector<double> & published,
                        RouteCapture * first_result = nullptr,
                        RouteCapture * second_result = nullptr,
                        bool capture_intermediates = false) {
    std::vector<double> staged(kHidden, 0.0);
    constexpr std::array<float, 2> weights{0.25f, 0.75f};
    for (size_t local_expert = 0; local_expert < kExperts; ++local_expert) {
        RouteCapture capture;
        bool graph_completed = false;
        const int32_t registry_expert = kRankLocalSlices[local_expert].resident_expert;
        if (!run_graph_case(local_expert + 1, static_cast<int32_t>(local_expert),
                            registry_expert, weights[local_expert],
                            reject_second && local_expert == 1, false, &capture,
                            &graph_completed, capture_intermediates)) {
            return false;
        }
        if (!graph_completed) return false;
        if (local_expert == 0 && first_result != nullptr) *first_result = capture;
        if (local_expert == 1 && second_result != nullptr) *second_result = capture;
        for (size_t i = 0; i < kHidden; ++i) staged[i] += capture.fp16_collector_output[i];
    }
    published = std::move(staged);
    return true;
}

bool reject_partial_combined_layer() {
    std::vector<double> published;
    // A first mapped route succeeds and is staged; the second route fails. The
    // layer wrapper must return before moving its staging vector to publication.
    const bool complete = run_combined_layer(true, published);
    return !complete && published.empty();
}

double max_abs_diff(const std::vector<float> & left, const std::vector<double> & right) {
    if (left.size() != right.size()) return INFINITY;
    double maximum = 0.0;
    for (size_t i = 0; i < left.size(); ++i) {
        maximum = std::max(maximum,
            std::fabs(static_cast<double>(left[i]) - right[i]));
    }
    return maximum;
}

const EvalTensorSnapshot * find_snapshot(const RouteCapture & capture,
                                         const std::string & label) {
    const auto found = std::find_if(capture.intermediates.begin(), capture.intermediates.end(),
        [&](const EvalTensorSnapshot & snapshot) { return snapshot.label == label; });
    return found == capture.intermediates.end() ? nullptr : &*found;
}

bool report_route_rounding_controls(const RouteCapture & capture, size_t local_expert) {
    std::vector<const EvalTensorSnapshot *> matmuls;
    for (const auto & snapshot : capture.intermediates) {
        if (snapshot.op == "MUL_MAT_ID") matmuls.push_back(&snapshot);
        std::cout << "R687 donor_stage expert=" << capture.registry_expert
                  << " label=" << snapshot.label << " op=" << snapshot.op
                  << " elements=" << snapshot.values.size()
                  << " first=" << (snapshot.values.empty() ? 0.0f : snapshot.values[0]) << '\n';
    }
    const auto * gate_tensor = find_snapshot(capture, "r687_gate_weights");
    const auto * up_tensor = find_snapshot(capture, "r687_up_weights");
    const auto * down_tensor = find_snapshot(capture, "r687_down_weights");
    // The source names can be lost through donor graph views. In that case
    // build_moe_ffn's topological MUL_MAT_ID order is up, gate, down.
    if (matmuls.size() == 3) {
        if (up_tensor == nullptr) up_tensor = matmuls[0];
        if (gate_tensor == nullptr) gate_tensor = matmuls[1];
        if (down_tensor == nullptr) down_tensor = matmuls[2];
    }
    const auto * swiglu_tensor = find_snapshot(capture, "swiglu");
    if (gate_tensor == nullptr || up_tensor == nullptr || down_tensor == nullptr ||
        swiglu_tensor == nullptr || gate_tensor->values.size() != kHidden ||
        up_tensor->values.size() != kHidden || down_tensor->values.size() != kHidden ||
        swiglu_tensor->values.size() != kHidden) {
        std::cerr << "R687 did not capture gate/up/SwiGLU/down donor stages for expert "
                  << capture.registry_expert << '\n';
        return false;
    }

    std::vector<float> one_hot(kHidden, 0.0f);
    one_hot[0] = 1.0f;
    const auto scalar_gate = scalar_q8_matvec(g_source_path, g_source_spans[0], kSourceExperts,
        kRankLocalSlices[local_expert].checkpoint_expert, kHidden, kHidden, one_hot);
    const auto scalar_up = scalar_q8_matvec(g_source_path, g_source_spans[1], kSourceExperts,
        kRankLocalSlices[local_expert].checkpoint_expert, kHidden, kHidden, one_hot);
    const auto quantized_gate = scalar_q8_matvec_with_q8_0_activation(
        g_source_path, g_source_spans[0], kSourceExperts,
        kRankLocalSlices[local_expert].checkpoint_expert, kHidden, kHidden, one_hot);
    const auto quantized_up = scalar_q8_matvec_with_q8_0_activation(
        g_source_path, g_source_spans[1], kSourceExperts,
        kRankLocalSlices[local_expert].checkpoint_expert, kHidden, kHidden, one_hot);
    const auto scalar_down = scalar_q8_matvec(g_source_path, g_source_spans[2], kSourceExperts,
        kRankLocalSlices[local_expert].checkpoint_expert, kHidden, kHidden,
        swiglu_tensor->values);
    const auto quantized_down = scalar_q8_matvec_with_q8_0_activation(
        g_source_path, g_source_spans[2], kSourceExperts,
        kRankLocalSlices[local_expert].checkpoint_expert, kHidden, kHidden,
        swiglu_tensor->values);
    const double gate_error = max_abs_diff(gate_tensor->values, scalar_gate);
    const double up_error = max_abs_diff(up_tensor->values, scalar_up);
    const double down_error = max_abs_diff(down_tensor->values, scalar_down);
    const double quantized_gate_error = max_abs_diff(gate_tensor->values, quantized_gate);
    const double quantized_up_error = max_abs_diff(up_tensor->values, quantized_up);
    const double quantized_down_error = max_abs_diff(down_tensor->values, quantized_down);
    double silu_formula_error = 0.0;
    for (size_t i = 0; i < kHidden; ++i) {
        const float expected = (gate_tensor->values[i] /
            (1.0f + ::expf(-gate_tensor->values[i]))) * up_tensor->values[i];
        silu_formula_error = std::max(silu_formula_error,
            std::fabs(static_cast<double>(swiglu_tensor->values[i]) - expected));
    }
    std::cout << "R687 stage_errors expert=" << capture.registry_expert
              << " gate_vs_scalar_q8=" << gate_error
              << " up_vs_scalar_q8=" << up_error
              << " swiglu_vs_explicit_expf=" << silu_formula_error
              << " down_vs_unquantized_scalar=" << down_error
              << " gate_vs_scalar_q8_0_dot=" << quantized_gate_error
              << " up_vs_scalar_q8_0_dot=" << quantized_up_error
              << " down_vs_scalar_q8_0_dot=" << quantized_down_error << '\n';

    double best_error = INFINITY;
    const char * best_name = "none";
    double q8_input_control_error = INFINITY;
    for (const auto & control : kRoundControls) {
        const auto slots = scalar_collector_control(local_expert, capture.route_weight, control);
        double error = 0.0;
        for (size_t i = 0; i < kHidden; ++i) {
            error = std::max(error, std::fabs(slots[i] - capture.fp16_collector_output[i]));
        }
        if (error < best_error) {
            best_error = error;
            best_name = control.name;
        }
        if (control.q8_0_linear_inputs) q8_input_control_error = error;
        std::cout << "R687 scalar_control expert=" << capture.registry_expert
                  << " name=" << control.name << " collector_max_abs=" << error << '\n';
    }
    std::cout << "R687 best_scalar_control expert=" << capture.registry_expert
              << " name=" << best_name << " collector_max_abs=" << best_error << '\n';
    if (q8_input_control_error > 1.0e-6) {
        std::cerr << "R687 Q8_0 activation-quantized oracle diverged for expert "
                  << capture.registry_expert << ": " << q8_input_control_error << '\n';
        return false;
    }
    return true;
}
} // namespace

int main() {
    const char *output_directory = std::getenv("VF_TEST_OUTPUT_DIR");
    if (output_directory == nullptr || !load_r659_rank_local_fixture(output_directory)) return 1;
    const char *rank_session_binary = std::getenv("VF_RANK_SESSION_BIN");
    if (rank_session_binary == nullptr || *rank_session_binary == '\0') {
        std::cerr << "R688 requires VF_RANK_SESSION_BIN\n";
        return 1;
    }
    const auto ledger_root = (std::filesystem::path(output_directory) / "rust-ledger").string();
    std::unique_ptr<void, void (*)(void *)> rust_context(
        vf_r688_open(g_source_path.c_str(), rank_session_binary, ledger_root.c_str()),
        vf_r688_close);
    if (!rust_context) {
        std::cerr << "R688 could not create VoltaFabric resident rank sessions\n";
        return 1;
    }
    // The rank-local GGUF uses compact checkpoint IDs 0 and 1 for resident
    // order [expert 40, expert 3]. The route boundary maps each ID once.
    if (!run_graph_case(1, 0, 40, 0.25f, false) ||
        !run_graph_case(2, 1, 3, 0.75f, false) ||
        !run_graph_case(3, 1, 3, 0.75f, true) ||
        !run_graph_case(4, 1, 3, 0.75f, false, true)) {
        std::cerr << "R659 Q8_0 GGUF donor MoE bridge failed\n";
        return 1;
    }
    std::vector<double> combined;
    RouteCapture first_result;
    RouteCapture second_result;
    if (!run_combined_layer(false, combined, &first_result, &second_result, true) || combined.size() != kHidden ||
        !reject_partial_combined_layer()) {
        std::cerr << "R687 combined layer, stage capture, or atomic refusal failed\n";
        return 1;
    }
    if (!report_route_rounding_controls(first_result, 0) ||
        !report_route_rounding_controls(second_result, 1)) return 1;
    std::vector<double> volta_fabric_output(kHidden,
        std::numeric_limits<double>::quiet_NaN());
    if (vf_r688_run_layer(rust_context.get(), volta_fabric_output.data(),
                          volta_fabric_output.size()) != 0 ||
        std::any_of(volta_fabric_output.begin(), volta_fabric_output.end(),
                    [](double value) { return !std::isfinite(value); })) {
        std::cerr << "R688 VoltaFabric resident layer failed or returned non-finite output\n";
        return 1;
    }
    std::vector<double> refused_output(kHidden, 42.0);
    if (vf_r688_run_layer(rust_context.get(), refused_output.data(), kHidden - 1) == 0 ||
        std::any_of(refused_output.begin(), refused_output.end(),
                    [](double value) { return value != 42.0; })) {
        std::cerr << "R688 invalid output shape was not refused before publication\n";
        return 1;
    }
    double donor_vs_voltafabric_max_abs = 0.0;
    size_t donor_vs_voltafabric_max_index = 0;
    for (size_t i = 0; i < kHidden; ++i) {
        const double error = std::fabs(combined[i] - volta_fabric_output[i]);
        if (error > donor_vs_voltafabric_max_abs) {
            donor_vs_voltafabric_max_abs = error;
            donor_vs_voltafabric_max_index = i;
        }
    }
    if (donor_vs_voltafabric_max_abs > 1.0e-6) {
        std::cerr << "R688 donor/VoltaFabric layer mismatch exceeds the independent oracle tolerance: "
                  << donor_vs_voltafabric_max_abs << '\n';
        return 1;
    }
    std::vector<double> scalar_combined(kHidden, 0.0);
    for (size_t local_expert = 0; local_expert < kExperts; ++local_expert) {
        const auto oracle = scalar_expert_ffn(local_expert,
            local_expert == 0 ? 0.25f : 0.75f, g_source_spans);
        for (size_t i = 0; i < kHidden; ++i) scalar_combined[i] += oracle.fp16_slot[i];
    }
    double maximum_layer_error = 0.0;
    size_t maximum_layer_error_index = 0;
    for (size_t i = 0; i < kHidden; ++i) {
        const double error = std::fabs(combined[i] - scalar_combined[i]);
        if (error > maximum_layer_error) {
            maximum_layer_error = error;
            maximum_layer_error_index = i;
        }
    }
    std::vector<double> quantized_scalar_combined(kHidden, 0.0);
    for (size_t local_expert = 0; local_expert < kExperts; ++local_expert) {
        const auto slots = scalar_collector_control(local_expert,
            local_expert == 0 ? 0.25f : 0.75f, kRoundControls.back());
        for (size_t i = 0; i < kHidden; ++i) quantized_scalar_combined[i] += slots[i];
    }
    double quantized_layer_error = 0.0;
    for (size_t i = 0; i < kHidden; ++i) {
        quantized_layer_error = std::max(quantized_layer_error,
            std::fabs(combined[i] - quantized_scalar_combined[i]));
    }
    if (first_result.scalar_collector_match || !second_result.scalar_collector_match ||
        maximum_layer_error <= 1.0e-6 || quantized_layer_error > 1.0e-6) {
        std::cerr << "R687 expected the unquantized oracle to diverge and Q8_0 activation oracle to close; "
                  << "first_match=" << first_result.scalar_collector_match
                  << " first_error=" << first_result.scalar_collector_max_abs_error
                  << " second_match=" << second_result.scalar_collector_match
                  << " second_error=" << second_result.scalar_collector_max_abs_error
                  << " combined_error=" << maximum_layer_error
                  << " q8_activation_oracle_error=" << quantized_layer_error << '\n';
        return 1;
    }
    std::vector<float> unhooked_top2;
    std::vector<float> voltafabric_graph_output;
    RouteCapture live_route;
    const uint64_t live_batch = 5;
    const uint64_t live_attempt = fresh_session_attempt();
    if (!run_unhooked_top2_graph(unhooked_top2) ||
        !run_session_hook_top2_graph(rust_context.get(), live_batch, live_attempt, false,
                                      voltafabric_graph_output, &live_route) ||
        live_route.session_fp32_output.size() != kHidden) {
        std::cerr << "R689 live graph did not accept the VoltaFabric Session output\n";
        return 1;
    }
    double live_graph_error = 0.0;
    double independent_oracle_error = 0.0;
    double donor_oracle_error = 0.0;
    const auto scalar_fp32_oracle = scalar_donor_fp32_top2_oracle(live_route);
    std::vector<float> captured_slot_sum(kHidden, 0.0f);
    for (size_t slot = 0; slot < live_route.weights.size(); ++slot) {
        for (size_t i = 0; i < kHidden; ++i) {
            const float weighted = live_route.raw_slots[slot * kHidden + i] * live_route.weights[slot];
            captured_slot_sum[i] = captured_slot_sum[i] + weighted;
        }
    }
    for (size_t i = 0; i < kHidden; ++i) {
        live_graph_error = std::max(live_graph_error,
            std::fabs(static_cast<double>(unhooked_top2[i]) - voltafabric_graph_output[i]));
        independent_oracle_error = std::max(independent_oracle_error,
            std::fabs(static_cast<double>(unhooked_top2[i]) - scalar_fp32_oracle[i]));
        donor_oracle_error = std::max(donor_oracle_error,
            std::fabs(static_cast<double>(unhooked_top2[i]) - captured_slot_sum[i]));
    }
    if (live_graph_error > 1.0e-6 || independent_oracle_error > 1.0e-6 ||
        donor_oracle_error > 1.0e-6) {
        std::cerr << "R689 donor top-2 graph, live VoltaFabric FP32 output, or independent scalar oracle differs: "
                  << "donor_vs_vf=" << live_graph_error
                  << " donor_vs_scalar=" << independent_oracle_error
                  << " donor_vs_captured_slots=" << donor_oracle_error << '\n';
        return 1;
    }
    std::vector<float> refused_live_output;
    if (!run_session_hook_top2_graph(rust_context.get(), 6, fresh_session_attempt(), true,
                                     refused_live_output) ||
        !refused_live_output.empty()) {
        std::cerr << "R689 live graph exposed output after collector refusal\n";
        return 1;
    }
    std::vector<float> duplicate_identity_output;
    if (!run_session_hook_top2_graph(rust_context.get(), live_batch, live_attempt, false,
                                     duplicate_identity_output, nullptr, true) ||
        !duplicate_identity_output.empty()) {
        std::cerr << "R689 reused Session identity was not rejected before graph output\n";
        return 1;
    }
    std::cout << "R689 live llama.cpp Qwen2MoE graph output came from resident VoltaFabric Sessions; "
              << "unhooked donor-vs-hooked FP32 max_abs=" << live_graph_error
              << ", donor-vs-independent Q8_0/FP32 slot-order oracle max_abs="
              << independent_oracle_error
              << ", donor-vs-captured donor slots max_abs=" << donor_oracle_error
              << "; duplicate completed identity and injected provider refusal both withheld output\n";
    std::cout << "R687 rank-local Q8_0: streamed " << g_rank_local_source_bytes << "/"
              << g_source_tensor_bytes << " source tensor bytes; mapped donor routes verified; "
              << "unquantized scalar FP16/f64 oracle divergence=" << maximum_layer_error
              << " at output " << maximum_layer_error_index
              << " (donor=" << combined[maximum_layer_error_index]
              << ", scalar=" << scalar_combined[maximum_layer_error_index]
              << "); independent scalar Q8_0 activation-quantized oracle max_error="
              << quantized_layer_error
              << "; expert0_match=" << first_result.scalar_collector_match
              << ", expert0_error=" << first_result.scalar_collector_max_abs_error
              << ", expert1_match=" << second_result.scalar_collector_match
              << ", expert1_error=" << second_result.scalar_collector_max_abs_error
              << "; partial provider failure withheld; no full-equivalence claim\n";
    std::cout << "R688 CPU host-process integration (pinned llama.cpp graph + Rust coordinator; "
              << "rank Session workers are child processes): resident Q8_0 expert slices "
              << "[1,0] -> [40,3], route weights applied once, "
              << "donor_vs_voltafabric_max_abs=" << donor_vs_voltafabric_max_abs
              << " at output " << donor_vs_voltafabric_max_index
              << " (donor=" << combined[donor_vs_voltafabric_max_index]
              << ", VoltaFabric=" << volta_fabric_output[donor_vs_voltafabric_max_index]
              << "); invalid output-shape request refused without modifying sentinel; "
              << "no complete-generation claim\n";
    return 0;
}
