#include "llama-context.h"
#include "gguf.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "qwen2moe_llama_route_bridge.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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

struct RouteCapture {
    uint64_t batch_id = 0;
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
    bool scalar_collector_match = true;
    double scalar_collector_max_abs_error = 0.0;
};

class R659MoEInputs final : public llm_graph_input_i {
public:
    R659MoEInputs(ggml_tensor *activations, ggml_tensor *logits,
                  ggml_tensor *gate, ggml_tensor *up, ggml_tensor *down,
                  int32_t checkpoint_expert)
        : activations_(activations), logits_(logits), gate_(gate),
          up_(up), down_(down), checkpoint_expert_(checkpoint_expert) {}

    void set_input(const llama_ubatch *ubatch) override {
        if (ubatch == nullptr || ubatch->n_tokens != 1) return;
        std::array<float, kHidden> activations{};
        activations[0] = 1.0f;
        const std::array<float, 2> logits = checkpoint_expert_ == 0
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
};

class R659Q8MoEGraph final : public llm_graph_context {
public:
    R659Q8MoEGraph(const llm_graph_params &params, int32_t checkpoint_expert)
        : llm_graph_context(params) {
        auto *activations = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kHidden, 1);
        auto *logits = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 1);
        auto *gate = ggml_new_tensor_3d(ctx0, GGML_TYPE_Q8_0, 32, 32, 2);
        auto *up = ggml_new_tensor_3d(ctx0, GGML_TYPE_Q8_0, 32, 32, 2);
        auto *down = ggml_new_tensor_3d(ctx0, GGML_TYPE_Q8_0, 32, 32, 2);
        if (!activations || !logits || !gate || !up || !down) {
            throw std::runtime_error("could not allocate Q8_0 donor graph tensors");
        }
        for (auto *input : {activations, logits, gate, up, down}) ggml_set_input(input);
        res->add_input(std::make_unique<R659MoEInputs>(
            activations, logits, gate, up, down, checkpoint_expert));
        ggml_tensor *routed = build_moe_ffn(
            activations, nullptr, up, gate, down, nullptr,
            2, 1, LLM_FFN_SILU, false, 1.0f,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, 0,
            logits, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (routed == nullptr) throw std::runtime_error("donor build_moe_ffn returned null");
        res->t_embd = routed;
        ggml_build_forward_expand(gf, routed);
    }
};

class R659Qwen2MoEModel final : public llama_model {
public:
    explicit R659Qwen2MoEModel(int32_t checkpoint_expert)
        : llama_model(llama_model_default_params()), checkpoint_expert_(checkpoint_expert) {
        arch = LLM_ARCH_QWEN2MOE;
        hparams.vocab_only = false;
        hparams.n_ctx_train = 256;
        hparams.n_embd = kHidden;
        hparams.n_embd_out_impl = kHidden;
        hparams.n_layer_all = 1;
        hparams.n_layer_nextn = 0;
        hparams.n_expert = 2;
        hparams.n_expert_used_arr[0] = 1;
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
        return std::make_unique<R659Q8MoEGraph>(params, checkpoint_expert_);
    }

private:
    int32_t checkpoint_expert_;
};

bool make_batch(void *userdata, const llama_ubatch &ubatch, llm_graph_type graph_type,
                uint32_t expected_layers, RouteBatch &batch) {
    auto &capture = *static_cast<RouteCapture *>(userdata);
    if (ubatch.n_tokens != 1 || graph_type != LLM_GRAPH_TYPE_DEFAULT || expected_layers != 1) {
        return false;
    }
    batch.identity = BatchIdentity{"r686-r659-rank-local", "r659-qwen2moe-rank-local-q8-fixture",
                                   "resident-index-to-registry-cpu", 0, 686,
                                   capture.batch_id, capture.batch_id};
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
                    RouteCapture * completed = nullptr, bool * graph_completed = nullptr) {
    if (graph_completed != nullptr) *graph_completed = false;
    R659Qwen2MoEModel model(checkpoint_expert);
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

    RouteCapture capture;
    capture.batch_id = batch_id;
    capture.checkpoint_expert = checkpoint_expert;
    capture.registry_expert = registry_expert;
    capture.route_weight = route_weight;
    capture.reject = reject;
    capture.duplicate_projection = duplicate_projection;
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
        capture.identity.run != "r686-r659-rank-local" || capture.identity.model != "r659-qwen2moe-rank-local-q8-fixture" ||
        capture.identity.plan != "resident-index-to-registry-cpu" || capture.identity.layer != 0 ||
        capture.identity.epoch != 686 || capture.identity.batch != batch_id ||
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

bool run_combined_layer(bool reject_second, std::vector<double> & published,
                        RouteCapture * first_result = nullptr,
                        RouteCapture * second_result = nullptr) {
    std::vector<double> staged(kHidden, 0.0);
    constexpr std::array<float, 2> weights{0.25f, 0.75f};
    for (size_t local_expert = 0; local_expert < kExperts; ++local_expert) {
        RouteCapture capture;
        bool graph_completed = false;
        const int32_t registry_expert = kRankLocalSlices[local_expert].resident_expert;
        if (!run_graph_case(local_expert + 1, static_cast<int32_t>(local_expert),
                            registry_expert, weights[local_expert],
                            reject_second && local_expert == 1, false, &capture,
                            &graph_completed)) {
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
} // namespace

int main() {
    const char *output_directory = std::getenv("VF_TEST_OUTPUT_DIR");
    if (output_directory == nullptr || !load_r659_rank_local_fixture(output_directory)) return 1;
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
    if (!run_combined_layer(false, combined, &first_result, &second_result) || combined.size() != kHidden ||
        !reject_partial_combined_layer()) {
        std::cerr << "R686 combined layer or atomic refusal failed\n";
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
    if (first_result.scalar_collector_match || !second_result.scalar_collector_match ||
        maximum_layer_error <= 1.0e-6) {
        std::cerr << "R686 expected source-checkpoint expert 1 to diverge at FP16 and expert 0 to match; "
                  << "first_match=" << first_result.scalar_collector_match
                  << " first_error=" << first_result.scalar_collector_max_abs_error
                  << " second_match=" << second_result.scalar_collector_match
                  << " second_error=" << second_result.scalar_collector_max_abs_error
                  << " combined_error=" << maximum_layer_error << '\n';
        return 1;
    }
    std::cout << "R686 rank-local Q8_0: streamed " << g_rank_local_source_bytes << "/"
              << g_source_tensor_bytes << " source tensor bytes; mapped donor routes verified; "
              << "scalar FP16/f64 oracle divergence=" << maximum_layer_error
              << " at output " << maximum_layer_error_index
              << " (donor=" << combined[maximum_layer_error_index]
              << ", scalar=" << scalar_combined[maximum_layer_error_index]
              << "); expert0_match=" << first_result.scalar_collector_match
              << ", expert0_error=" << first_result.scalar_collector_max_abs_error
              << ", expert1_match=" << second_result.scalar_collector_match
              << ", expert1_error=" << second_result.scalar_collector_max_abs_error
              << "; partial provider failure withheld; no full-equivalence claim\n";
    return 0;
}
