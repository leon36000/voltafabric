// Sampled scalar reference for the QKV projection in a GDN stage result.
// GGML conversion and activation quantization are shared with the provider;
// the row products below are accumulated independently in double precision.
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 6) return 1;
    const int kind = std::atoi(argv[3]);
    if (kind != 12 && kind != 13 && kind != 23) return 2;
    const int row_count = std::atoi(argv[5]);
    if (row_count != 6144 && row_count != 10240) return 2;
    char *end = nullptr;
    const auto offset = std::strtoll(argv[4], &end, 10);
    if (!end || *end || offset < 0) return 2;
    std::ifstream shard(argv[1], std::ios::binary | std::ios::ate);
    std::ifstream result(argv[2], std::ios::binary | std::ios::ate);
    if (!shard || shard.tellg() != std::streamoff(55063448064LL) || !result ||
        result.tellg() != std::streamoff(19044 * 4)) return 3;
    std::vector<float> mixed(2560), actual(10240), activation(2560), row(2560);
    result.seekg(0);
    if (!result.read(reinterpret_cast<char *>(mixed.data()), 2560 * 4)) return 4;
    result.seekg(std::streamoff(2564 * 4));
    if (!result.read(reinterpret_cast<char *>(actual.data()), 10240 * 4)) return 4;
    for (float v : actual) if (!std::isfinite(v)) return 5;
    ggml_cpu_init();
    const auto type = static_cast<ggml_type>(kind);
    const auto *weights = ggml_get_type_traits(type);
    const auto *quantizer = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (!weights || !weights->to_float || !quantizer || !quantizer->from_float) return 6;
    const size_t row_bytes = ggml_row_size(type, 2560);
    std::vector<uint8_t> packed(row_bytes), quantized(ggml_row_size(GGML_TYPE_Q8_K, 2560));
    quantizer->from_float(mixed.data(), quantized.data(), 2560);
    for (size_t block = 0; block < 10; ++block) {
        float scale;
        std::memcpy(&scale, quantized.data() + block * 292, 4);
        for (size_t j = 0; j < 256; ++j) {
            int8_t v;
            std::memcpy(&v, quantized.data() + block * 292 + 4 + j, 1);
            activation[block * 256 + j] = scale * v;
        }
    }
    const int rows[] = {0, 1, 127, 2047, 2048, 4095, 4096, 8191, 10239};
    double max_error = 0;
    for (int i : rows) {
        if (i >= row_count) continue;
        shard.seekg(offset + std::streamoff(i) * std::streamoff(row_bytes));
        if (!shard.read(reinterpret_cast<char *>(packed.data()), packed.size())) return 7;
        weights->to_float(packed.data(), row.data(), 2560);
        double expected = 0;
        for (size_t j = 0; j < row.size(); ++j) expected += double(row[j]) * activation[j];
        const double error = std::abs(double(actual[i]) - expected);
        max_error = std::max(max_error, error);
        if (!std::isfinite(expected) || error > 1e-5 + 1e-3 * std::abs(expected)) return 8;
    }
    std::printf("type=%d sampled_rows=%d checked_values=10240 max_abs=%.9g\n", kind, row_count == 6144 ? 7 : 9, max_error);
}
