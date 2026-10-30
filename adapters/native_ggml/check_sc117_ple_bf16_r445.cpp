// Sample GGML CPU BF16 dot against a scalar BF16/BF16 accumulation.
// Caller supplies the local, independently identified shard and 2560-float input.
#include <ggml.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    FILE* x = std::fopen(argv[2], "rb");
    if (!f || !x) return 3;
    std::vector<float> input(2560), activation(2560), weight(2560);
    std::vector<uint16_t> quant(2560), row(2560);
    if (std::fread(input.data(), sizeof(float), input.size(), x) != input.size()) return 4;
    const auto *trait = ggml_get_type_traits_cpu(GGML_TYPE_BF16);
    const auto *decoder = ggml_get_type_traits(GGML_TYPE_BF16);
    if (!trait || trait->vec_dot_type != GGML_TYPE_BF16 || !trait->from_float || !trait->vec_dot || !decoder || !decoder->to_float) return 5;
    trait->from_float(input.data(), quant.data(), 2560);
    decoder->to_float(quant.data(), activation.data(), 2560);
    double max_abs = 0.0;
    for (int matrix = 0; matrix < 2; ++matrix) {
        const int rows = matrix ? 2560 : 10240;
        const int samples[] = {0, 1, 7, rows/4, rows/2, rows-2, rows-1};
        const int64_t base = matrix ? 2967313280LL : 2914761600LL;
        for (int index : samples) {
            if (fseeko(f, base + int64_t(index)*5120, SEEK_SET) || std::fread(row.data(), 2, 2560, f) != 2560) return 6;
            decoder->to_float(row.data(), weight.data(), 2560);
            double ref = 0.0;
            for (int i=0; i<2560; ++i) ref += double(weight[i])*double(activation[i]);
            float native = 0.0;
            trait->vec_dot(2560, &native, 0, row.data(), 0, quant.data(), 0, 1);
            const double error = std::fabs(double(native)-float(ref));
            if (!std::isfinite(native) || error > 2e-5 + 2e-5 * std::fabs(ref)) return 7;
            max_abs = std::max(max_abs, error);
        }
    }
    std::printf("GGML BF16 sampled rows 14 max_abs %.9g\n", max_abs);
    std::fclose(f); std::fclose(x); return 0;
}
