// Executes only the extracted opt-in donor guard; invalid inputs must never
// reach the hook, touch output, or issue a VoltaFabric batch.
#include "strata/core/expert_source.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
int calls = 0;
bool unexpected_hook(void*, const float*, const int32_t*, int64_t, int64_t,
                     float*, int64_t) noexcept {
    ++calls;
    return false;
}
}

int main() {
    std::vector<float> x(2560, 0.001f), out(2 * 2560, 9.0f);
    int32_t ids[2] = {0, 1};
    const int64_t invalid_tokens[] = {0, -1, std::numeric_limits<int64_t>::max()};
    for (int64_t tokens : invalid_tokens) {
        strata::core::ExpertDispatch d;
        d.vf_cpu_pool = &unexpected_hook;
        d.n_expert = 512;
        std::fill(out.begin(), out.end(), 9.0f);
        strata::core::expert_pool_dispatch_multi(d, x.data(), ids, tokens, 2, out.data());
        if (!d.failed || d.layers != 0 || calls != 0) return 1;
        for (float value : out) if (value != 9.0f) return 2;
    }
    return 0;
}
