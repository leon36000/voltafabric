#include "pool_hook.hpp"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr size_t kLayers = 48;
constexpr size_t kHidden = 2560;
constexpr int32_t kExpert = 8;

bool parse_u64(const char* value, uint64_t& out) {
    if (!value || !*value || *value == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || !end || *end != '\0') return false;
    out = static_cast<uint64_t>(parsed);
    return true;
}

bool parse_fd(const char* value, int& out) {
    uint64_t parsed = 0;
    if (!parse_u64(value, parsed) || parsed > static_cast<uint64_t>(std::numeric_limits<int>::max())) return false;
    out = static_cast<int>(parsed);
    return true;
}
}

int main(int argc, char** argv) {
    // run, model, plan, epoch, first batch, first attempt, then 48 inherited fds.
    if (argc != 7 + static_cast<int>(kLayers)) return 1;
    uint64_t epoch = 0, first_batch = 0, first_attempt = 0;
    if (!parse_u64(argv[4], epoch) || !parse_u64(argv[5], first_batch) ||
        !parse_u64(argv[6], first_attempt)) return 2;

    std::array<int, kLayers> fds{};
    for (size_t layer = 0; layer < kLayers; ++layer) {
        if (!parse_fd(argv[7 + layer], fds[layer])) return 3;
    }

    using namespace vf::strata_bridge;
    std::string error;
    SessionIdentity identity{argv[1], argv[2], argv[3], epoch};
    auto owner = PoolFleetOwner::create_from_fds(fds, 30'000, std::move(identity),
                                                  first_batch, error, first_attempt);
    if (!owner || !error.empty()) {
        std::cerr << "PoolFleetOwner admission failed: " << error << '\n';
        return 4;
    }

    std::array<float, kHidden> input{};
    input.fill(0.001f);
    std::array<int32_t, 1> experts{kExpert};
    std::array<std::array<float, kHidden>, kLayers> first_outputs{};
    for (size_t layer = 0; layer < kLayers; ++layer) {
        first_outputs[layer].fill(-91.0f);
        if (!pool_cpu_by_layer(owner->callback_user(), input.data(), experts.data(), 1, 1,
                               first_outputs[layer].data(), static_cast<int64_t>(layer))) {
            std::cerr << "first callback failed at layer " << layer << '\n';
            return 5;
        }
        bool any_nonzero = false;
        for (float value : first_outputs[layer]) {
            if (!std::isfinite(value)) return 6;
            any_nonzero = any_nonzero || value != 0.0f;
        }
        if (layer == 0 && !any_nonzero) {
            std::cerr << "the previously qualified nonzero layer-0 fixture became zero\n";
            return 7;
        }
    }

    for (size_t layer = 0; layer < kLayers; ++layer) {
        std::array<float, kHidden> second{};
        second.fill(-92.0f);
        if (!pool_cpu_by_layer(owner->callback_user(), input.data(), experts.data(), 1, 1,
                               second.data(), static_cast<int64_t>(layer))) {
            std::cerr << "second callback failed at layer " << layer << '\n';
            return 8;
        }
        if (std::memcmp(second.data(), first_outputs[layer].data(), sizeof(second)) != 0) {
            std::cerr << "same-provider output changed across batches at layer " << layer << '\n';
            return 9;
        }
    }

    float sentinel = 73.0f;
    if (pool_cpu_by_layer(owner->callback_user(), input.data(), experts.data(), 1, 1,
                          &sentinel, static_cast<int64_t>(kLayers)) || sentinel != 73.0f) {
        std::cerr << "invalid layer was accepted or modified its destination\n";
        return 10;
    }
    sentinel = 74.0f;
    if (pool_cpu_by_layer(owner->callback_user(), input.data(), experts.data(), 1, 1,
                          &sentinel, 0) || sentinel != 74.0f) {
        std::cerr << "terminal fleet accepted a callback after a routing error\n";
        return 11;
    }
    if (!owner->shutdown_all(error)) {
        std::cerr << "fleet shutdown failed: " << error << '\n';
        return 12;
    }
    std::cout << "R651_CPP_POOL_PASS layers=48 batches=2 callbacks=96 terminal_refusal=PASS shutdown=PASS\n";
    return 0;
}
