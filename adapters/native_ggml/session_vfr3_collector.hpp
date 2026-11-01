#pragma once

#include "pipe_client.hpp"
#include "route_graph_transaction.hpp"

#include <cstddef>
#include <string>

namespace vf::ggml {

struct Vfr3CollectorContext {
    vf::strata_bridge::Client *client = nullptr;
    std::string error;
    size_t calls = 0;
};

// Merge raw VFB3 remote-owner results into local raw slot outputs, then apply
// each original route coefficient once and accumulate in token/slot order.
RouteStatus collect_vfr3(const RouteBatch &batch, const float *activations,
                         size_t activation_count, const float *slot_outputs,
                         size_t slot_output_count, float *output,
                         size_t output_count, void *userdata);

} // namespace vf::ggml
