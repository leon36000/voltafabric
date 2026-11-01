#include "route_graph_transaction.hpp"
#include "session_vfr3_collector.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

namespace {

using vf::ggml::BatchIdentity;
using vf::ggml::RouteBatch;
using vf::ggml::RouteGraphTransaction;
using vf::ggml::RouteStatus;
using vf::ggml::Vfr3CollectorContext;
using vf::strata_bridge::Client;
using vf::strata_bridge::SessionIdentity;

constexpr uint32_t kLayer = 2;
constexpr uint64_t kBatch = 7;
constexpr uint64_t kAttempt = 1;
constexpr uint64_t kEpoch = 41;
constexpr size_t kHidden = 2560;
const std::string kRun = "run-r671";
const std::string kModel = "qwen-moe-fixture";
const std::string kPlan = "plan-r671";

void put16(std::vector<uint8_t> &out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
}
void put32(std::vector<uint8_t> &out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}
void put64(std::vector<uint8_t> &out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}
uint16_t get16(const uint8_t *p) {
    return static_cast<uint16_t>(p[0]) | static_cast<uint16_t>(p[1] << 8);
}
uint32_t get32(const uint8_t *p) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= uint32_t(p[i]) << (8 * i);
    return value;
}
uint64_t get64(const uint8_t *p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= uint64_t(p[i]) << (8 * i);
    return value;
}

bool read_exact(int fd, uint8_t *data, size_t size) {
    while (size != 0) {
        const ssize_t got = read(fd, data, size);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        data += got;
        size -= static_cast<size_t>(got);
    }
    return true;
}
bool send_exact(int fd, const uint8_t *data, size_t size) {
    while (size != 0) {
        const ssize_t sent = send(fd, data, size, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return false;
        data += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}
void append_identity(std::vector<uint8_t> &out) {
    out.insert(out.end(), kRun.begin(), kRun.end());
    out.insert(out.end(), kModel.begin(), kModel.end());
    out.insert(out.end(), kPlan.begin(), kPlan.end());
}
bool identity_bytes_match(const std::vector<uint8_t> &bytes) {
    std::vector<uint8_t> expected;
    append_identity(expected);
    return bytes == expected;
}
std::vector<uint8_t> response_header(uint32_t status, uint64_t batch, uint32_t rows,
                                     uint64_t attempt) {
    std::vector<uint8_t> out;
    out.insert(out.end(), {'V','F','R','3'});
    put32(out, status);
    put64(out, batch);
    put32(out, rows);
    put32(out, static_cast<uint32_t>(kHidden));
    put32(out, kLayer);
    put64(out, kEpoch);
    put16(out, static_cast<uint16_t>(kRun.size()));
    put16(out, static_cast<uint16_t>(kModel.size()));
    put16(out, static_cast<uint16_t>(kPlan.size()));
    put16(out, 0);
    put64(out, attempt);
    append_identity(out);
    return out;
}

bool serve_fake_vfr3(int fd, bool omit_row) {
    uint8_t header[52]{};
    if (!read_exact(fd, header, sizeof(header)) ||
        std::memcmp(header, "VFB3", 4) != 0 ||
        get32(header + 4) != 1 || get64(header + 8) != kBatch ||
        get32(header + 16) != 1 || get32(header + 20) != 2 ||
        get32(header + 24) != kLayer || get64(header + 28) != kEpoch ||
        get16(header + 42) != 0 || get64(header + 44) != kAttempt ||
        get16(header + 36) != kRun.size() || get16(header + 38) != kModel.size() ||
        get16(header + 40) != kPlan.size()) {
        return false;
    }
    const size_t identity_size = kRun.size() + kModel.size() + kPlan.size();
    std::vector<uint8_t> identity(identity_size);
    if (!read_exact(fd, identity.data(), identity.size()) || !identity_bytes_match(identity)) return false;
    const size_t body_size = kHidden * 4 + 2 * 4 + 2 * 4;
    std::vector<uint8_t> body(body_size);
    if (!read_exact(fd, body.data(), body.size()) ||
        get32(body.data()) != 0x3f000000U ||
        get32(body.data() + kHidden * 4) != 2 ||
        get32(body.data() + kHidden * 4 + 4) != 7 ||
        get32(body.data() + kHidden * 4 + 8) != std::numeric_limits<uint32_t>::max() ||
        get32(body.data() + kHidden * 4 + 12) != 0) {
        return false;
    }

    auto reply = response_header(0, kBatch, omit_row ? 0U : 1U, kAttempt);
    if (!send_exact(fd, reply.data(), reply.size())) return false;
    if (omit_row) return true;
    std::vector<uint8_t> row;
    put32(row, 0);
    for (size_t i = 0; i < kHidden; ++i) put32(row, 0x3f8013a9U); // raw remote value = 1.0006f, rounds to FP16 1.0009765625
    if (!send_exact(fd, row.data(), row.size())) return false;

    uint8_t shutdown[52]{};
    if (!read_exact(fd, shutdown, sizeof(shutdown)) ||
        std::memcmp(shutdown, "VFB3", 4) != 0 ||
        get32(shutdown + 4) != 2 || get64(shutdown + 8) != 0 ||
        get32(shutdown + 16) != 0 || get32(shutdown + 20) != 0 ||
        get32(shutdown + 24) != kLayer || get64(shutdown + 28) != kEpoch ||
        get64(shutdown + 44) != 0) {
        return false;
    }
    const size_t shutdown_identity_size = kRun.size() + kModel.size() + kPlan.size();
    std::vector<uint8_t> shutdown_identity(shutdown_identity_size);
    if (!read_exact(fd, shutdown_identity.data(), shutdown_identity.size()) ||
        !identity_bytes_match(shutdown_identity)) return false;
    auto closed = response_header(2, 0, 0, 0);
    return send_exact(fd, closed.data(), closed.size());
}

RouteBatch make_batch() {
    RouteBatch batch;
    batch.identity = BatchIdentity{kRun, kModel, kPlan, kLayer, kEpoch, kBatch, kAttempt};
    batch.hidden = kHidden;
    batch.slots = 2;
    batch.tokens = 1;
    batch.experts = 8;
    batch.expert_ids = {2, 7};
    batch.owner_kinds = {-1, 0};
    batch.weights = {0.25f, 0.75f};
    return batch;
}

void copy_downstream(ggml_tensor *dst, const ggml_tensor *src, int ith, int nth, void *userdata) {
    auto &calls = *static_cast<int *>(userdata);
    if (ith != 0 || nth != 1 || dst->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 ||
        dst->data == nullptr || src->data == nullptr || ggml_nelements(dst) != ggml_nelements(src)) return;
    std::memcpy(dst->data, src->data, static_cast<size_t>(ggml_nelements(src)) * sizeof(float));
    ++calls;
}

struct Graph {
    ggml_context *ctx = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_cplan plan{};
    std::vector<uint8_t> work;
    ggml_tensor *slot_outputs = nullptr;
    ggml_tensor *activations = nullptr;
    ggml_tensor *routed = nullptr;
    ggml_tensor *downstream = nullptr;
    ~Graph() {
        if (ctx != nullptr) ggml_free(ctx);
    }
};

bool make_graph(RouteGraphTransaction &transaction, int &downstream_calls, Graph &g) {
    ggml_init_params params{};
    params.mem_size = 1U << 20;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    g.ctx = ggml_init(params);
    if (g.ctx == nullptr) return false;
    g.slot_outputs = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, kHidden, 2, 1);
    g.activations = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kHidden, 1);
    if (g.slot_outputs == nullptr || g.activations == nullptr ||
        !ggml_is_contiguous(g.slot_outputs) || !ggml_is_contiguous(g.activations) ||
        g.slot_outputs->data == nullptr || g.activations->data == nullptr) return false;
    auto *slots = static_cast<float *>(g.slot_outputs->data);
    for (size_t i = 0; i < kHidden; ++i) {
        slots[i] = 0.0f;          // remote slot placeholder, replaced only by VFR3
        slots[kHidden + i] = 2.0006f; // local owner output rounds to FP16 2.0
    }
    auto *x = static_cast<float *>(g.activations->data);
    std::fill(x, x + kHidden, 0.5f);

    g.routed = transaction.append(g.ctx, g.slot_outputs, g.activations);
    if (g.routed == nullptr) return false;
    g.downstream = ggml_map_custom1(g.ctx, g.routed, copy_downstream, 1, &downstream_calls);
    g.graph = ggml_new_graph(g.ctx);
    if (g.downstream == nullptr || g.graph == nullptr) return false;
    ggml_build_forward_expand(g.graph, g.downstream);
    g.plan = ggml_graph_plan(g.graph, 1, nullptr);
    g.work.resize(g.plan.work_size);
    g.plan.work_data = g.work.empty() ? nullptr : g.work.data();
    transaction.attach_abort(g.plan);
    return true;
}

int run_case(bool omit_row) {
    int fds[2]{};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
        std::cerr << "socketpair failed\n";
        return 1;
    }
    bool peer_ok = false;
    std::thread peer([&] {
        peer_ok = serve_fake_vfr3(fds[1], omit_row);
        close(fds[1]);
    });
    SessionIdentity identity{kRun, kModel, kPlan, kEpoch};
    Client client(fds[0], fds[0], 2000, kLayer, identity);
    Vfr3CollectorContext collector_context{&client, {}, 0};
    RouteBatch batch = make_batch();
    RouteGraphTransaction transaction(batch, vf::ggml::collect_vfr3, &collector_context);
    Graph graph;
    int downstream_calls = 0;
    if (!client.usable() || !make_graph(transaction, downstream_calls, graph)) {
        std::cerr << "client or graph initialization failed\n";
        close(fds[0]);
        peer.join();
        return 1;
    }
    const ggml_status status = ggml_graph_compute(graph.graph, &graph.plan);
    const bool accepted = transaction.accepts(status);
    if (omit_row) {
        const bool ok = !accepted && status == GGML_STATUS_ABORTED &&
            transaction.status() == RouteStatus::provider_failure &&
            downstream_calls == 0 && collector_context.calls == 1 &&
            !collector_context.error.empty();
        close(fds[0]);
        peer.join();
        if (!peer_ok || !ok) {
            std::cerr << "missing VFR3 row was not rejected atomically: " << collector_context.error << "\n";
            return 1;
        }
        return 0;
    }

    // Scalar oracle: FP16(1.0006)=1.0009765625 and FP16(2.0006)=2;
    // .25*1.0009765625 + .75*2 = 1.750244140625.
    bool values_match = accepted && downstream_calls == 1 && collector_context.calls == 1;
    if (values_match) {
        const auto *values = static_cast<const float *>(graph.downstream->data);
        for (size_t i = 0; i < kHidden; ++i) {
            if (std::fabs(values[i] - 1.750244140625f) > 1e-7f) {
                values_match = false;
                break;
            }
        }
    }
    const ggml_status replay = ggml_graph_compute(graph.graph, &graph.plan);
    std::string shutdown_error;
    const bool closed = client.shutdown(shutdown_error);
    close(fds[0]);
    peer.join();
    if (!peer_ok || !values_match || replay != GGML_STATUS_ABORTED ||
        transaction.status() != RouteStatus::invalid_batch || collector_context.calls != 1 || !closed) {
        std::cerr << "VFR3 graph transaction failed: status=" << ggml_status_to_string(status)
                  << " replay=" << ggml_status_to_string(replay)
                  << " error=" << collector_context.error
                  << " shutdown=" << shutdown_error << "\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (run_case(false) != 0) return 1;
    if (run_case(true) != 0) return 1;
    std::cout << "GGML VFR3 route transaction: identity, FP16/f64 weighting, failure and single-use passed\n";
    return 0;
}
