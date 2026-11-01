#include "pipe_client.hpp"
#include "pool_hook.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>

namespace {
using namespace vf::strata_bridge;

uint16_t get16(const uint8_t* p) {
    return uint16_t(p[0]) | uint16_t(p[1]) << 8;
}
uint32_t get32(const uint8_t* p) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= uint32_t(p[i]) << (8 * i);
    return value;
}
uint64_t get64(const uint8_t* p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= uint64_t(p[i]) << (8 * i);
    return value;
}
void put16(std::vector<uint8_t>& out, uint16_t value) {
    for (int i = 0; i < 2; ++i) out.push_back(uint8_t(value >> (8 * i)));
}
void put32(std::vector<uint8_t>& out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.push_back(uint8_t(value >> (8 * i)));
}
void put64(std::vector<uint8_t>& out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out.push_back(uint8_t(value >> (8 * i)));
}
bool read_exact(int fd, uint8_t* data, size_t size) {
    while (size) {
        const ssize_t n = recv(fd, data, size, 0);
        if (n <= 0) return false;
        data += n;
        size -= size_t(n);
    }
    return true;
}
bool write_exact(int fd, const uint8_t* data, size_t size) {
    while (size) {
        const ssize_t n = send(fd, data, size, MSG_NOSIGNAL);
        if (n <= 0) return false;
        data += n;
        size -= size_t(n);
    }
    return true;
}
std::vector<uint8_t> identity_bytes(const SessionIdentity& identity) {
    std::vector<uint8_t> out;
    out.insert(out.end(), identity.run_id.begin(), identity.run_id.end());
    out.insert(out.end(), identity.model_revision.begin(), identity.model_revision.end());
    out.insert(out.end(), identity.plan_id.begin(), identity.plan_id.end());
    return out;
}
std::vector<uint8_t> reply(const SessionIdentity& identity, uint32_t layer, uint32_t status,
                           uint64_t batch, uint64_t attempt, bool include_row) {
    const auto ids = identity_bytes(identity);
    std::vector<uint8_t> out;
    out.insert(out.end(), {'V','F','R','3'});
    put32(out, status);
    put64(out, batch);
    put32(out, include_row ? 1u : 0u);
    put32(out, 2560);
    put32(out, layer);
    put64(out, identity.epoch);
    put16(out, uint16_t(identity.run_id.size()));
    put16(out, uint16_t(identity.model_revision.size()));
    put16(out, uint16_t(identity.plan_id.size()));
    put16(out, 0);
    put64(out, attempt);
    out.insert(out.end(), ids.begin(), ids.end());
    if (include_row) {
        put32(out, 0);
        for (size_t i = 0; i < 2560; ++i) put32(out, 0x40600000u); // 3.5f
    }
    return out;
}
bool request_identity_matches(const uint8_t* header, const SessionIdentity& identity, uint32_t layer,
                             uint64_t attempt, const uint8_t* bytes) {
    if (std::memcmp(header, "VFB3", 4) != 0 || get32(header + 24) != layer ||
        get64(header + 28) != identity.epoch || get16(header + 36) != identity.run_id.size() ||
        get16(header + 38) != identity.model_revision.size() || get16(header + 40) != identity.plan_id.size() ||
        get16(header + 42) != 0 || get64(header + 44) != attempt) return false;
    const auto expected = identity_bytes(identity);
    return expected.size() == identity.run_id.size() + identity.model_revision.size() + identity.plan_id.size() &&
           std::equal(expected.begin(), expected.end(), bytes);
}
}

int main() {
    SessionIdentity identity{"run-r638", "model-revision-7", "plan-r638-a", 17};
    int fd[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) return 1;
    Client client(fd[0], fd[0], 1000, 4, identity);
    if (!client.usable()) return 2;

    bool server_ok = false;
    std::thread peer([&] {
        uint8_t header[52]{};
        if (!read_exact(fd[1], header, sizeof(header))) return;
        const size_t ids_size = identity.run_id.size() + identity.model_revision.size() + identity.plan_id.size();
        std::vector<uint8_t> ids(ids_size);
        if (!read_exact(fd[1], ids.data(), ids.size()) ||
            !request_identity_matches(header, identity, 4, 19, ids.data()) ||
            get32(header + 4) != 1 || get64(header + 8) != 42 ||
            get32(header + 16) != 1 || get32(header + 20) != 1) return;
        std::array<uint8_t, (2560 + 2) * 4> body{};
        if (!read_exact(fd[1], body.data(), body.size())) return;
        if (get32(body.data() + 2560 * 4) != 5 || get32(body.data() + 2561 * 4) != uint32_t(-1)) return;
        const auto completed = reply(identity, 4, 0, 42, 19, true);
        if (!write_exact(fd[1], completed.data(), completed.size())) return;

        if (!read_exact(fd[1], header, sizeof(header)) || get32(header + 4) != 2 ||
            get64(header + 8) != 0 || !read_exact(fd[1], ids.data(), ids.size()) ||
            !request_identity_matches(header, identity, 4, 0, ids.data())) return;
        const auto closed = reply(identity, 4, 2, 0, 0, false);
        if (!write_exact(fd[1], closed.data(), closed.size())) return;
        server_ok = true;
    });

    std::vector<float> x(2560, 0.25f), destination(2560, 9.0f);
    const std::vector<int32_t> experts{5}, kinds{-1};
    std::string error;
    const bool run_ok = client.run(42, 19, 1, 1, x, experts, kinds, destination, error);
    const bool output_ok = std::all_of(destination.begin(), destination.end(), [](float v) { return v == 3.5f; });
    const bool shutdown_ok = run_ok && client.shutdown(error);
    if (!run_ok) (void)::shutdown(fd[0], SHUT_RDWR);
    peer.join();
    close(fd[1]);
    close(fd[0]);
    if (!run_ok || !output_ok || !shutdown_ok || !server_ok) return 3;

    SessionIdentity expected{"run-r638", "model-revision-7", "plan-r638-a", 17};
    SessionIdentity wrong_reply = expected;
    wrong_reply.plan_id = "plan-r638-b";
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) return 7;
    Client mismatched(fd[0], fd[0], 1000, 2, expected);
    std::thread wrong_peer([&] {
        uint8_t request_header[52]{};
        if (!read_exact(fd[1], request_header, sizeof(request_header))) return;
        std::array<uint8_t, (2560 + 2) * 4> body{};
        const size_t id_size = expected.run_id.size() + expected.model_revision.size() + expected.plan_id.size();
        std::vector<uint8_t> ids(id_size);
        if (!read_exact(fd[1], ids.data(), ids.size()) || !read_exact(fd[1], body.data(), body.size())) return;
        const auto response = reply(wrong_reply, 2, 0, 77, 29, true);
        (void)write_exact(fd[1], response.data(), response.size());
    });
    std::vector<float> unchanged(2560, 11.0f);
    const bool rejected_identity = !mismatched.run(77, 29, 1, 1, x, experts, kinds, unchanged, error);
    const bool output_unchanged = std::all_of(unchanged.begin(), unchanged.end(), [](float v) { return v == 11.0f; });
    (void)::shutdown(fd[0], SHUT_RDWR);
    wrong_peer.join();
    close(fd[1]);
    close(fd[0]);
    if (!rejected_identity || !output_unchanged) return 4;

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) return 8;
    Client wrong_attempt(fd[0], fd[0], 1000, 2, expected);
    std::thread attempt_peer([&] {
        uint8_t request_header[52]{};
        if (!read_exact(fd[1], request_header, sizeof(request_header))) return;
        const size_t id_size = expected.run_id.size() + expected.model_revision.size() + expected.plan_id.size();
        std::vector<uint8_t> ids(id_size);
        if (!read_exact(fd[1], ids.data(), ids.size())) return;
        std::array<uint8_t, (2560 + 2) * 4> body{};
        if (!read_exact(fd[1], body.data(), body.size())) return;
        const auto response = reply(expected, 2, 0, 78, 30, true);
        (void)write_exact(fd[1], response.data(), response.size());
    });
    std::vector<float> attempt_unchanged(2560, 13.0f);
    const bool rejected_attempt = !wrong_attempt.run(78, 29, 1, 1, x, experts, kinds, attempt_unchanged, error);
    const bool attempt_output_unchanged = std::all_of(attempt_unchanged.begin(), attempt_unchanged.end(), [](float v) { return v == 13.0f; });
    (void)::shutdown(fd[0], SHUT_RDWR);
    attempt_peer.join();
    close(fd[1]);
    close(fd[0]);
    if (!rejected_attempt || !attempt_output_unchanged) return 9;

    SessionIdentity fleet_identity{"run-r638", "model-revision-7", "plan-r638-a", 17};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) return 10;
    Client fleet_client(fd[0], fd[0], 1000, 0, fleet_identity);
    PoolContext context{fleet_client, 0, 500, false, {}, {}, 7};
    std::thread fleet_peer([&] {
        for (const auto& [batch, attempt] : {std::pair<uint64_t, uint64_t>{500, 7}, {501, 8}}) {
            uint8_t request_header[52]{};
            if (!read_exact(fd[1], request_header, sizeof(request_header))) return;
            const size_t id_size = fleet_identity.run_id.size() + fleet_identity.model_revision.size() + fleet_identity.plan_id.size();
            std::vector<uint8_t> ids(id_size);
            std::array<uint8_t, (2560 + 2) * 4> body{};
            if (!read_exact(fd[1], ids.data(), ids.size()) || !read_exact(fd[1], body.data(), body.size()) ||
                !request_identity_matches(request_header, fleet_identity, 0, attempt, ids.data()) ||
                get64(request_header + 8) != batch) return;
            const auto response = reply(fleet_identity, 0, 0, batch, attempt, true);
            if (!write_exact(fd[1], response.data(), response.size())) return;
        }
        uint8_t shutdown_header[52]{};
        if (!read_exact(fd[1], shutdown_header, sizeof(shutdown_header))) return;
        const size_t id_size = fleet_identity.run_id.size() + fleet_identity.model_revision.size() + fleet_identity.plan_id.size();
        std::vector<uint8_t> ids(id_size);
        if (!read_exact(fd[1], ids.data(), ids.size()) ||
            !request_identity_matches(shutdown_header, fleet_identity, 0, 0, ids.data())) return;
        const auto closed = reply(fleet_identity, 0, 2, 0, 0, false);
        (void)write_exact(fd[1], closed.data(), closed.size());
    });
    std::vector<float> fleet_input(2560, 0.25f), fleet_output(2560, 11.0f);
    const std::vector<int32_t> fleet_ids{5};
    const bool first_fleet_ok = pool_cpu_only(&context, fleet_input.data(), fleet_ids.data(), 1, 1,
                                              fleet_output.data(), 0);
    const bool first_fleet_output = std::all_of(fleet_output.begin(), fleet_output.end(), [](float v) { return v == 3.5f; });
    const bool second_fleet_ok = pool_cpu_only(&context, fleet_input.data(), fleet_ids.data(), 1, 1,
                                               fleet_output.data(), 0);
    const bool second_fleet_output = std::all_of(fleet_output.begin(), fleet_output.end(), [](float v) { return v == 3.5f; });
    const bool counters_advanced = context.next_batch == 502 && context.next_attempt == 9;
    const bool fleet_shutdown = fleet_client.shutdown(error);
    if (!first_fleet_ok || !second_fleet_ok || !first_fleet_output || !second_fleet_output || !counters_advanced || !fleet_shutdown)
        (void)::shutdown(fd[0], SHUT_RDWR);
    fleet_peer.join();
    close(fd[1]);
    close(fd[0]);
    return first_fleet_ok && second_fleet_ok && first_fleet_output && second_fleet_output &&
                   counters_advanced && fleet_shutdown
               ? 0
               : 11;
}
