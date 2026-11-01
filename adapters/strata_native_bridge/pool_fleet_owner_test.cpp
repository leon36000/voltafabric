#include "pool_hook.hpp"

#include <array>
#include <memory>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace {
using namespace vf::strata_bridge;
struct SocketPairs {
    std::array<int, 48> client{};
    std::array<int, 48> peer{};
    SocketPairs() { client.fill(-1); peer.fill(-1); }
    ~SocketPairs() {
        for (int fd : peer) if (fd >= 0) close(fd);
        for (int fd : client) if (fd >= 0) close(fd);
    }
    bool make_pairs() {
        for (size_t layer = 0; layer < client.size(); ++layer) {
            int pair[2] = {-1, -1};
            if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) return false;
            client[layer] = pair[0];
            peer[layer] = pair[1];
        }
        return true;
    }
    bool make_clients(PoolFleetOwner::Clients& clients, bool mismatch_identity = false) {
        if (!make_pairs()) return false;
        for (size_t layer = 0; layer < client.size(); ++layer) {
            SessionIdentity identity{"run-r638", "model-r638", "plan-r638", 91};
            if (mismatch_identity && layer == 7) identity.plan_id = "different-plan";
            clients[layer] = std::make_unique<Client>(client[layer], client[layer], 5,
                                                       static_cast<uint32_t>(layer), identity);
        }
        return true;
    }
    void close_peers() {
        for (int& fd : peer) { if (fd >= 0) close(fd); fd = -1; }
    }
};
}

int main() {
    using namespace vf::strata_bridge;
    std::string error;

    SocketPairs bad_sockets;
    PoolFleetOwner::Clients bad_clients{};
    if (!bad_sockets.make_clients(bad_clients)) return 1;
    std::swap(bad_clients[0], bad_clients[1]);
    auto rejected = PoolFleetOwner::create(std::move(bad_clients), 70, error);
    if (rejected || error.find("wrong layer 0") == std::string::npos) return 2;

    SocketPairs mismatch_sockets;
    PoolFleetOwner::Clients mismatch_clients{};
    if (!mismatch_sockets.make_clients(mismatch_clients, true)) return 3;
    auto mismatch = PoolFleetOwner::create(std::move(mismatch_clients), 70, error);
    if (mismatch || error.find("identity mismatch at layer 7") == std::string::npos) return 4;

    SocketPairs sockets;
    if (!sockets.make_pairs()) return 5;
    SessionIdentity identity{"run-r638", "model-r638", "plan-r638", 91};
    auto owner = PoolFleetOwner::create_from_fds(sockets.client, 5, identity, 70, error);
    if (!owner || !error.empty()) return 6;
    if (owner->callback_user() != static_cast<void*>(&owner->fleet())) return 7;

    auto zero_attempt = PoolFleetOwner::create_from_fds(sockets.client, 5, identity, 70, error, 0);
    if (zero_attempt || error.find("attempt id") == std::string::npos) return 13;

    auto duplicate_fds = sockets.client;
    duplicate_fds[1] = duplicate_fds[0];
    auto duplicate = PoolFleetOwner::create_from_fds(duplicate_fds, 5, identity, 70, error);
    if (duplicate || error.find("descriptors must be valid and unique") == std::string::npos) return 8;

    float destination = 17.0f;
    if (pool_cpu_by_layer(owner->callback_user(), nullptr, nullptr, 1, 1, &destination, 48)) return 9;
    if (destination != 17.0f) return 10;

    sockets.close_peers();
    if (owner->shutdown_all(error)) return 11;
    if (owner->shutdown_all(error) || error != "session shutdown already attempted") return 12;
    return 0;
}
