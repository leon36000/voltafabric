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
    bool make_clients(PoolFleetOwner::Clients& clients) {
        for (size_t layer = 0; layer < client.size(); ++layer) {
            int pair[2] = {-1, -1};
            if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) return false;
            client[layer] = pair[0];
            peer[layer] = pair[1];
            clients[layer] = std::make_unique<Client>(pair[0], pair[0], 5, static_cast<uint32_t>(layer));
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

    SocketPairs sockets;
    PoolFleetOwner::Clients clients{};
    if (!sockets.make_clients(clients)) return 3;
    auto owner = PoolFleetOwner::create(std::move(clients), 70, error);
    if (!owner || !error.empty()) return 4;
    if (owner->callback_user() != static_cast<void*>(&owner->fleet())) return 5;

    float destination = 17.0f;
    if (pool_cpu_by_layer(owner->callback_user(), nullptr, nullptr, 1, 1, &destination, 48)) return 6;
    if (destination != 17.0f) return 7;

    sockets.close_peers();
    if (owner->shutdown_all(error)) return 8;
    if (owner->shutdown_all(error) || error != "session shutdown already attempted") return 9;
    return 0;
}
