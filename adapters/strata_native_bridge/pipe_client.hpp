#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vf::strata_bridge {
// Owns no descriptors. Both must be connected AF_UNIX stream sockets; they may
// refer to the same socket. The caller must keep them alive and exchange INIT
// and the binary-ready handshake before constructing this client.
class Client {
public:
    Client(int read_fd, int write_fd, int timeout_ms, uint32_t layer);
    uint32_t layer() const noexcept { return layer_; }
    bool usable() const noexcept { return usable_; }
    bool run(uint64_t batch, int tokens, int top_k, const std::vector<float>& inputs,
             const std::vector<int32_t>& ids, const std::vector<int32_t>& kinds,
             std::vector<float>& destination, std::string& error);
    bool shutdown(std::string& error);
private:
    bool transfer(bool writing, uint8_t* data, size_t size, int64_t deadline, std::string& error);
    int read_fd_, write_fd_, timeout_ms_;
    uint32_t layer_;
    bool usable_;
};
}
