#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vf::strata_bridge {
// Identity copied from the trusted per-layer Init used to prepare a binary
// socket. VFB3 repeats it on every request so the coordinator can reject a
// cross-run/model/plan/epoch socket before claiming the batch.
struct SessionIdentity {
    std::string run_id;
    std::string model_revision;
    std::string plan_id;
    uint64_t epoch = 0;
};

// Owns no descriptors. Both must be connected AF_UNIX stream sockets; they may
// refer to the same socket. The caller must keep them alive and exchange INIT
// and the binary-ready handshake before constructing this client.
class Client {
public:
    Client(int read_fd, int write_fd, int timeout_ms, uint32_t layer);
    Client(int read_fd, int write_fd, int timeout_ms, uint32_t layer, SessionIdentity identity);
    uint32_t layer() const noexcept { return layer_; }
    bool usable() const noexcept { return usable_; }
    const SessionIdentity* session_identity() const noexcept { return identity_ ? &*identity_ : nullptr; }
    bool run(uint64_t batch, uint64_t attempt, int tokens, int top_k, const std::vector<float>& inputs,
             const std::vector<int32_t>& ids, const std::vector<int32_t>& kinds,
             std::vector<float>& destination, std::string& error);
    bool shutdown(std::string& error);
private:
    void initialize() noexcept;
    bool transfer(bool writing, uint8_t* data, size_t size, int64_t deadline, std::string& error);
    int read_fd_, write_fd_, timeout_ms_;
    uint32_t layer_;
    std::optional<SessionIdentity> identity_;
    bool usable_;
};
}
