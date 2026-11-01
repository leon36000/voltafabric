#include "provider.hpp"
#include "route_graph_transaction.hpp"
#include "session_vfr3_collector.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef VF_R681_ENABLE_QWEN2MOE_BRIDGE
#include "qwen2moe_llama_route_bridge.hpp"
using vf::ggml::Qwen2MoELlamaRouteBridge;
#endif

#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using vf::ggml::BatchIdentity;
using vf::ggml::RouteBatch;
using vf::ggml::RouteGraphTransaction;
using vf::ggml::RouteStatus;
using vf::ggml::Vfr3CollectorContext;
using vf::strata_bridge::Client;
using vf::strata_bridge::SessionIdentity;

constexpr size_t kHidden = 2560;
constexpr uint32_t kLayer = 0;
constexpr uint64_t kEpoch = 6730261102ULL;
constexpr uint64_t kAttempt = 1;
constexpr uint64_t kPackBytes = 50'292'326'400ULL;
constexpr uint64_t kBlobBytes = 2'176'000ULL;
constexpr uint64_t kBlobOffset = 8ULL * kBlobBytes;
constexpr char kIndexSha[] = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
constexpr char kBlobSha[] = "2c7dcdcf2992bb6127b86407fb43abe67f4ba202acbe005a9bc67d7df9e072b0";

std::string json_quote(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[c >> 4] << hex[c & 15];
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << '"';
    return out.str();
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

bool send_line(int fd, const std::string &line) {
    return send_exact(fd, reinterpret_cast<const uint8_t *>(line.data()), line.size()) &&
        send_exact(fd, reinterpret_cast<const uint8_t *>("\n"), 1);
}

bool read_line(int fd, std::string &line) {
    line.clear();
    while (line.size() <= 65'536) {
        char c = 0;
        if (!read_exact(fd, reinterpret_cast<uint8_t *>(&c), 1)) return false;
        if (c == '\n') return true;
        line.push_back(c);
    }
    return false;
}

bool write_exact_at(int fd, const uint8_t *data, size_t size, uint64_t offset) {
    while (size != 0) {
        const ssize_t written = pwrite(fd, data, size, static_cast<off_t>(offset));
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        data += written;
        size -= static_cast<size_t>(written);
        offset += static_cast<uint64_t>(written);
    }
    return true;
}

std::vector<uint8_t> synthetic_expert_blob() {
    std::vector<uint8_t> blob(static_cast<size_t>(kBlobBytes), 0);
    constexpr size_t gate_up_bytes = (kHidden / 256 * 98) * 640 * 2;
    for (size_t offset = 0; offset < gate_up_bytes; offset += 98) blob[offset + 1] = 0x28;
    for (size_t offset = gate_up_bytes; offset < blob.size(); offset += 18) blob[offset + 1] = 0x28;
    return blob;
}

bool make_pack(const std::filesystem::path &path, const std::vector<uint8_t> &blob) {
    const int fd = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    bool ok = ftruncate(fd, static_cast<off_t>(kPackBytes)) == 0 &&
        write_exact_at(fd, blob.data(), blob.size(), kBlobOffset) && fsync(fd) == 0;
    const int saved = errno;
    close(fd);
    errno = saved;
    return ok;
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
    ~Graph() { if (ctx != nullptr) ggml_free(ctx); }
};

template <typename AppendRoute>
bool make_graph(AppendRoute &&append_route, const std::vector<float> &input,
                int &downstream_calls, Graph &g) {
    ggml_init_params params{};
    params.mem_size = 1U << 20;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    g.ctx = ggml_init(params);
    if (g.ctx == nullptr) return false;
    g.slot_outputs = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, kHidden, 1, 1);
    g.activations = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kHidden, 1);
    if (g.slot_outputs == nullptr || g.activations == nullptr || g.slot_outputs->data == nullptr ||
        g.activations->data == nullptr || !ggml_is_contiguous(g.slot_outputs) || !ggml_is_contiguous(g.activations)) return false;
    std::fill(static_cast<float *>(g.slot_outputs->data),
              static_cast<float *>(g.slot_outputs->data) + kHidden, 0.0f);
    std::copy(input.begin(), input.end(), static_cast<float *>(g.activations->data));
    ggml_tensor *router_logits = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 512, 1);
    if (router_logits == nullptr || router_logits->data == nullptr) return false;
    auto *logits = static_cast<float *>(router_logits->data);
    std::fill(logits, logits + 512, -100.0f);
    logits[8] = 0.0f;
    ggml_tensor *probabilities = ggml_soft_max(g.ctx, router_logits);
    ggml_tensor *selected = ggml_argsort_top_k(g.ctx, probabilities, 1);
    ggml_tensor *probabilities_3d = ggml_reshape_3d(g.ctx, probabilities, 1, 512, 1);
    ggml_tensor *selected_weights = ggml_get_rows(g.ctx, probabilities_3d, selected);
    ggml_tensor *final_weights = ggml_scale(g.ctx, selected_weights, 0.25f);
    if (probabilities == nullptr || selected == nullptr || selected_weights == nullptr ||
        final_weights == nullptr) return false;
    g.routed = append_route(g.ctx, g.slot_outputs, g.activations, selected, final_weights);
    if (g.routed == nullptr) return false;
    g.downstream = ggml_map_custom1(g.ctx, g.routed, copy_downstream, 1, &downstream_calls);
    g.graph = ggml_new_graph(g.ctx);
    if (g.downstream == nullptr || g.graph == nullptr) return false;
    ggml_build_forward_expand(g.graph, g.downstream);
    g.plan = ggml_graph_plan(g.graph, 1, nullptr);
    g.work.resize(g.plan.work_size);
    g.plan.work_data = g.work.empty() ? nullptr : g.work.data();
    return true;
}

struct Child {
    pid_t pid = -1;
    bool waited = false;
    ~Child() {
        if (pid > 0 && !waited) {
            kill(pid, SIGTERM);
            int status = 0;
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        }
    }
};

bool wait_child(Child &child, int &status) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (std::chrono::steady_clock::now() < deadline) {
        const pid_t result = waitpid(child.pid, &status, WNOHANG);
        if (result == child.pid) {
            child.waited = true;
            return true;
        }
        if (result < 0 && errno != EINTR) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    kill(child.pid, SIGKILL);
    while (waitpid(child.pid, &status, 0) < 0 && errno == EINTR) {}
    child.waited = true;
    return false;
}

#ifdef VF_R681_ENABLE_QWEN2MOE_BRIDGE
struct BridgeFactoryState {
    std::string run_id;
    std::string model_revision;
    std::string plan_id;
    uint64_t epoch = 0;
    uint64_t batch_id = 0;
    uint64_t attempt = 0;
};

bool make_qwen2moe_batch(void *userdata, const llama_ubatch &ubatch,
                         llm_graph_type graph_type, uint32_t expected_layers,
                         RouteBatch &batch) {
    if (userdata == nullptr || ubatch.n_tokens != 1 ||
        graph_type != LLM_GRAPH_TYPE_DEFAULT || expected_layers != 1) {
        return false;
    }
    const auto &state = *static_cast<const BridgeFactoryState *>(userdata);
    batch.identity = BatchIdentity{state.run_id, state.model_revision, state.plan_id,
        0, state.epoch, state.batch_id, state.attempt};
    batch.owner_by_expert.assign(512, 0);
    batch.owner_by_expert[8] = -1;
    return true;
}
#endif

int run_live_case(bool qwen2moe_bridge_mode) {
#ifndef VF_R681_ENABLE_QWEN2MOE_BRIDGE
    if (qwen2moe_bridge_mode) {
        std::cerr << "R681 Qwen2MoE bridge process target was not enabled at build time\n";
        return 77;
    }
#endif
    const char *bridge_env = std::getenv("VF_R672_BRIDGE_EXE");
    const char *worker_env = std::getenv("VF_R672_SESSION_EXE");
    const char *tmp_env = std::getenv("VF_R672_TMP_ROOT");
    const char *index_env = std::getenv("VF_R672_INDEX_PATH");
    if (bridge_env == nullptr || worker_env == nullptr || tmp_env == nullptr || index_env == nullptr) {
        std::cerr << "R672 integration requires explicit bridge, worker, index and Naos temp paths\n";
        return 77;
    }
    const std::filesystem::path bridge_path = std::filesystem::absolute(bridge_env);
    const std::filesystem::path worker_path = std::filesystem::absolute(worker_env);
    const std::filesystem::path temp_root = std::filesystem::absolute(tmp_env);
    const std::filesystem::path index_path = std::filesystem::absolute(index_env);
    if (!std::filesystem::is_regular_file(bridge_path) || !std::filesystem::is_regular_file(worker_path) ||
        !std::filesystem::is_regular_file(index_path) || !temp_root.is_absolute()) {
        std::cerr << "R672 integration inputs are not regular files or absolute paths\n";
        return 1;
    }
    std::filesystem::create_directories(temp_root);
    const std::string prefix = qwen2moe_bridge_mode ? "r681" : "r673";
    const std::string temp_template = (temp_root / (prefix + "-process-XXXXXX")).string();
    std::vector<char> temp_mutable(temp_template.begin(), temp_template.end());
    temp_mutable.push_back('\0');
    char *made_temp = mkdtemp(temp_mutable.data());
    if (made_temp == nullptr) {
        std::cerr << "could not create fresh VoltaFabric process-test directory\n";
        return 1;
    }
    const std::filesystem::path run_root(made_temp);
    const std::filesystem::path pack_path = run_root / "synthetic-native-pack.gguf";
    const std::filesystem::path ledger_path = run_root / "fresh-ledger";
    std::filesystem::create_directory(ledger_path);
    std::ifstream index_file(index_path, std::ios::binary);
    const std::string index((std::istreambuf_iterator<char>(index_file)), std::istreambuf_iterator<char>());
    const auto blob = synthetic_expert_blob();
    if (index_file.bad() || index.empty() || !make_pack(pack_path, blob)) {
        std::cerr << "could not create bounded sparse SC117-format fixture pack\n";
        return 1;
    }

    const std::string suffix = std::to_string(static_cast<unsigned long long>(getpid()));
    const uint64_t epoch = qwen2moe_bridge_mode ? 681'026'1101ULL : kEpoch;
    const uint64_t batch_id = (qwen2moe_bridge_mode ? 681'000'000ULL : 673'000'000ULL) +
        static_cast<uint64_t>(getpid());
    const std::string run_id = prefix + "-live-coordinator-" + suffix;
    const std::string model_revision = prefix + "-synthetic-sc117-" + suffix;
    const std::string plan_id = prefix + "-live-plan-" + suffix;
    std::ostringstream init;
    init << "{\"template\":{\"hidden\":2560,\"experts\":[8],\"routes\":[[{\"expert\":8,\"weight\":0.25}]],"
         << "\"registry\":{\"epoch\":" << epoch << ",\"copies\":[{\"expert\":8,\"rank\":0,\"local\":0}]},"
         << "\"stamp\":{\"run_id\":" << json_quote(run_id) << ",\"model_revision\":" << json_quote(model_revision)
         << ",\"plan_id\":" << json_quote(plan_id) << ",\"layer\":" << kLayer << ",\"batch\":0,\"epoch\":" << epoch << "},"
         << "\"preferred_owners\":{},\"resident_order\":{\"0\":[0]}},"
         << "\"model_experts\":[8],\"index\":" << json_quote(index)
         << ",\"pack_path\":" << json_quote(pack_path.string())
         << ",\"ledger_path\":" << json_quote(ledger_path.string())
         << ",\"endpoints\":[{\"rank\":0,\"argv\":[" << json_quote(worker_path.string()) << "]}],"
         << "\"bindings\":[{\"model_revision\":" << json_quote(model_revision)
         << ",\"rank\":0,\"layer\":" << kLayer << ",\"expert\":8,\"local\":0,\"epoch\":" << epoch
         << ",\"index_sha256\":" << json_quote(kIndexSha) << ",\"blob_sha256\":" << json_quote(kBlobSha) << "}],"
         << "\"resident_payload_limit\":3000000,\"timeout_ms\":10000}";

    int fds[2]{};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
        std::cerr << "AF_UNIX socketpair failed\n";
        return 1;
    }
    Child child;
    child.pid = fork();
    if (child.pid == 0) {
        close(fds[0]);
        if (dup2(fds[1], STDIN_FILENO) < 0 || dup2(fds[1], STDOUT_FILENO) < 0) _exit(126);
        if (fds[1] > STDERR_FILENO) close(fds[1]);
        execl(bridge_path.c_str(), bridge_path.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    close(fds[1]);
    if (child.pid < 0) {
        close(fds[0]);
        std::cerr << "fork of native bridge coordinator failed\n";
        return 1;
    }
    std::string line;
    if (!send_line(fds[0], init.str()) || !read_line(fds[0], line) ||
        line.find("\"state\":\"ready\"") == std::string::npos ||
        line.find("\"ranks\":[0]") == std::string::npos) {
        std::cerr << "actual native bridge coordinator did not admit: " << line << "\n";
        close(fds[0]);
        return 1;
    }
    if (!send_line(fds[0], "{\"op\":\"enter_binary\"}") || !read_line(fds[0], line) ||
        line != "{\"state\":\"binary_ready\"}") {
        std::cerr << "actual native bridge did not enter VFB3 mode: " << line << "\n";
        close(fds[0]);
        return 1;
    }

    SessionIdentity identity{run_id, model_revision, plan_id, epoch};
    Client client(fds[0], fds[0], 10'000, kLayer, identity);
    if (!client.usable()) {
        std::cerr << "VFB3 Client rejected the actual coordinator socket\n";
        close(fds[0]);
        return 1;
    }
    Vfr3CollectorContext collector_context{&client, {}, 0};
    std::unique_ptr<RouteGraphTransaction> transaction;
#ifdef VF_R681_ENABLE_QWEN2MOE_BRIDGE
    std::unique_ptr<Qwen2MoELlamaRouteBridge> route_bridge;
    const llm_vf_route_hooks *hooks = nullptr;
    BridgeFactoryState factory_state{run_id, model_revision, plan_id, epoch, batch_id, kAttempt};
#endif
    if (qwen2moe_bridge_mode) {
#ifdef VF_R681_ENABLE_QWEN2MOE_BRIDGE
        route_bridge = std::make_unique<Qwen2MoELlamaRouteBridge>(
            make_qwen2moe_batch, &factory_state, vf::ggml::collect_vfr3, &collector_context);
        hooks = route_bridge->hooks();
        llama_ubatch ubatch{};
        ubatch.n_tokens = 1;
        if (hooks == nullptr || hooks->begin == nullptr || hooks->append_qwen2moe == nullptr ||
            hooks->finish == nullptr || hooks->abort == nullptr ||
            !hooks->begin(hooks->userdata, ubatch, LLM_GRAPH_TYPE_DEFAULT, 1)) {
            std::cerr << "R681 callback bridge rejected fresh ubatch identity\n";
            close(fds[0]);
            return 1;
        }
#endif
    } else {
        RouteBatch batch;
        batch.identity = BatchIdentity{run_id, model_revision, plan_id, kLayer, epoch, batch_id, kAttempt};
        batch.hidden = kHidden;
        batch.slots = 1;
        batch.tokens = 1;
        batch.experts = 512;
        batch.owner_kinds = {-1};
        transaction = std::make_unique<RouteGraphTransaction>(
            std::move(batch), vf::ggml::collect_vfr3, &collector_context);
    }
    std::vector<float> input(kHidden);
    for (size_t i = 0; i < kHidden; ++i) input[i] = std::sin(static_cast<float>(i) * 0.17f) / 16.0f;
    Graph graph;
    int downstream_calls = 0;
    auto append_route = [&](ggml_context *ctx, ggml_tensor *slot_outputs,
                            ggml_tensor *activations, ggml_tensor *selected,
                            ggml_tensor *final_weights) -> ggml_tensor * {
#ifdef VF_R681_ENABLE_QWEN2MOE_BRIDGE
        if (route_bridge != nullptr) {
            return hooks->append_qwen2moe(hooks->userdata, ctx, static_cast<int32_t>(kLayer),
                activations, selected, final_weights, slot_outputs);
        }
#endif
        return transaction->append_dynamic(ctx, slot_outputs, activations, selected, final_weights);
    };
    if (!make_graph(append_route, input, downstream_calls, graph)) {
#ifdef VF_R681_ENABLE_QWEN2MOE_BRIDGE
        if (route_bridge != nullptr) hooks->abort(hooks->userdata);
#endif
        std::cerr << "could not build single-route GGML graph\n";
        close(fds[0]);
        return 1;
    }
    if (transaction != nullptr) transaction->attach_abort(graph.plan);
    const ggml_status status = ggml_graph_compute(graph.graph, &graph.plan);
    bool accepted = false;
#ifdef VF_R681_ENABLE_QWEN2MOE_BRIDGE
    if (route_bridge != nullptr) {
        accepted = hooks->finish(hooks->userdata, status == GGML_STATUS_SUCCESS);
        if (!accepted) hooks->abort(hooks->userdata);
    } else
#endif
    {
        accepted = transaction->accepts(status);
    }

    std::vector<float> raw_expected;
    std::string provider_error;
    const bool oracle_ok = vf::native_ggml::forward(18, 20, blob, input, raw_expected, provider_error);
    bool values_match = accepted && oracle_ok && downstream_calls == 1 && collector_context.calls == 1 &&
        (transaction == nullptr || transaction->status() == RouteStatus::ready) &&
        raw_expected.size() == kHidden;
    if (values_match) {
        const auto *actual = static_cast<const float *>(graph.downstream->data);
        for (size_t i = 0; i < kHidden; ++i) {
            const float decoded = ggml_fp16_to_fp32(ggml_fp32_to_fp16(raw_expected[i]));
            const float expected = static_cast<float>(0.25 * static_cast<double>(decoded));
            if (!std::isfinite(actual[i]) || !std::isfinite(expected) || std::fabs(actual[i] - expected) > 1e-7f) {
                values_match = false;
                break;
            }
        }
    }
    ggml_status replay = GGML_STATUS_ABORTED;
    if (transaction != nullptr) replay = ggml_graph_compute(graph.graph, &graph.plan);
    std::string shutdown_error;
    const bool closed = client.shutdown(shutdown_error);
    close(fds[0]);
    int child_status = 0;
    const bool exited = wait_child(child, child_status);

    size_t claim_count = 0;
    std::string claim_text;
    for (const auto &entry : std::filesystem::directory_iterator(ledger_path)) {
        ++claim_count;
        std::ifstream claim(entry.path());
        claim_text.assign(std::istreambuf_iterator<char>(claim), std::istreambuf_iterator<char>());
    }
    const bool claim_ok = claim_count == 1 && claim_text.find("RESERVED_UNKNOWN_NO_REPLAY") != std::string::npos &&
        claim_text.find(run_id) != std::string::npos && claim_text.find(std::to_string(batch_id)) != std::string::npos &&
        claim_text.find("\"attempt\":1") != std::string::npos;
    const bool replay_ok = transaction == nullptr ||
        (replay == GGML_STATUS_ABORTED && transaction->status() == RouteStatus::invalid_batch);
    if (!values_match || !replay_ok ||
        collector_context.calls != 1 || !closed || !exited || !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0 || !claim_ok) {
        std::cerr << "live bridge test failed: graph=" << ggml_status_to_string(status)
                  << " replay=" << (transaction != nullptr ? ggml_status_to_string(replay) : "not-run")
                  << " accepted=" << accepted
                  << " values=" << values_match << " oracle=" << provider_error
                  << " collector=" << collector_context.error << " shutdown=" << shutdown_error
                  << " child_exited=" << exited << " child_status=" << child_status
                  << " claims=" << claim_count << "\n";
        return 1;
    }
    std::cout << (qwen2moe_bridge_mode
        ? "R681 Qwen2MoE callback bridge through actual coordinator + vf-rank-session PASS"
        : "R673 GGML softmax/top-k/gather/scale route through actual coordinator + vf-rank-session PASS")
              << " run=" << run_id << " model=" << model_revision << " plan=" << plan_id
              << " layer=" << kLayer << " batch=" << batch_id << " epoch=" << epoch
              << " attempt=" << kAttempt << " sparse_pack_logical_bytes=" << kPackBytes
              << " payload_bytes=" << blob.size() << "\n";
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    if (argc == 1) return run_live_case(false);
    if (argc == 2 && std::strcmp(argv[1], "--qwen2moe-bridge") == 0) {
        return run_live_case(true);
    }
    std::cerr << "unsupported process fixture mode\n";
    return 2;
}
