#!/usr/bin/env python3
"""Generate an opt-in Strata generator patch that adopts VoltaFabric's 48 CPU sessions."""
import difflib
import hashlib
import pathlib
import shutil
import subprocess
import tempfile

REV = "fb58e0dbc8399662c0e47c76578c6e878b14f6cf"
CPU_HOOK_SHA256 = "2e86bbcb65ad6763cd9c47ec02a85da7ce3147ee50633e974ec5815f2f6bd10a"
R640_SHA256 = "c8e2848813ebd079a71cfa81a8d31c62214a413e0be50803e0b9619b8481da9b"
FILES = [
    "CMakeLists.txt",
    "include/strata/core/verify.hpp",
    "include/strata/core/expert_source.hpp",
    "src/core/expert_source.cpp",
    "src/core/verify.cpp",
    "src/program/generate.cpp",
]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)

def patch_text(before, after, path):
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/" + path, tofile="b/" + path, n=3,
    ))

def transform_generate(text):
    text = replace_once(
        text,
        '#include "strata/artifact/gguf_reader.hpp"\n',
        '#include "strata/artifact/gguf_reader.hpp"\n'
        '#ifdef STRATA_VOLTAFABRIC_POOL\n'
        '#include "pool_hook.hpp"\n'
        '#endif\n',
        "conditional bridge include",
    )
    text = replace_once(
        text,
        "namespace {\n// Windows' WDDM driver model:",
        "#ifdef STRATA_VOLTAFABRIC_POOL\n"
        "static std::unique_ptr<vf::strata_bridge::PoolFleetOwner> g_vf_pool_owner;\n"
        "#endif\n\n"
        "namespace {\n// Windows' WDDM driver model:",
        "generation-scoped owner storage",
    )
    text = replace_once(
        text,
        'struct Options {\n    std::string pack = "pack/full";\n',
        'struct Options {\n    std::string pack = "pack/full";\n'
        '#ifdef STRATA_VOLTAFABRIC_POOL\n'
        '    std::string vf_session_fds, vf_run_id, vf_model_revision, vf_plan_id;\n'
        '    uint64_t vf_epoch = 0, vf_first_batch = 0, vf_first_attempt = 1;\n'
        '    int vf_timeout_ms = 5000;\n'
        '    bool vf_epoch_set = false, vf_first_batch_set = false;\n'
        '    bool vf_first_attempt_set = false, vf_timeout_set = false;\n'
        '#endif\n',
        "VoltaFabric CLI state",
    )
    text = replace_once(
        text,
        "bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {\n",
        "#ifdef STRATA_VOLTAFABRIC_POOL\n"
        "bool parse_vf_u64(const char* text, uint64_t& value) {\n"
        "    if (text == nullptr || text[0] == '\\0') return false;\n"
        "    const char* end = text + std::strlen(text);\n"
        "    auto parsed = std::from_chars(text, end, value);\n"
        "    return parsed.ec == std::errc{} && parsed.ptr == end;\n"
        "}\n"
        "bool parse_vf_session_fds(const std::string& text, std::array<int, 48>& fds) {\n"
        "    size_t begin = 0;\n"
        "    for (size_t i = 0; i < fds.size(); ++i) {\n"
        "        size_t end = text.find(',', begin);\n"
        "        if (end == std::string::npos) end = text.size();\n"
        "        if (end == begin) return false;\n"
        "        int fd = -1;\n"
        "        auto parsed = std::from_chars(text.data() + begin, text.data() + end, fd);\n"
        "        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + end || fd < 3) return false;\n"
        "        for (size_t prior = 0; prior < i; ++prior) if (fds[prior] == fd) return false;\n"
        "        fds[i] = fd;\n"
        "        begin = end + 1;\n"
        "        if (i + 1 < fds.size() && end == text.size()) return false;\n"
        "        if (i + 1 == fds.size() && end != text.size()) return false;\n"
        "    }\n"
        "    return true;\n"
        "}\n"
        "#endif\n\n"
        "bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {\n",
        "bounded descriptor and integer parsing",
    )
    text = replace_once(
        text,
        "int main(int argc, char** argv) {\n",
        "#ifdef STRATA_VOLTAFABRIC_POOL\n"
        "static int strata_generate_main(int argc, char** argv) {\n"
        "#else\n"
        "int main(int argc, char** argv) {\n"
        "#endif\n",
        "owner-aware entrypoint",
    )
    text = replace_once(
        text,
        '        if (a == "--expert-cache") {\n',
        '        if (a == "--vf-session-fds") o.vf_session_fds = next("--vf-session-fds");\n'
        '        else if (a == "--vf-run-id") o.vf_run_id = next("--vf-run-id");\n'
        '        else if (a == "--vf-model-revision") o.vf_model_revision = next("--vf-model-revision");\n'
        '        else if (a == "--vf-plan-id") o.vf_plan_id = next("--vf-plan-id");\n'
        '        else if (a == "--vf-epoch") {\n'
        '            o.vf_epoch_set = parse_vf_u64(next("--vf-epoch"), o.vf_epoch);\n'
        '            if (!o.vf_epoch_set) { std::fprintf(stderr, "invalid --vf-epoch\\n"); return 2; }\n'
        '        }\n'
        '        else if (a == "--vf-first-batch") {\n'
        '            o.vf_first_batch_set = parse_vf_u64(next("--vf-first-batch"), o.vf_first_batch);\n'
        '            if (!o.vf_first_batch_set) { std::fprintf(stderr, "invalid --vf-first-batch\\n"); return 2; }\n'
        '        }\n'
        '        else if (a == "--vf-first-attempt") {\n'
        '            o.vf_first_attempt_set = parse_vf_u64(next("--vf-first-attempt"), o.vf_first_attempt);\n'
        '            if (!o.vf_first_attempt_set) { std::fprintf(stderr, "invalid --vf-first-attempt\\n"); return 2; }\n'
        '        }\n'
        '        else if (a == "--vf-timeout-ms") {\n'
        '            uint64_t value = 0;\n'
        '            o.vf_timeout_set = parse_vf_u64(next("--vf-timeout-ms"), value) && value >= 1 && value <= 60000;\n'
        '            if (!o.vf_timeout_set) { std::fprintf(stderr, "invalid --vf-timeout-ms\\n"); return 2; }\n'
        '            o.vf_timeout_ms = static_cast<int>(value);\n'
        '        }\n'
        '        else if (a == "--expert-cache") {\n',
        "session option parsing",
    )
    text = replace_once(
        text,
        "    }\n#if defined(STRATA_USE_HIP)\n    {   // gfx1151",
        "    }\n#ifdef STRATA_VOLTAFABRIC_POOL\n"
        "    {\n"
        "        const bool any_vf_identity = !o.vf_run_id.empty() || !o.vf_model_revision.empty() ||\n"
        "            !o.vf_plan_id.empty() || o.vf_epoch_set || o.vf_first_batch_set ||\n"
        "            o.vf_first_attempt_set || o.vf_timeout_set;\n"
        "        if (o.vf_session_fds.empty()) {\n"
        "            if (any_vf_identity) { std::fprintf(stderr, \"VoltaFabric identity requires --vf-session-fds\\n\"); return 2; }\n"
        "        } else {\n"
        "            std::array<int, 48> fds{};\n"
        "            if (!parse_vf_session_fds(o.vf_session_fds, fds) || o.vf_run_id.empty() ||\n"
        "                o.vf_model_revision.empty() || o.vf_plan_id.empty() || !o.vf_epoch_set ||\n"
        "                !o.vf_first_batch_set || !o.vf_first_attempt_set || !o.vf_timeout_set) {\n"
        "                std::fprintf(stderr, \"VoltaFabric pool needs 48 unique inherited stream fds and a complete identity\\n\");\n"
        "                return 2;\n"
        "            }\n"
        "            vf::strata_bridge::SessionIdentity identity{o.vf_run_id, o.vf_model_revision, o.vf_plan_id, o.vf_epoch};\n"
        "            std::string vf_error;\n"
        "            g_vf_pool_owner = vf::strata_bridge::PoolFleetOwner::create_from_fds(\n"
        "                fds, o.vf_timeout_ms, std::move(identity), o.vf_first_batch, vf_error, o.vf_first_attempt);\n"
        "            if (!g_vf_pool_owner) {\n"
        "                std::fprintf(stderr, \"VoltaFabric pool admission failed: %s\\n\", vf_error.c_str());\n"
        "                return 2;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "#endif\n"
        "#if defined(STRATA_USE_HIP)\n    {   // gfx1151",
        "pre-CUDA owner admission",
    )
    text = replace_once(
        text,
        '            vf::strata_bridge::SessionIdentity identity{o.vf_run_id, o.vf_model_revision, o.vf_plan_id, o.vf_epoch};\n',
        '            if (o.no_pool || o.no_capture) {\n'
        '                std::fprintf(stderr, "VoltaFabric pool requires the captured expert-pool generation path\\n");\n'
        '                return 2;\n'
        '            }\n'
        '            vf::strata_bridge::SessionIdentity identity{o.vf_run_id, o.vf_model_revision, o.vf_plan_id, o.vf_epoch};\n',
        "reject paths that bypass the routed pool",
    )
    text = replace_once(
        text,
        '            if (o.no_pool || o.no_capture) {\n'
        '                std::fprintf(stderr, "VoltaFabric pool requires the captured expert-pool generation path\\n");\n'
        '                return 2;\n'
        '            }\n',
        '            if (o.no_pool || o.no_capture) {\n'
        '                std::fprintf(stderr, "VoltaFabric pool requires the captured expert-pool generation path\\n");\n'
        '                return 2;\n'
        '            }\n'
        '            if (!o.dump_logits.empty() || !o.dump_residual.empty() || !o.dump_mixed.empty() ||\n'
        '                !o.dump_layers.empty() || !o.dump_halves.empty() || !o.dump_routing.empty() ||\n'
        '                !o.dump_final_r.empty() || !o.window_hashes.empty() ||\n'
        '                std::getenv("STRATA_DUMP_FIRST_LOGITS") != nullptr) {\n'
        '                std::fprintf(stderr, "VoltaFabric mode refuses diagnostic sidecar files because they are not transactional\\n");\n'
        '                return 2;\n'
        '            }\n',
        "refuse non-transactional diagnostic output sidecars",
    )
    text = replace_once(
        text,
        "    Drive drive;\n    // #731 (opt-in, STRATA_DISJOINT_ADAPT=1):",
        "    Drive drive;\n"
        "#ifdef STRATA_VOLTAFABRIC_POOL\n"
        "    if (g_vf_pool_owner) {\n"
        "        drive.d.vf_cpu_miss_pool = &vf::strata_bridge::pool_cpu_misses_by_layer;\n"
        "        drive.d.vf_cpu_miss_context = g_vf_pool_owner->callback_user();\n"
        "    }\n"
        "#endif\n"
        "    // #731 (opt-in, STRATA_DISJOINT_ADAPT=1):",
        "Drive callback installation",
    )
    failure_block = (
        '        if (drive.d.failed) {\n'
        '            std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\\n",\n'
        '                         (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,\n'
        '                         drive.d.fail ? drive.d.fail : "(no message)");\n'
        '            return 1;\n'
        '        }\n'
    )
    text = replace_once(
        text,
        '        if (final_r != nullptr) {\n',
        failure_block + '        if (final_r != nullptr) {\n',
        "refuse failed pool output before writing per-token artifacts",
    )
    text = replace_once(
        text,
        '            std::fwrite(final_r_host.data(), sizeof(float), final_r_host.size(), final_r);\n'
        '        }\n' + failure_block +
        '        {\n'
        '            // **THE LAYER LOOP ITSELF, WHICH IS WHAT `--gpu-only-full` HAS TO BE COMPARED AGAINST.**\n',
        '            std::fwrite(final_r_host.data(), sizeof(float), final_r_host.size(), final_r);\n'
        '        }\n'
        '        {\n'
        '            // **THE LAYER LOOP ITSELF, WHICH IS WHAT `--gpu-only-full` HAS TO BE COMPARED AGAINST.**\n',
        "remove the old post-artifact pool failure check",
    )
    if not text.endswith("    return 0;\n}\n"):
        raise SystemExit("generator main tail changed")
    text += (
        "\n#ifdef STRATA_VOLTAFABRIC_POOL\n"
        "int main(int argc, char** argv) {\n"
        "    const int result = strata_generate_main(argc, argv);\n"
        "    if (g_vf_pool_owner) {\n"
        "        std::string error;\n"
        "        if (!g_vf_pool_owner->shutdown_all(error)) {\n"
        "            std::fprintf(stderr, \"VoltaFabric pool shutdown failed: %s\\n\", error.c_str());\n"
        "            if (result == 0) return 1;\n"
        "        }\n"
        "        g_vf_pool_owner.reset();\n"
        "    }\n"
        "    return result;\n"
        "}\n"
        "#endif\n"
    )
    return text

def transform_cmake(text):
    text = replace_once(
        text,
        'option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1012/gfx1030 unvalidated)" OFF)\n',
        'option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1012/gfx1030 unvalidated)" OFF)\n'
        'option(STRATA_VOLTAFABRIC_POOL "Opt-in VoltaFabric CPU-miss callback in strata generate (Linux CUDA only)" OFF)\n'
        'set(VOLTAFABRIC_ADAPTER_ROOT "" CACHE PATH "VoltaFabric checkout containing adapters/strata_native_bridge")\n',
        "opt-in CMake options",
    )
    text = replace_once(
        text,
        'if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)\n  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")\nendif()\n',
        'if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)\n  message(FATAL_ERROR "STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP are separate backends; enable only one")\nendif()\n'
        'if(STRATA_VOLTAFABRIC_POOL)\n'
        '  find_package(Threads REQUIRED)\n'
        '  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP OR STRATA_HIP_GFX906)\n'
        '    message(FATAL_ERROR "STRATA_VOLTAFABRIC_POOL currently requires Linux with the native CUDA backend")\n'
        '  endif()\n'
        '  if(NOT EXISTS "${VOLTAFABRIC_ADAPTER_ROOT}/adapters/strata_native_bridge/pool_hook.cpp" OR\n'
        '     NOT EXISTS "${VOLTAFABRIC_ADAPTER_ROOT}/adapters/strata_native_bridge/pool_hook.hpp" OR\n'
        '     NOT EXISTS "${VOLTAFABRIC_ADAPTER_ROOT}/adapters/strata_native_bridge/pipe_client.cpp" OR\n'
        '     NOT EXISTS "${VOLTAFABRIC_ADAPTER_ROOT}/adapters/strata_native_bridge/pipe_client.hpp")\n'
        '    message(FATAL_ERROR "VOLTAFABRIC_ADAPTER_ROOT must point to a VoltaFabric checkout with the Strata bridge")\n'
        '  endif()\n'
        '  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/core/expert_source.hpp" _strata_expert_source)\n'
        '  string(FIND "${_strata_expert_source}" "vf_cpu_miss_pool" _strata_vf_callback)\n'
        '  if(_strata_vf_callback EQUAL -1)\n'
        '    message(FATAL_ERROR "Apply the reviewed R640 mixed-owner Strata patch before enabling VoltaFabric")\n'
        '  endif()\n'
        'endif()\n',
        "backend and adapter root guard",
    )
    text = replace_once(
        text,
        '  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec ${_strata_gpu_runtime_target})\n',
        '  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec ${_strata_gpu_runtime_target})\n'
        '  if(STRATA_VOLTAFABRIC_POOL)\n'
        '    target_sources(strata PRIVATE\n'
        '      "${VOLTAFABRIC_ADAPTER_ROOT}/adapters/strata_native_bridge/pipe_client.cpp"\n'
        '      "${VOLTAFABRIC_ADAPTER_ROOT}/adapters/strata_native_bridge/pool_hook.cpp")\n'
        '    target_include_directories(strata PRIVATE "${VOLTAFABRIC_ADAPTER_ROOT}/adapters/strata_native_bridge")\n'
        '    target_compile_definitions(strata PRIVATE STRATA_VOLTAFABRIC_POOL=1)\n'
        '    target_link_libraries(strata PRIVATE Threads::Threads)\n'
        '  endif()\n',
        "generator bridge target wiring",
    )
    text = replace_once(
        text,
        '  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)\n',
        '  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)\n'
        '  if(STRATA_VOLTAFABRIC_POOL)\n'
        '    target_compile_definitions(strata_engine PRIVATE STRATA_VOLTAFABRIC_POOL=1)\n'
        '  endif()\n',
        "compile the final-owner hook in the engine library",
    )
    return text

def transform_expert_source(text):
    text = replace_once(
        text,
        '    if (d.failed) return;   // a previous layer already failed; do not make it worse\n',
        '    if (d.failed) {\n'
        '        return;   // fail-closed output handling is inserted below\n'
        '    }\n',
        "prepare the dispatch failure guard",
    )
    text = replace_once(
        text,
        '    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);\n\n'
        '    d.src->begin_layer(d.layers, ids, k);\n',
        '    if (k > (int64_t) d.jobs.size()) d.jobs.resize((size_t) k);\n'
        '#ifdef STRATA_VOLTAFABRIC_POOL\n'
        '    if (d.vf_cpu_miss_pool != nullptr) {\n'
        '        if (k < 1 || k > 64) {\n'
        '            d.failed = true; d.fail = "VoltaFabric single-token route width is outside 1..64";\n'
        '            d.fail_layer = d.layers; return;\n'
        '        }\n'
        '        for (int64_t i = 0; i < k; ++i) {\n'
        '            if (ids[i] < 0 || ids[i] >= d.n_expert) {\n'
        '                d.failed = true; d.fail = "a routed expert id is out of range";\n'
        '                d.fail_layer = d.layers; d.fail_expert = ids[i]; return;\n'
        '            }\n'
        '        }\n'
        '    }\n'
        '#endif\n'
        '\n'
        '    d.src->begin_layer(d.layers, ids, k);\n',
        "validate the opt-in single-token route extent",
    )
    text = replace_once(
        text,
        '    d.src->begin_layer(d.layers, ids, k);\n\n'
        '    // Clause 1: rebuilt from `x_f` on EVERY call.',
        '    d.src->begin_layer(d.layers, ids, k);\n'
        '#ifdef STRATA_VOLTAFABRIC_POOL\n'
        '    const bool vf_pool_active = d.vf_cpu_miss_pool != nullptr;\n'
        '    std::vector<const uint8_t*> vf_blobs;\n'
        '    if (vf_pool_active) {\n'
        '        vf_blobs.resize((size_t) k);\n'
        '        for (int64_t i = 0; i < k; ++i) {\n'
        '            vf_blobs[(size_t) i] = d.src->blob(d.layers, ids[i]);\n'
        '            if (vf_blobs[(size_t) i] == nullptr) {\n'
        '                d.failed = true; d.fail = "the expert source could not produce a blob";\n'
        '                d.fail_layer = d.layers; d.fail_expert = ids[i]; ++d.missing; return;\n'
        '            }\n'
        '        }\n'
        '    }\n'
        '#endif\n'
        '\n'
        '    // Clause 1: rebuilt from `x_f` on EVERY call.',
        "preflight the pinned expert source before owner callbacks",
    )
    text = replace_once(
        text,
        '    for (int64_t i = 0; i < k; ++i) {\n'
        '        const int64_t e = ids[i];\n',
        '#ifdef STRATA_VOLTAFABRIC_POOL\n'
        '    std::vector<int32_t> vf_kinds;\n'
        '    std::vector<uint8_t> vf_handled;\n'
        '    if (vf_pool_active) {\n'
        '        vf_kinds.resize((size_t) k);\n'
        '        vf_handled.assign((size_t) k, 0);\n'
        '        bool any_vf_cpu_miss = false;\n'
        '        for (int64_t i = 0; i < k; ++i) {\n'
        '            const int64_t e = ids[i];\n'
        '            const bool hit = use_hits && (graph_hits\n'
        '                ? d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) e] >= 0\n'
        '                : d.is_hit[(size_t) i] != 0);\n'
        '            bool remote_owns = false;\n'
        '            if (!hit)\n'
        '                for (int r = 0; r < d.remote_count; ++r) remote_owns |= d.remote[r]->owns(i);\n'
        '            vf_kinds[(size_t) i] = hit ? 0 : remote_owns ? 2 : -1;\n'
        '            any_vf_cpu_miss |= vf_kinds[(size_t) i] < 0;\n'
        '        }\n'
        '        if (any_vf_cpu_miss) {\n'
        '            if (!d.vf_cpu_miss_pool(d.vf_cpu_miss_context, x_f, ids, vf_kinds.data(), 1, k, out, d.layers)) {\n'
        '                bool external_drained = true;\n'
        '                if (d.remote_count > 0) {\n'
        '                    static thread_local std::string drain_error;\n'
        '                    for (int r = 0; r < d.remote_count; ++r)\n'
        '                        if (!d.remote[r]->finish(out, drain_error)) external_drained = false;\n'
        '                }\n'
        '                d.failed = true;\n'
        '                d.fail = external_drained\n'
        '                    ? "VoltaFabric single-token callback failed after external owners drained"\n'
        '                    : "VoltaFabric single-token callback failed and an external owner did not drain";\n'
        '                d.fail_layer = d.layers; return;\n'
        '            }\n'
        '            for (int64_t i = 0; i < k; ++i) vf_handled[(size_t) i] = vf_kinds[(size_t) i] < 0;\n'
        '        }\n'
        '    }\n'
        '#endif\n'
        '\n'
        '    for (int64_t i = 0; i < k; ++i) {\n'
        '        const int64_t e = ids[i];\n',
        "dispatch the final single-token owner map to VoltaFabric",
    )
    text = replace_once(
        text,
        '        if (remote_owns) {\n'
        '            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));\n'
        '            continue;\n'
        '        }\n'
        '\n'
        '        // `njobs` indexes the JOB ARRAY',
        '        if (remote_owns) {\n'
        '            std::memset(out + (size_t) i * (size_t) n_embd, 0, (size_t) n_embd * sizeof(float));\n'
        '            continue;\n'
        '        }\n'
        '#ifdef STRATA_VOLTAFABRIC_POOL\n'
        '        if (vf_pool_active && vf_handled[(size_t) i]) continue;\n'
        '#endif\n'
        '\n'
        '        // `njobs` indexes the JOB ARRAY',
        "skip rows committed by the VoltaFabric owner",
    )
    text = replace_once(
        text,
        '            if (!d.remote[r]->finish(out, remote_error)) {\n'
        '                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;\n'
        '            }\n'
        '    }\n'
        '    ++d.layers;\n',
        '            if (!d.remote[r]->finish(out, remote_error)) {\n'
        '                d.failed = true; d.fail = remote_error.c_str(); d.fail_layer = d.layers; return;\n'
        '            }\n'
        '    }\n'
        '    ++d.layers;\n',
        "clear an opt-in mixed row set when remote drain fails",
    )
    start = text.index("void expert_pool_dispatch(void* user")
    end = text.index("\nnamespace {\n// the verify window", start)
    dispatch = text[start:end]
    dispatch = replace_once(
        dispatch,
        '        const uint8_t* b = d.src->blob(d.layers, e);\n',
        '#ifdef STRATA_VOLTAFABRIC_POOL\n'
        '        const uint8_t* b = vf_pool_active ? vf_blobs[(size_t) i] : d.src->blob(d.layers, e);\n'
        '#else\n'
        '        const uint8_t* b = d.src->blob(d.layers, e);\n'
        '#endif\n',
        "reuse the preflighted source row in opt-in mode",
    )
    dispatch = replace_once(
        dispatch,
        '    ExpertDispatch& d = *(ExpertDispatch*) user;\n'
        '    if (d.failed) {\n',
        '    ExpertDispatch& d = *(ExpertDispatch*) user;\n'
        '    auto vf_clear_output = [&]() noexcept {\n'
        '        if (d.vf_cpu_miss_pool != nullptr && out != nullptr && n_embd > 0 && n_embd <= 65536 && k > 0 && k <= 64)\n'
        '            std::memset(out, 0, (size_t) k * (size_t) n_embd * sizeof(float));\n'
        '    };\n'
        '    if (d.failed) {\n'
        '        vf_clear_output();\n',
        "fail-closed callback buffer guard",
    )
    dispatch = dispatch.replace('d.failed = true;', 'vf_clear_output(); d.failed = true;')
    text = text[:start] + dispatch + text[end:]
    return text

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("donor", type=pathlib.Path)
    parser.add_argument("cpu_hook_patch", type=pathlib.Path)
    parser.add_argument("r640_patch", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    root = args.donor.resolve()
    cpu_hook_patch, r640_patch = args.cpu_hook_patch.resolve(), args.r640_patch.resolve()
    output = args.output.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REV:
        raise SystemExit(f"wrong Strata revision: {revision}")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", *FILES], text=True).strip()
    if dirty:
        raise SystemExit("pinned Strata generator or CMake files are modified")
    if sha256(cpu_hook_patch) != CPU_HOOK_SHA256:
        raise SystemExit("R639 CPU-hook patch SHA-256 mismatch")
    if sha256(r640_patch) != R640_SHA256:
        raise SystemExit("R640 mixed-owner patch SHA-256 mismatch")
    original = {path: (root / path).read_text() for path in FILES}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r641-strata-", dir=str(output.parent)) as td:
        stage = pathlib.Path(td)
        for path in FILES:
            target = stage / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(original[path])
        subprocess.run(["patch", "--silent", "-p1", "-d", str(stage), "-i", str(cpu_hook_patch)], check=True)
        subprocess.run(["patch", "--silent", "-p1", "-d", str(stage), "-i", str(r640_patch)], check=True)
        after_r640 = {path: (stage / path).read_text() for path in FILES}
        final = dict(after_r640)
        final["CMakeLists.txt"] = transform_cmake(final["CMakeLists.txt"])
        final["src/core/expert_source.cpp"] = transform_expert_source(final["src/core/expert_source.cpp"])
        final["src/program/generate.cpp"] = transform_generate(final["src/program/generate.cpp"])
        delta = "".join(patch_text(after_r640[path], final[path], path) for path in FILES if after_r640[path] != final[path])
        if not delta:
            raise SystemExit("no R641 generator changes produced")
        # Apply to a fresh copy of the post-R640 tree, then compare every changed source byte-for-byte.
        verify = stage / "verify"
        for path in FILES:
            target = verify / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(after_r640[path])
        delta_file = stage / "r641.patch"
        delta_file.write_text(delta)
        subprocess.run(["patch", "--silent", "-p1", "-d", str(verify), "-i", str(delta_file)], check=True)
        for path in FILES:
            expected = final[path] if path in ("CMakeLists.txt", "src/core/expert_source.cpp", "src/program/generate.cpp") else after_r640[path]
            if (verify / path).read_text() != expected:
                raise SystemExit(f"R641 readback mismatch: {path}")
        if ("g_vf_pool_owner->shutdown_all(error)" not in delta or
                "drive.d.vf_cpu_miss_context" not in delta or
                "vf_cpu_miss_pool(d.vf_cpu_miss_context" not in delta or
                "find_package(Threads REQUIRED)" not in delta):
            raise SystemExit("generator lifetime/callback integration missing")
        output.write_text(delta)
    print(f"donor_revision={REV}")
    print(f"cpu_hook_patch_sha256={CPU_HOOK_SHA256}")
    print(f"r640_patch_sha256={R640_SHA256}")
    print(f"r641_patch_sha256={sha256(output)}")
    print(f"patch_bytes={output.stat().st_size}")
    print(f"output={output}")

if __name__ == "__main__":
    main()
