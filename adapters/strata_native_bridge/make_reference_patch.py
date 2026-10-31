#!/usr/bin/env python3
"""Build a narrow, opt-in candidate patch for the pinned Strata CUDA source.
No checkout is modified. This is not a GPU qualification or deployment tool.
"""
import difflib
import pathlib
import subprocess
import sys

REV = 'fb58e0dbc8399662c0e47c76578c6e878b14f6cf'
FILES = ['include/strata/core/verify.hpp', 'include/strata/core/expert_source.hpp', 'src/core/expert_source.cpp', 'src/core/verify.cpp', 'src/program/generate.cpp']
root = pathlib.Path(sys.argv[1]).resolve()
revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
if revision != REV or subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--', *FILES], text=True).strip():
    raise SystemExit('unexpected revision or modified donor source')
original = {p: (root / p).read_text() for p in FILES}
modified = dict(original)
def change(path, old, new):
    data = modified[path]
    if data.count(old) != 1:
        raise SystemExit(f'expected unique donor site in {path}, found {data.count(old)}')
    modified[path] = data.replace(old, new, 1)

h = FILES[0]
change(h, 'using PoolMultiFn = void (*)(void* user,', 'using PoolMultiFn = bool (*)(void* user,')
h = FILES[1]
change(h, 'struct ExpertDispatch {\n', '''struct ExpertDispatch {
    // VoltaFabric opt-in. The hook is admitted only for a CPU-only expert plan.
    bool (*vf_cpu_pool)(void*, const float*, const int32_t*, int64_t, int64_t, float*, int64_t) = nullptr;
    void* vf_cpu_context = nullptr;
''')
s = FILES[2]
old = '''    if (d.lookahead != nullptr) d.lookahead->submit(d.layers, x_f, n_tok, d.host_res);   // CS-T: warm layer + 1
'''
new = '''    if (d.vf_cpu_pool != nullptr) {
        // This first candidate refuses all other expert owners before launching them.
        if (!x_f || !ids || !out || k < 1 || k > 64 || n_tok < 1 ||
            n_tok > kMaxWindowEntries / k ||
            d.layers < 0 || d.layers >= 48 || d.n_expert != 512 || d.fs != nullptr ||
            d.peer != nullptr || d.remote_count != 0 || d.lookahead != nullptr || d.pcie_num != 0) {
            d.failed = true; d.fail = "VoltaFabric requires a CPU-only expert plan"; d.fail_layer = d.layers; return;
        }
        for (int64_t i = 0; i < n_tok * k; ++i) if (ids[i] < 0 || ids[i] >= d.n_expert) {
            d.failed = true; d.fail = "VoltaFabric routed expert out of range"; d.fail_layer = d.layers; return;
        }
        for (int64_t i = 0; d.host_res != nullptr && i < n_tok * k; ++i)
            if (d.host_res[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] >= 0) {
                d.failed = true; d.fail = "VoltaFabric routed expert has a GPU owner"; d.fail_layer = d.layers; return;
            }
        if (!d.vf_cpu_pool(d.vf_cpu_context, x_f, ids, n_tok, k, out, d.layers)) {
            d.failed = true; d.fail = "VoltaFabric CPU pool failed"; d.fail_layer = d.layers; return;
        }
        ++d.layers; d.experts += n_tok * k; return;
    }
''' + old
change(s,old,new)
s=FILES[3]
old = '''        while (cudaStreamQuery(s) == cudaErrorNotReady) {
            if (ms_since(t0) > timeout_ms) {
'''
new = '''        for (;;) {
            const cudaError_t query = cudaStreamQuery(s);
            if (query == cudaSuccess) break;
            if (query != cudaErrorNotReady) return false;
            if (ms_since(t0) > timeout_ms) {
'''
change(s,old,new)
old = '''        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        if (remote_opt_) remote_opt_->end();
'''
new = '''        const bool pool_ok = pool == nullptr ||
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        if (remote_opt_) remote_opt_->end();
        if (!pool_ok) { err = "verify: expert pool failed" + released_note(release_gpu_waits(5000)); return false; }
'''
change(s,old,new)
old = '''        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, l);
'''
new = '''        if (pool != nullptr && !pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, l)) {
            err = "verify batch: expert pool failed" + released_note(release_gpu_waits(5000)); return false;
        }
'''
change(s,old,new)
old = '''        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, lb_ + b_k_);
'''
new = '''        if (pool != nullptr && !pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, lb_ + b_k_)) {
            err = "verify batch: expert pool failed" + released_note(release_gpu_waits(5000)); b_running_ = false; return -1;
        }
'''
change(s,old,new)
old = '''        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        if (g_trace) trace_ev'''
new = '''        if (pool != nullptr && !pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l)) {
            err = "verify: expert pool failed" + released_note(release_gpu_waits(5000)); fl_active_ = false; return -1;
        }
        if (g_trace) trace_ev'''
change(s,old,new)
s=FILES[4]
change(s,'void drive_pool_multi(void* user,','bool drive_pool_multi(void* user,')
change(s, '''    }
}

/// Layer split: every verify stage shares one Drive''','''    }
    return !t->d.failed;
}

/// Layer split: every verify stage shares one Drive''')
change(s,'void drive_pool_split(void* user,','bool drive_pool_split(void* user,')
change(s,'    drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);\n','    return drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);\n')
patch = ''.join(''.join(difflib.unified_diff(original[p].splitlines(keepends=True), modified[p].splitlines(keepends=True), fromfile='a/'+p, tofile='b/'+p)) for p in FILES)
sys.stdout.write(patch)
