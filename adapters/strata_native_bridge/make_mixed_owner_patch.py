#!/usr/bin/env python3
"""Generate the incremental R640 mixed-owner dispatch patch without modifying Strata."""
import difflib
import hashlib
import pathlib
import shutil
import subprocess
import sys
import tempfile

REV = "fb58e0dbc8399662c0e47c76578c6e878b14f6cf"
BASE_PATCH_SHA256 = "2e86bbcb65ad6763cd9c47ec02a85da7ce3147ee50633e974ec5815f2f6bd10a"
FILES = [
    "include/strata/core/verify.hpp",
    "include/strata/core/expert_source.hpp",
    "src/core/expert_source.cpp",
    "src/core/verify.cpp",
    "src/program/generate.cpp",
]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def change(data, old, new, label):
    count = data.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return data.replace(old, new, 1)

def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: make_mixed_owner_patch.py PINNED_STRATA_ROOT CPU_HOOK_PATCH OUTPUT_PATCH")
    root = pathlib.Path(sys.argv[1]).resolve()
    base_patch = pathlib.Path(sys.argv[2]).resolve()
    output = pathlib.Path(sys.argv[3]).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REV:
        raise SystemExit(f"wrong Strata revision: {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--", *FILES], text=True
    ).strip()
    if dirty:
        raise SystemExit("pinned Strata donor files are modified")
    if sha256(base_patch) != BASE_PATCH_SHA256:
        raise SystemExit("CPU-hook base patch hash mismatch")

    original = {p: (root / p).read_text() for p in FILES}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r640-strata-", dir=str(output.parent)) as td:
        stage = pathlib.Path(td)
        for p in FILES:
            target = stage / p
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(original[p])
        subprocess.run(
            ["patch", "--silent", "-p1", "-d", str(stage), "-i", str(base_patch)],
            check=True,
        )
        base = {p: (stage / p).read_text() for p in FILES}
        final = dict(base)

        header = FILES[1]
        final[header] = change(
            final[header],
            "    void* vf_cpu_context = nullptr;\n",
            "    void* vf_cpu_context = nullptr;\n"
            "    // Optional mixed-owner callback. It handles only final kind -1 rows.\n"
            "    bool (*vf_cpu_miss_pool)(void*, const float*, const int32_t*, const int32_t*, "
            "int64_t, int64_t, float*, int64_t) = nullptr;\n"
            "    void* vf_cpu_miss_context = nullptr;\n",
            "ExpertDispatch mixed callback fields",
        )

        source = FILES[2]
        anchor = "    static const bool dec_batch = []"
        block = """    // The owner map is final only after the peer and remote owners have claimed their rows.
    // Keep Strata's -1..2 kinds intact for VFB3; remember handled CPU rows separately.
    bool vf_cpu_handled[kMaxWindowEntries] = {};
    if (d.vf_cpu_miss_pool != nullptr) {
        bool any_vf_cpu_miss = false;
        for (int64_t i = 0; i < n; ++i)
            if (kind[i] < 0) { any_vf_cpu_miss = true; break; }
        if (any_vf_cpu_miss &&
            !d.vf_cpu_miss_pool(d.vf_cpu_miss_context, x_f, ids, kind, n_tok, k, out, d.layers)) {
            // The verifier will refuse this window; drain already-started external owners first.
            bool external_drained = true;
            if (d.remote_count > 0) {
                static thread_local std::string drain_error;
                for (int r = 0; r < d.remote_count; ++r)
                    if (!d.remote[r]->finish(out, drain_error)) external_drained = false;
            }
            if (d.peer != nullptr) {
                std::string drain_error;
                if (!d.peer->finish(out, drain_error)) external_drained = false;
            }
            d.failed = true;
            d.fail = external_drained
                ? "VoltaFabric mixed CPU-miss callback failed after external owners drained"
                : "VoltaFabric mixed CPU-miss callback failed and an external owner did not drain";
            d.fail_layer = d.layers;
            return;
        }
        for (int64_t i = 0; i < n; ++i)
            if (kind[i] < 0) vf_cpu_handled[i] = true;
    }
"""
        final[source] = change(final[source], anchor, block + anchor, "post-final-owner-map callback")

        final[source] = change(
            final[source],
            "    for (int64_t i = 0; i < n; ++i)\n        if (kind[i] < 0) { any_cpu = true; break; }",
            "    for (int64_t i = 0; i < n; ++i)\n"
            "        if (kind[i] < 0 && !vf_cpu_handled[i]) { any_cpu = true; break; }",
            "remaining local CPU detection",
        )
        final[source] = change(
            final[source],
            "        for (int64_t t = 0; t < n_tok; ++t) {\n"
            "            if (ep && std::all_of(kind + t * k, kind + (t + 1) * k, [](int32_t v) { return v >= 0; })) continue;",
            "        for (int64_t t = 0; t < n_tok; ++t) {\n"
            "            if (d.vf_cpu_miss_pool != nullptr) {\n"
            "                bool token_has_local_cpu = false;\n"
            "                for (int64_t j = 0; j < k; ++j) {\n"
            "                    const int64_t i = t * k + j;\n"
            "                    if (kind[i] < 0 && !vf_cpu_handled[i]) { token_has_local_cpu = true; break; }\n"
            "                }\n"
            "                if (!token_has_local_cpu) continue;\n"
            "            }\n"
            "            if (ep && std::all_of(kind + t * k, kind + (t + 1) * k, [](int32_t v) { return v >= 0; })) continue;",
            "activation quantization row filter",
        )
        final[source] = change(
            final[source],
            "            if (kind[i] < 0 && ids[i] >= 0 && ids[i] < d.n_expert &&\n"
            "                std::find(miss.begin(), miss.end(), (int64_t) ids[i]) == miss.end())",
            "            if (kind[i] < 0 && !vf_cpu_handled[i] && ids[i] >= 0 && ids[i] < d.n_expert &&\n"
            "                std::find(miss.begin(), miss.end(), (int64_t) ids[i]) == miss.end())",
            "CPU prefetch filter",
        )
        final[source] = change(
            final[source],
            "                return;\n"
            "            }\n"
            "            if (kind[i] >= 0) {             // CUDA0, PCIe, or a remote/peer result staged into this row below",
            "                return;\n"
            "            }\n"
            "            if (vf_cpu_handled[i]) continue;\n"
            "            if (kind[i] >= 0) {             // CUDA0, PCIe, or a remote/peer result staged into this row below",
            "skip already handled row",
        )

        patch = "".join(
            "".join(difflib.unified_diff(
                base[p].splitlines(keepends=True),
                final[p].splitlines(keepends=True),
                fromfile="a/" + p,
                tofile="b/" + p,
                n=3,
            ))
            for p in FILES if base[p] != final[p]
        )
        if not patch:
            raise SystemExit("no donor changes were generated")
        output.write_text(patch)
        print(f"revision={REV}")
        print(f"base_patch_sha256={BASE_PATCH_SHA256}")
        print(f"incremental_patch_sha256={sha256(output)}")
        print(f"changed_files={sum(base[p] != final[p] for p in FILES)}")
        print(f"patch_bytes={output.stat().st_size}")
        print(f"output={output}")

if __name__ == "__main__":
    main()
