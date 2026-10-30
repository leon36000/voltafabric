#!/usr/bin/env python3
"""Apply the opt-in candidate to a small private overlay and syntax-check it.
AArch64 intrinsics stubs are for parsing only; no GPU or donor runtime is executed.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

root = pathlib.Path(sys.argv[1]).resolve()
ggml = pathlib.Path(sys.argv[2]).resolve()
output_root = pathlib.Path(sys.argv[3]).resolve()
cuda = pathlib.Path(sys.argv[4]).resolve()
script_dir = pathlib.Path(__file__).resolve().parent
files = ['include/strata/core/verify.hpp', 'include/strata/core/expert_source.hpp', 'src/core/expert_source.cpp', 'src/core/verify.cpp', 'src/program/generate.cpp']
expected = subprocess.check_output([sys.executable, str(script_dir / 'make_reference_patch.py'), str(root)])
patch = script_dir / 'strata-fb58-cpu-hook.patch'
if expected != patch.read_bytes():
    raise SystemExit('candidate patch differs from deterministic donor regeneration')
subprocess.run(['git', '-C', str(root), 'apply', '--check', str(patch)], check=True)
overlay = pathlib.Path(tempfile.mkdtemp(prefix='r390-patch-', dir=output_root))
for name in files:
    dest = overlay / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, dest)
subprocess.run(['patch', '-p1', '--batch', '--forward', '-i', str(patch)], cwd=overlay, check=True, stdout=subprocess.DEVNULL)
(overlay / 'include/immintrin.h').write_text('#pragma once\nstatic inline void _mm_pause() {}\nstatic inline void _mm_sfence() {}\n')
(overlay / 'check.cpp').write_text('''#include "strata/core/verify.hpp"
#include "pool_hook.hpp"
#include <type_traits>
static_assert(std::is_convertible_v<decltype(&vf::strata_bridge::pool_cpu_only), strata::core::PoolMultiFn>);
static_assert(std::is_convertible_v<decltype(&vf::strata_bridge::pool_cpu_by_layer), strata::core::PoolMultiFn>);
int main() { strata::core::ExpertDispatch d; d.vf_cpu_pool = &vf::strata_bridge::pool_cpu_only; return d.vf_cpu_pool ? 0 : 1; }
''')
includes = ['-I',str(overlay/'include'), '-I',str(root/'include'), '-I',str(script_dir), '-I',str(cuda), '-I',str(ggml/'ggml/include')]
for source in ['check.cpp', 'src/core/expert_source.cpp', 'src/core/verify.cpp', 'src/program/generate.cpp']:
    subprocess.run(['c++','-std=c++20','-DSTRATA_VERSION="syntax-test"','-fsyntax-only',*includes,str(overlay/source)], check=True)
if len(sys.argv) > 5:
    # Build only the actual, patched function prefix containing the opt-in path.
    # The unmodified donor CPU/GPU fallback is intentionally excluded, so this
    # executable cannot claim to test a complete donor or its CUDA verifier.
    source = (overlay / 'src/core/expert_source.cpp').read_text()
    start = source.index('void expert_pool_dispatch_multi(')
    end = source.index('    if (d.lookahead != nullptr)', start)
    branch = source[start:end]
    if branch.count('if (d.vf_cpu_pool != nullptr)') != 1:
        raise SystemExit('unexpected opt-in branch')
    extracted = overlay / 'donor_cpu_branch.cpp'
    extracted.write_text('#include "strata/core/expert_source.hpp"\nnamespace strata::core {\nnamespace { constexpr int64_t kMaxWindowEntries = 128; }\n' + branch + '\n    d.failed = true; d.fail = "test-only ordinary path unavailable";\n}\n}\n')
    optimization = sys.argv[6] if len(sys.argv) > 6 else '2'
    if optimization not in ('0', '2'):
        raise SystemExit('optimization must be 0 or 2')
    mode = sys.argv[7] if len(sys.argv) > 7 else 'extracted'
    if mode not in ('extracted', 'full', 'full-fleet'):
        raise SystemExit('mode must be extracted, full or full-fleet')
    # Full mode links the complete patched donor source with every unavailable
    # GPU, peer and fallback dependency replaced by an aborting test-only stub.
    # Reaching any such dependency fails the test, never launches CUDA.
    donor_sources = [str(extracted)] if mode == 'extracted' else [str(overlay/'src/core/expert_source.cpp'), str(script_dir/'donor_forbidden_stubs.cpp')]
    defines = ['-DVF_FULL_DONOR_TEST'] if mode == 'full' else []
    test_source = 'donor_fleet_live_test.cpp' if mode == 'full-fleet' else 'donor_branch_live_test.cpp'
    subprocess.run(['c++','-std=c++20','-O'+optimization,'-pthread','-ffunction-sections','-fdata-sections','-Wl,--gc-sections',*defines,*includes, *donor_sources, str(script_dir/'pipe_client.cpp'), str(script_dir/'pool_hook.cpp'), str(script_dir/test_source), '-o',str(pathlib.Path(sys.argv[5]).resolve())], check=True)
print('candidate applies and four translation units pass syntax checking:', overlay)
