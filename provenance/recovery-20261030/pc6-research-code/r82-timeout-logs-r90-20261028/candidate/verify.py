#!/usr/bin/env python3
"""Reproduce CPU proof gates. Does not access GPUs, models or LiteLLM."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import locale
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def source_hashes(root: Path) -> dict[str, str]:
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (
            "src",
            "crates",
            "tests",
            "scripts",
            "adapters",
            "cuda",
            "fixtures",
            "third_party",
        )
        for p in sorted((root / folder).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    for name in (
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "requirements-cpu.txt",
    ):
        p = root / name
        if p.exists():
            hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in (
        "agents/routes.example.json",
        "agents/queue.example.json",
        "handoff/h1_bundle_run.sh",
        "handoff/hardware_canary.sh",
        "evidence/continuation_a105_all_backbone_headers/fetch_headers.py",
    ):
        path = root / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in json.loads((root / "fixtures/h1_reproduction_files.json").read_text()):
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("reproduction input escapes source")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument(
        "--run-id", default=datetime.now(timezone.utc).strftime("cpu-%Y%m%dT%H%M%SZ")
    )
    ap.add_argument(
        "--python-only", action="store_true", help="Partial check, never CPU_FULL_PASS"
    )
    args = ap.parse_args()
    root = args.root.resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_id):
        ap.error("invalid run id")
    out = (root / "evidence" / "runs" / args.run_id).resolve()
    if not out.is_relative_to(root):
        ap.error("output escapes project")
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root)
    target = Path(env.get("CARGO_TARGET_DIR", str(root / "target")))
    if not target.is_absolute():
        target = root / target
    binary = target / "release" / ("vf.exe" if os.name == "nt" else "vf")
    env["VF_BINARY"] = str(binary)
    steps = [
        (
            "python_agents",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/python", "-v"],
        ),
        (
            "agent_dry_run",
            [
                sys.executable,
                "scripts/vf_agents.py",
                "run",
                "--root",
                str(root),
                "--config",
                "agents/routes.example.json",
                "--queue",
                "agents/queue.example.json",
            ],
        ),
    ]
    if not args.python_only:
        # Integration tests discover VF_BINARY and may skip when it is absent.
        # Build both current CLI/worker schemas before Python, never test a stale
        # release binary (or silently skip native cases on a fresh checkout).
        steps.insert(
            0,
            (
                "build_native_for_python",
                ["cargo", "build", "--release", "--bins", "--locked"],
            ),
        )
        steps += [
            ("rustfmt", ["cargo", "fmt", "--all", "--", "--check"]),
            (
                "clippy",
                [
                    "cargo",
                    "clippy",
                    "--all-targets",
                    "--locked",
                    "--",
                    "-D",
                    "warnings",
                ],
            ),
            ("rust_debug", ["cargo", "test", "--all-targets", "--locked"]),
            (
                "rust_release",
                ["cargo", "test", "--release", "--all-targets", "--locked"],
            ),
            # --all-targets never runs doctests, the only tests that keep the fields of
            # CompressedArtifact private (compile_fail, A53-R20).
            ("rust_doctests", ["cargo", "test", "--doc", "--locked"]),
            (
                "c_abi_clippy",
                [
                    "cargo",
                    "clippy",
                    "--workspace",
                    "--all-targets",
                    "--locked",
                    "--features",
                    "c-abi-validation",
                    "--",
                    "-D",
                    "warnings",
                ],
            ),
            (
                "c_abi_rust_debug",
                [
                    "cargo",
                    "test",
                    "--workspace",
                    "--all-targets",
                    "--locked",
                    "--features",
                    "c-abi-validation",
                ],
            ),
            (
                "c_abi_rust_release",
                [
                    "cargo",
                    "test",
                    "--workspace",
                    "--release",
                    "--all-targets",
                    "--locked",
                    "--features",
                    "c-abi-validation",
                ],
            ),
            ("build_release", ["cargo", "build", "--release", "--locked"]),
            (
                "audit_regression_oracle",
                [
                    sys.executable,
                    "scripts/a13_crosscheck.py",
                    "--binary",
                    str(binary),
                    "--out",
                    str(out / "a13_crosscheck.json"),
                ],
            ),
            (
                "independent_oracle",
                [
                    sys.executable,
                    "scripts/crosscheck.py",
                    "--binary",
                    str(binary),
                    "--seeds",
                    "192",
                    "--out",
                    str(out / "crosscheck.json"),
                ],
            ),
            (
                "preweighted_fp32_oracle",
                [
                    sys.executable,
                    "scripts/preweighted_crosscheck.py",
                    "--binary",
                    str(binary),
                    "--out",
                    str(out / "preweighted_fp32_crosscheck.json"),
                ],
            ),
            (
                "mxfp4_tiny_independent_oracle",
                [sys.executable, "-B", "tests/reference/mxfp4_tiny_oracle.py"],
            ),
            (
                "mxfp4_persistent_session_oracle",
                [
                    sys.executable,
                    "scripts/mxfp4_session_crosscheck.py",
                    "--binary",
                    str(binary),
                    "--out",
                    str(out / "mxfp4_session_crosscheck.json"),
                ],
            ),
            (
                "independent_planner",
                [
                    sys.executable,
                    "scripts/planner_crosscheck.py",
                    "--binary",
                    str(binary),
                    "--out",
                    str(out / "planner_crosscheck.json"),
                ],
            ),
            (
                "quantized_adapter",
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests/adapter",
                    "-v",
                ],
            ),
            (
                "resident_buffers_oracle",
                [
                    sys.executable,
                    "scripts/resident_crosscheck.py",
                    "--binary",
                    str(binary),
                    "--out",
                    str(out / "resident_crosscheck.json"),
                ],
            ),
            (
                "real_checkpoint_codec",
                [
                    sys.executable,
                    "scripts/checkpoint_crosscheck.py",
                    "--binary",
                    str(binary),
                    "--out",
                    str(out / "checkpoint_crosscheck.json"),
                ],
            ),
        ]
    if not args.python_only:
        lint_files = [
            "adapters/skinny_moe.py",
            "adapters/ffi_provider.py",
            "adapters/persistent_gpu_owner.py",
            "tests/adapter/test_a147_ffi_provider.py",
            "tests/adapter/test_a149_admission_flags.py",
            "tests/adapter/package_fixture_utils.py",
            "tests/adapter/test_a150_h1_successor.py",
            "tests/adapter/test_a151_local_slot_bridge.py",
            "tests/adapter/test_a152_compact_rank_dispatch.py",
            "tests/adapter/test_a152_persistent_gpu_owner.py",
            "tests/adapter/test_skinny_adapter.py",
            "tests/adapter/test_metadata_collection.py",
            "tests/adapter/test_canary_integration.py",
            "scripts/checkpoint_crosscheck.py",
            "scripts/resident_crosscheck.py",
            "scripts/launch_crosscheck.py",
            "scripts/completion_crosscheck.py",
            "scripts/full_expert_crosscheck.py",
            "scripts/a13_crosscheck.py",
            "scripts/a14_crosscheck.py",
            "scripts/inspection_crosscheck.py",
            "scripts/a16_planner_crosscheck.py",
            "scripts/frontier_mutations.py",
            "scripts/rust_mutations.py",
            "scripts/oracle_sensitivity.py",
            "scripts/guard_schemata.py",
            "scripts/package_h1.py",
            "scripts/h1_preflight.py",
            "scripts/deepseek_v41_qpn2_topk6_canary.py",
            "tests/adapter/test_a16_numerical.py",
            "tests/adapter/test_a17_provider.py",
            "tests/adapter/test_h1_preflight.py",
            "tests/adapter/test_a16_h1_bundle.py",
            "tests/adapter/test_deepseek_v41_qpn2_topk6_canary.py",
            "tests/adapter/test_a14_boundaries.py",
            "tests/adapter/test_a43_preflight_refusals.py",
            "tests/adapter/test_a43_canary_refusals.py",
            "tests/adapter/test_a43_package_refusals.py",
            "tests/adapter/test_a43_adapter_refusals.py",
            "tests/python/test_a43_agent_refusals.py",
            "tests/adapter/test_a44_artifact_refusals.py",
            "tests/adapter/test_a44_full_expert_refusals.py",
            "tests/adapter/test_a45_handler_refusals.py",
            "tests/python/test_a45_agent_handlers.py",
            "tests/python/test_a53_agent_context_symlinks.py",
            "tests/python/test_a53_agent_gateway_idna.py",
            "tests/python/test_a105_backbone_fetch.py",
            "tests/python/test_a106_expert383_extractor.py",
            "tests/adapter/test_a46_defect_fixes.py",
            "tests/adapter/test_a47_launcher_refusals.py",
            "tests/adapter/test_a51_raise_reasons.py",
            "tests/adapter/test_a53_gate_doctests.py",
            "tests/adapter/test_a53_guard_verdict.py",
            "tests/adapter/test_a53_gate_mutant_verdict.py",
            "tests/adapter/test_a53_mutant_hygiene.py",
            "tests/adapter/test_a53_guard_timeout_cleanup.py",
            "tests/adapter/test_a53_mutation_failures.py",
            "tests/adapter/test_a53_cpu_proof_exit_codes.py",
            "scripts/deepseek_v41_crosscheck.py",
            "scripts/deepseek_v41_expert_crosscheck.py",
            "scripts/deepseek_v41_format_crosscheck.py",
            "scripts/deepseek_v41_shard_header_crosscheck.py",
            "scripts/deepseek_v41_mtp_header_crosscheck.py",
            "scripts/deepseek_v41_backbone_headers_crosscheck.py",
            "scripts/deepseek_v41_activation_crosscheck.py",
            "scripts/deepseek_v41_expert_forward_oracle.py",
            "evidence/continuation_a105_all_backbone_headers/fetch_headers.py",
            "scripts/verify.py",
            "scripts/preweighted_crosscheck.py",
            "tests/reference/fp32_add_oracle.py",
            "tests/reference/mxfp4_tiny_oracle.py",
            "tests/reference/preweighted_fault_peer.py",
            "scripts/mxfp4_session_crosscheck.py",
        ]
        for name in (
            "scripts/hardware_canary.py",
            "scripts/resident_artifact.py",
            "scripts/full_expert_canary.py",
            "tests/adapter/test_a20_full_canary.py",
            "tests/adapter/test_a19_artifact.py",
            "tests/adapter/test_a19_canary.py",
            "tests/adapter/test_hardware_contract.py",
        ):
            if (root / name).exists():
                lint_files.append(name)
        steps += [
            (
                "python_format",
                [sys.executable, "-m", "ruff", "format", "--check", *lint_files],
            ),
            (
                "python_lint",
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--select",
                    "E9,F",
                    *lint_files,
                ],
            ),
            (
                "ffi_cpu_build",
                [
                    "gcc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-fsanitize=address,undefined",
                    "tests/qpn2_moe_validate.c",
                    "cuda/qpn2_moe_validate.c",
                    "-o",
                    str(out / "validate_ffi"),
                ],
            ),
            ("ffi_cpu_validation", [str(out / "validate_ffi")]),
            (
                "full_expert_oracle",
                [
                    sys.executable,
                    "scripts/full_expert_crosscheck.py",
                    "--binary",
                    str(binary),
                    "--out",
                    str(out / "full_expert_crosscheck.json"),
                ],
            ),
            (
                "completion_output_oracle",
                [
                    sys.executable,
                    "scripts/completion_crosscheck.py",
                    "--binary",
                    str(binary),
                    "--out",
                    str(out / "completion_crosscheck.json"),
                ],
            ),
            (
                "ffi_validator_library",
                [
                    "gcc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-shared",
                    "-fPIC",
                    "cuda/qpn2_moe_validate.c",
                    "-o",
                    str(out / "libvf_validate.so"),
                ],
            ),
            (
                "launch_metadata_oracle",
                [
                    sys.executable,
                    "scripts/launch_crosscheck.py",
                    "--binary",
                    str(binary),
                    "--validator",
                    str(out / "libvf_validate.so"),
                    "--out",
                    str(out / "launch_crosscheck.json"),
                ],
            ),
        ]
    if not args.python_only:
        steps.append(
            (
                "sm70_scale_domain_oracle",
                [
                    sys.executable,
                    "scripts/a14_crosscheck.py",
                    "--binary",
                    str(binary),
                ],
            )
        )
    if not args.python_only:
        steps.append(
            (
                "inspection_output_oracle",
                [
                    sys.executable,
                    "scripts/inspection_crosscheck.py",
                    "--binary",
                    str(binary),
                ],
            )
        )
    if not args.python_only:
        steps.extend(
            [
                (
                    "a16_weighted_score_oracle",
                    [
                        sys.executable,
                        "scripts/a16_planner_crosscheck.py",
                        "--binary",
                        str(binary),
                        "--out",
                        str(out / "a16_planner_crosscheck.json"),
                    ],
                ),
                (
                    "h1_bundle_launcher_syntax",
                    ["bash", "-n", "handoff/h1_bundle_run.sh"],
                ),
                (
                    "h1_canary_launcher_syntax",
                    ["bash", "-n", "handoff/hardware_canary.sh"],
                ),
                ("sm70_build_syntax", ["bash", "-n", "cuda/build_sm70.sh"]),
                (
                    "deepseek_v41_ops_oracle",
                    [
                        sys.executable,
                        "scripts/deepseek_v41_crosscheck.py",
                        "--binary",
                        str(binary),
                        "--out",
                        str(out / "deepseek_v41_crosscheck.json"),
                    ],
                ),
                (
                    "deepseek_v41_real_expert_oracle",
                    [
                        sys.executable,
                        "scripts/deepseek_v41_expert_crosscheck.py",
                        "--out",
                        str(out / "deepseek_v41_expert_crosscheck.json"),
                    ],
                ),
                (
                    "deepseek_v41_official_format",
                    [
                        sys.executable,
                        "scripts/deepseek_v41_format_crosscheck.py",
                        "--out",
                        str(out / "deepseek_v41_format_crosscheck.json"),
                    ],
                ),
                (
                    "deepseek_v41_qpn2_conversion_oracle",
                    [
                        sys.executable,
                        "scripts/deepseek_v41_qpn2_conversion_oracle.py",
                        "--binary",
                        str(binary),
                        "--out",
                        str(out / "deepseek_v41_qpn2_conversion_oracle.json"),
                    ],
                ),
                (
                    "deepseek_v41_official_shard_header",
                    [
                        sys.executable,
                        "scripts/deepseek_v41_shard_header_crosscheck.py",
                        "--out",
                        str(out / "deepseek_v41_shard_header_crosscheck.json"),
                    ],
                ),
                (
                    "deepseek_v41_official_mtp_headers",
                    [
                        sys.executable,
                        "scripts/deepseek_v41_mtp_header_crosscheck.py",
                        "--out",
                        str(out / "deepseek_v41_mtp_header_crosscheck.json"),
                    ],
                ),
                (
                    "deepseek_v41_official_backbone_headers",
                    [
                        sys.executable,
                        "scripts/deepseek_v41_backbone_headers_crosscheck.py",
                        "--out",
                        str(out / "deepseek_v41_backbone_headers_crosscheck.json"),
                    ],
                ),
                (
                    "deepseek_v41_activation_quantizer_oracle",
                    [
                        sys.executable,
                        "scripts/deepseek_v41_activation_crosscheck.py",
                        "--binary",
                        str(binary),
                        "--out",
                        str(out / "deepseek_v41_activation_crosscheck.json"),
                    ],
                ),
                (
                    "deepseek_v41_compressed_expert_forward_oracle",
                    [
                        sys.executable,
                        "scripts/deepseek_v41_expert_forward_oracle.py",
                        "--out",
                        str(out / "deepseek_v41_expert_forward_oracle.json"),
                    ],
                ),
                (
                    "a106_official_expert383_forward_oracle",
                    [
                        sys.executable,
                        "scripts/deepseek_v41_expert_forward_oracle.py",
                        "--artifact",
                        "fixtures/deepseek_v41_layer0_expert383_314113f8.safetensors",
                        "--expected-sha256",
                        "9f80cd6bd1244876fb9df5c3a8571ea2c44a981767bbff039e05bab3380e1ab3",
                        "--expert-id",
                        "383",
                        "--golden",
                        "fixtures/deepseek_v41_expert383_forward_bf16_314113f8.json",
                        "--out",
                        str(out / "a106_expert383_forward_oracle.json"),
                    ],
                ),
            ]
        )
    hashes = source_hashes(root)
    results = []
    for label, command in steps:
        start = time.monotonic()
        try:
            run = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
            )
            rc, text = run.returncode, run.stdout
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode(
                    locale.getpreferredencoding(False), errors="backslashreplace"
                )
            if partial and not partial.endswith("\n"):
                partial += "\n"
            rc, text = -1, f"{partial}{type(exc).__name__}: {exc}"
        except OSError as exc:
            rc, text = -1, f"{type(exc).__name__}: {exc}"
        (out / f"{label}.log").write_text(text)
        results.append(
            {
                "step": label,
                "argv": command,
                "exit_code": rc,
                "elapsed_seconds": time.monotonic() - start,
                "log": f"{label}.log",
            }
        )
        print(f"{label}: exit={rc}", flush=True)
        if rc:
            break
    good = len(results) == len(steps) and all(x["exit_code"] == 0 for x in results)
    sources_unchanged = source_hashes(root) == hashes
    good = good and sources_unchanged
    binary_receipt = {}
    if good and not args.python_only:
        binary_receipt = {
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "binary_bytes": binary.stat().st_size,
        }
    summary = {
        "schema_version": 1,
        "status": ("PYTHON_ONLY_PASS" if args.python_only else "CPU_FULL_PASS")
        if good
        else "FAIL",
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashes,
        "sources_unchanged": sources_unchanged,
        **binary_receipt,
        "steps": results,
        "cuda_executed": False,
        "rdma_tested": False,
        "live_llm_calls": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(str(out / "summary.json"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
