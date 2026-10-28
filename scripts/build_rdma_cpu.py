#!/usr/bin/env python3
"""Build the explicitly selected, bounded host-RDMA CPU bridge. No NIC/GPU calls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def build(output, *, cc="cc", sanitized=False):
    # Do not resolve the leaf: an existing directory or symlink is an error,
    # including a previous failed/uncertain invocation. No implicit rebuild.
    output = Path(os.path.abspath(output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "output already exists; inspect its receipt before a new build"
        )
    source = Path(__file__).with_name("rdma_cpu_bridge.c")
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 131072:
        raise ValueError("missing, symlinked or oversized bridge source")
    source_bytes = source.read_bytes()
    compiler = shutil.which(cc)
    if not compiler:
        raise ValueError("C compiler not found")
    output.mkdir(mode=0o700)
    frozen = output / "rdma_cpu_bridge.c"
    frozen.write_bytes(source_bytes)
    (output / "tmp").mkdir(mode=0o700)
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror"]
    flags += (
        ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        if sanitized
        else ["-O2", "-fstack-protector-strong"]
    )
    binary = output / "vf-rsocket-cpu"
    partial = output / "vf-rsocket-cpu.partial"
    argv = [compiler, *flags, str(frozen), "-o", str(partial), "-ldl"]
    receipt = output / "build.json"
    report = dict(
        schema_version=1,
        status="BUILD_RUNNING_NO_REPLAY",
        scope="EXPERIMENTAL_BOUNDED_HOST_RDMA_CPU",
        automatic_retry=False,
        source=frozen.name,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        binary=binary.name,
        machine=platform.machine(),
        libc=platform.libc_ver(),
        sanitized=sanitized,
        argv=argv,
        compiler_sha256=digest(Path(compiler)),
        runtime_library="librdmacm.so.1",
        host_buffer_bytes_per_direction=65536,
        requested_local_wait_ns=100000,
        native_alarm_seconds=12,
        cpu_tested=False,
        gpu_used=False,
    )
    save(receipt, report)
    start = time.monotonic()
    proc = None
    try:
        proc = subprocess.Popen(
            argv,
            cwd=output,
            env={**os.environ, "TMPDIR": str(output / "tmp")},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        report["compiler_pid"] = proc.pid
        save(receipt, report)
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate(timeout=5)
            report.update(
                status="COMPILE_TIMEOUT_TERMINAL_NO_REPLAY", exit_code=proc.returncode
            )
            raise RuntimeError("compiler timeout; inspect receipt, no replay")
        report.update(
            exit_code=proc.returncode,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )
        if proc.returncode != 0:
            report["status"] = "COMPILE_FAILED"
            raise RuntimeError("C compilation failed; inspect build.json")
        partial.replace(binary)
        report.update(status="COMPILED_NATIVE_ONLY", binary_sha256=digest(binary))
        return report
    except BaseException as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate(timeout=5)
        if report["status"] == "BUILD_RUNNING_NO_REPLAY":
            report["status"] = "BUILD_FAILED_NO_REPLAY"
        raise
    finally:
        report["seconds"] = time.monotonic() - start
        save(receipt, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New output directory; never overwritten",
    )
    parser.add_argument(
        "--cc", default="cc", help="Explicit C compiler executable, no shell command"
    )
    parser.add_argument("--sanitized", action="store_true")
    args = parser.parse_args()
    try:
        report = build(args.out, cc=args.cc, sanitized=args.sanitized)
    except Exception as exc:
        print(
            json.dumps(dict(status="BUILD_REFUSED_OR_FAILED", error=str(exc))),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "machine",
                    "binary_sha256",
                    "source_sha256",
                    "sanitized",
                ]
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
