"""Reuse the R52 pinned one-job recipe, building only the private fresh-ID hardware harness."""

import fcntl, hashlib, json, os, shutil, signal, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path(
    "/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c"
)
RUST = BASE / ".tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def available():
    return (
        int(
            next(
                x.split()[1]
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            )
        )
        * 1024
    )


def main():
    assert __debug__
    out = ROOT / "build"
    out.mkdir(exist_ok=False)
    (ROOT / "tmp").mkdir(exist_ok=False)
    inputs = json.loads((ROOT / "PAYLOAD.json").read_text())
    assert all(sha(ROOT / n) == h for n, h in inputs.items())
    config = json.loads((ROOT / "BUILD_PLAN.json").read_text())
    assert sha(ROOT / "native/libvf_r13.so") == config["native_sha256"]
    lock = (ROOT / "compiler.lease").open("x+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = {
        "status": "BUILD_STARTED_NO_GPU",
        "pid": os.getpid(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "source_hashes": inputs,
        "steps": [],
        "gpu_executed": False,
        "automatic_retry": False,
    }

    def save():
        (out / "operation.json").write_text(json.dumps(report, indent=2) + "\n")

    env = dict(
        os.environ,
        PATH=str(RUST) + ":" + os.environ["PATH"],
        RUSTC=str(RUST / "rustc"),
        CARGO_HOME=str(BASE / ".tools/cargo"),
        RUSTUP_HOME=str(BASE / ".tools/rustup"),
        CARGO_TARGET_DIR=str(ROOT / "target"),
        CARGO_BUILD_JOBS="1",
        CARGO_INCREMENTAL="0",
        CARGO_NET_OFFLINE="true",
        RUST_TEST_THREADS="1",
        TMPDIR=str(ROOT / "tmp"),
        VF_MXFP4_NATIVE_DIR=str(ROOT / "native"),
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH=str(ROOT / "native"),
    )

    def run(name, args):
        assert available() > 3 * 1024**3 and shutil.disk_usage(ROOT).free > 4 * 1024**3
        report["active_step"] = name
        save()
        started = time.monotonic_ns()
        argv = ["/usr/bin/time", "-v", "-o", str(out / (name + ".time")), *args]
        with (out / (name + ".log")).open("x") as f:
            p = subprocess.Popen(
                argv,
                cwd=ROOT / "source",
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            report["active_process"] = {"pid": p.pid, "argv": argv}
            save()
            reason = None
            while p.poll() is None:
                if time.monotonic_ns() - started > 360 * 10**9:
                    reason = "CPU_DEADLINE"
                    break
                if available() < 1536 * 1024**2:
                    reason = "CPU_RESOURCE_FLOOR"
                    break
                time.sleep(0.25)
            if reason:
                os.killpg(p.pid, signal.SIGKILL)
            code = p.wait(timeout=5)
        report["steps"].append(
            {
                "name": name,
                "pid": p.pid,
                "argv": argv,
                "exit_code": code,
                "start_ns": started,
                "end_ns": time.monotonic_ns(),
                "process_reconciled": p.poll() is not None,
                "resource_stop": reason,
            }
        )
        report.pop("active_process", None)
        save()
        assert code == 0 and reason is None, (name, code, reason)

    try:
        cargo = [str(RUST / "cargo")]
        m = ["--manifest-path", "gpu-worker/Cargo.toml"]
        run("version", [str(RUST / "rustc"), "-vV"])
        run("fmt", cargo + ["fmt", *m, "--", "--check"])
        run(
            "clippy",
            cargo
            + [
                "clippy",
                "--offline",
                "--locked",
                *m,
                "--test",
                "hardware_r53",
                "--",
                "-D",
                "warnings",
            ],
        )
        for mode in ["debug", "release"]:
            flags = ["--release"] if mode == "release" else []
            label = "hardware_" + mode
            run(
                label,
                cargo
                + [
                    "test",
                    "--offline",
                    "--locked",
                    *m,
                    "--test",
                    "hardware_r53",
                    "--no-run",
                    "--message-format=json",
                    *flags,
                ],
            )
            records = [
                json.loads(line)
                for line in (out / (label + ".log")).read_text().splitlines()
                if line.startswith("{")
            ]
            paths = [
                x["executable"]
                for x in records
                if x.get("reason") == "compiler-artifact"
                and x.get("target", {}).get("name") == "hardware_r53"
                and x.get("executable")
            ]
            assert len(paths) == 1
            shutil.copyfile(paths[0], out / label)
            (out / label).chmod(0o700)
        for name, digest in config["worker_sha256"].items():
            origin = BASE / ".runs/r52-sm120-rust-worker-20261026/build" / name
            assert sha(origin) == digest
            shutil.copyfile(origin, out / name)
            (out / name).chmod(0o700)
        run("loader", ["ldd", str(out / "hardware_release")])
        assert "not found" not in (out / "loader.log").read_text()
        assert all(sha(ROOT / n) == h for n, h in inputs.items())
        report.update(
            status="FRESH_NAMESPACE_X86_HARNESS_COMPILED_GPU_NOT_EXECUTED",
            binaries_sha256={
                n: sha(out / n)
                for n in [
                    "hardware_debug",
                    "hardware_release",
                    "worker_debug",
                    "worker_release",
                ]
            },
        )
    except Exception as e:
        report.update(status="BUILD_FAILED_NO_GPU", error=repr(e))
    finally:
        report.pop("active_step", None)
        report["end_ns"] = time.monotonic_ns()
        save()
        lock.close()
    print(json.dumps({"status": report["status"], "error": report.get("error")}))
    raise SystemExit(
        0
        if report["status"] == "FRESH_NAMESPACE_X86_HARNESS_COMPILED_GPU_NOT_EXECUTED"
        else 1
    )


if __name__ == "__main__":
    main()
