"""Pinned one-job native x86 build of unchanged Rust code, no visible CUDA device."""

import fcntl, hashlib, json, os, platform, shutil, signal, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path(
    "/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c"
)
RUST = BASE / ".tools/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin"
EXPECTED = "f856c9bdf5c4d021203b2b7ce5b1f393058370e3f5d2d0f05072fb067e739213"


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
    assert __debug__ and platform.machine() == "x86_64"
    out = ROOT / "build"
    out.mkdir(exist_ok=False)
    (ROOT / "tmp").mkdir(exist_ok=False)
    manifest = json.loads((ROOT / "PAYLOAD.json").read_text())
    assert all(sha(ROOT / n) == h for n, h in manifest.items())
    assert sha(ROOT / "native/libvf_r13.so") == EXPECTED
    lease = (ROOT / "compiler.lease").open("x+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
    report = {
        "status": "BUILD_STARTED_NO_GPU",
        "pid": os.getpid(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "start_ns": time.monotonic_ns(),
        "gpu_executed": False,
        "automatic_retry": False,
        "steps": [],
        "target": env["CARGO_TARGET_DIR"],
        "source_manifest": manifest,
        "native_sha256": EXPECTED,
        "resource_policy": {
            "jobs": 1,
            "before_step_min_available_bytes": 3 * 1024**3,
            "abort_below_available_bytes": 1536 * 1024**2,
            "max_step_seconds": 360,
        },
    }

    def save():
        (out / "operation.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, args):
        assert (
            available() > 3 * 1024**3 and shutil.disk_usage(ROOT).free > 4 * 1024**3
        ), "CPU build resource floor"
        report["active_step"] = name
        save()
        started = time.monotonic_ns()
        timed = ["/usr/bin/time", "-v", "-o", str(out / (name + ".time")), *args]
        with (out / (name + ".log")).open("x") as log:
            p = subprocess.Popen(
                timed,
                cwd=ROOT / "source",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            step = {"name": name, "argv": timed, "pid": p.pid, "start_ns": started}
            report["active_process"] = step
            save()
            reason = None
            while p.poll() is None:
                if time.monotonic_ns() - started > 360 * 10**9:
                    reason = "CPU_BUILD_DEADLINE"
                    break
                if available() < 1536 * 1024**2:
                    reason = "CPU_RESOURCE_FLOOR"
                    break
                time.sleep(0.25)
            if reason:
                os.killpg(p.pid, signal.SIGKILL)
            code = p.wait(timeout=5)
        step.update(
            exit_code=code,
            end_ns=time.monotonic_ns(),
            resource_stop=reason,
            process_reconciled=p.poll() is not None,
        )
        report["steps"].append(step)
        report.pop("active_process", None)
        save()
        print(json.dumps({"step": name, "exit_code": code}), flush=True)
        assert code == 0 and reason is None, (name, code, reason)

    try:
        cargo = [str(RUST / "cargo")]
        m = ["--manifest-path", "gpu-worker/Cargo.toml"]
        locked = ["--offline", "--locked"]
        run("version", [str(RUST / "rustc"), "-vV"])
        run("fmt", cargo + ["fmt", *m, "--", "--check"])
        run(
            "clippy",
            cargo + ["clippy", *locked, *m, "--all-targets", "--", "-D", "warnings"],
        )
        for mode in ["debug", "release"]:
            flags = ["--release"] if mode == "release" else []
            run("worker_" + mode, cargo + ["build", *locked, *m, *flags])
            shutil.copyfile(
                ROOT / "target" / mode / "vf-mxfp4-gpu-worker", out / ("worker_" + mode)
            )
            (out / ("worker_" + mode)).chmod(0o700)
            for test in ["hardware", "truncation"]:
                label = test + "_" + mode
                run(
                    label,
                    cargo
                    + [
                        "test",
                        *locked,
                        *m,
                        "--test",
                        test,
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
                    and x.get("target", {}).get("name") == test
                    and x.get("executable")
                ]
                assert len(paths) == 1
                shutil.copyfile(paths[0], out / label)
                (out / label).chmod(0o700)
        run(
            "coordinator",
            cargo
            + [
                "build",
                *locked,
                "--release",
                "--bin",
                "vf",
                "--bin",
                "vf-rank-session",
            ],
        )
        for name, new in [("vf", "vf"), ("vf-rank-session", "cpu_worker")]:
            shutil.copyfile(ROOT / "target/release" / name, out / new)
            (out / new).chmod(0o700)
        run("elf", ["readelf", "-h", str(out / "worker_release")])
        run("dynamic", ["readelf", "-d", str(out / "worker_release")])
        run(
            "native_symbols",
            ["readelf", "--version-info", str(ROOT / "native/libvf_r13.so")],
        )
        run("loader", ["ldd", str(out / "worker_release")])
        assert "not found" not in (out / "loader.log").read_text()
        assert all(sha(ROOT / n) == h for n, h in manifest.items()), (
            "inputs drifted during build"
        )
        report.update(
            status="PINNED_X86_RUST_PACKAGE_BUILD_PASS_NO_GPU",
            binaries_sha256={
                n: sha(out / n)
                for n in [
                    "worker_debug",
                    "worker_release",
                    "hardware_debug",
                    "hardware_release",
                    "truncation_debug",
                    "truncation_release",
                    "vf",
                    "cpu_worker",
                ]
            },
        )
    except Exception as e:
        report.update(status="BUILD_FAILED_NO_GPU", error=repr(e))
    finally:
        report.pop("active_step", None)
        report["end_ns"] = time.monotonic_ns()
        save()
        lease.close()
    print(json.dumps({"status": report["status"], "error": report.get("error")}))
    raise SystemExit(
        0 if report["status"] == "PINNED_X86_RUST_PACKAGE_BUILD_PASS_NO_GPU" else 1
    )


if __name__ == "__main__":
    main()
