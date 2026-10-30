#!/usr/bin/env python3
"""Publish one reviewed R116 branch without force, then verify a fresh clone."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
REPO = Path("/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/gitlab-import-r113-20261028/repository")
URL = "https://pc4.tail0c55ff.ts.net/root/voltafabric.git"
LOCAL_BRANCH = "r116-output-token-bound"
EXPECTED = "3474ca11fe04412f17bfceba399397c0d048496b"
PARENT = "9c7878b41fae56a6968da2217764aad99c7484ed"
CANONICAL_SHA = "41f179ce7d7076748e907e566182aa1d3298b4deb776cb52ecd0f694edba526d"
R116_MANIFEST_SHA = "fc05a3015a83aa3c9a909058ccfe59f17e7c4b93d95d926e36128953969d753a"
ASKPASS = Path("/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/gitlab-import-r113-20261028/askpass_memfd.py")
fd = None
clone = None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def receipt(name, value):
    path = ROOT / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    out_fd = os.open(path, flags, 0o600)
    with os.fdopen(out_fd, "wb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    dir_fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def capture(argv, cwd, env, timeout=40, cap=2 * 1024 * 1024):
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    buffers = [bytearray(), bytearray()]
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + timeout
    try:
        for index, stream in enumerate((proc.stdout, proc.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, index)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("bounded subprocess deadline")
            for key, _ in selector.select(min(0.2, remaining)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                else:
                    buffers[key.data].extend(chunk)
                    if sum(map(len, buffers)) > cap:
                        raise ValueError("bounded subprocess output exceeded cap")
        code = proc.wait(timeout=max(0.001, deadline - time.monotonic()))
        return code, bytes(buffers[0]), bytes(buffers[1])
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=5)
        selector.close()
        proc.stdout.close()
        proc.stderr.close()


def main():
    global fd, clone
    frame = sys.stdin.buffer.readline(8193)
    if not frame.endswith(b"\n") or len(frame) > 8192:
        raise ValueError("invalid credential frame")
    auth = json.loads(frame)
    username, password = auth.get("username"), auth.get("password")
    if type(username) is not str or re.fullmatch(r"[A-Za-z0-9._-]+", username) is None:
        raise ValueError("invalid username frame")
    if type(password) is not str or not password or any(ord(c) < 33 or ord(c) > 126 for c in password):
        raise ValueError("invalid password frame")
    if (ROOT / "PUSH_INTENT.json").exists():
        raise FileExistsError("push already attempted; no retry")
    if sha(ASKPASS.read_bytes()) != "1b95342fb7c4e7bbc9d280a24ed66e48fcfb5061188e207fbddf4175915d1102":
        raise ValueError("pinned askpass helper mismatch")
    if REPO.resolve() != REPO or REPO.is_symlink():
        raise ValueError("repository path is not canonical")
    git = ["git", "-c", "credential.helper=", "-c",
           "http.proxy=socks5h://127.0.0.1:1055", "-c", "http.sslVerify=true"]
    check = ["git", "-c", "credential.helper=", "-c",
             "http.proxy=socks5h://127.0.0.1:1055", "-c", "http.sslVerify=true"]
    head = subprocess.run(check + ["rev-parse", "HEAD"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.strip()
    parent = subprocess.run(check + ["rev-parse", "HEAD^"], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(check + ["symbolic-ref", "--short", "HEAD"], cwd=REPO,
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(check + ["status", "--porcelain=v1", "-z"], cwd=REPO,
                           check=True, capture_output=True).stdout
    if (head, parent, branch, dirty) != (EXPECTED, PARENT, LOCAL_BRANCH, b""):
        raise ValueError("local commit, parent, branch, or worktree mismatch")
    tracked = subprocess.run(check + ["ls-files", "-z"], cwd=REPO, check=True,
                             capture_output=True).stdout.split(b"\0")
    needle = password.encode()
    matches = 0
    for encoded in tracked:
        if not encoded:
            continue
        path = REPO / os.fsdecode(encoded)
        file_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise ValueError("tracked path is not a regular file")
            overlap = b""
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                if needle in overlap + chunk:
                    matches += 1
                overlap = (overlap + chunk)[-(len(needle) - 1):] if len(needle) > 1 else b""
        finally:
            os.close(file_fd)
    if matches:
        raise ValueError("credential scan found a match in tracked files")
    r116_manifest = (REPO / "research/R116/MANIFEST.sha256").read_bytes()
    if sha(r116_manifest) != R116_MANIFEST_SHA:
        raise ValueError("R116 evidence manifest mismatch")
    for line in r116_manifest.decode().splitlines():
        digest, rel = line.split("  ", 1)
        path = REPO / "research/R116" / rel
        if path.is_symlink() or not path.is_file() or sha(path.read_bytes()) != digest:
            raise ValueError("R116 manifested file mismatch")
    canonical_raw = (REPO / "provenance/source_final_r59.json").read_bytes()
    if sha(canonical_raw) != CANONICAL_SHA:
        raise ValueError("canonical source manifest mismatch")
    canonical = json.loads(canonical_raw)
    if len(canonical) != 370:
        raise ValueError("canonical manifest entry count mismatch")
    for rel, digest in canonical.items():
        path = REPO / rel
        if path.is_symlink() or not path.is_file() or sha(path.read_bytes()) != digest:
            raise ValueError("canonical file differs from its R59 manifest")

    fd = os.memfd_create("vf-r117-gitlab-auth", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    os.write(fd, json.dumps(auth, separators=(",", ":")).encode())
    os.lseek(fd, 0, os.SEEK_SET)
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
    env = dict(os.environ, GIT_ASKPASS=str(ASKPASS), GIT_TERMINAL_PROMPT="0",
               VF_R113_AUTH_MEMFD=f"/proc/{os.getpid()}/fd/{fd}",
               GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    for key in list(env):
        if key.startswith(("GIT_TRACE", "GIT_CURL_VERBOSE", "GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")) or key in (
                "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_SYSTEM", "GIT_SSL_NO_VERIFY"):
            env.pop(key, None)

    def run(args, cwd=REPO, timeout=40, cap=2 * 1024 * 1024):
        code, out, err = capture(git + args, cwd, env, timeout, cap)
        if password.encode() in out or password.encode() in err:
            raise ValueError("credential was reflected by subprocess")
        return code, out, err

    code, refs_raw, _ = run(["ls-remote", "--heads", URL])
    if code:
        raise RuntimeError("remote refs could not be read")
    refs = {}
    for line in refs_raw.decode("utf-8", "strict").splitlines():
        digest, ref = line.split("\t", 1)
        if ref in refs or not ref.startswith("refs/heads/"):
            raise ValueError("remote ref listing invalid")
        refs[ref] = digest
    if refs.get("refs/heads/main") != "7cc00f04eeef104e2d65997c5a8ebf50472193a4":
        raise ValueError("remote main moved; preserve and inspect before publishing")
    r114_ref = "refs/heads/r114-portable-snapshot-integrity-a58005e20d8d"
    if refs.get(r114_ref) != PARENT:
        raise ValueError("published R114 parent branch changed")
    suffix = os.urandom(6).hex()
    remote_branch = "r116-output-token-bound-" + suffix
    remote_ref = "refs/heads/" + remote_branch
    if remote_ref in refs:
        raise FileExistsError("random target branch already exists")
    receipt("PUSH_INTENT.json", {
        "status": "PUSH_STARTED_UNKNOWN_UNTIL_VERIFIED", "commit": EXPECTED,
        "parent": PARENT, "remote_branch": remote_branch, "remote_ref": remote_ref,
        "remote_main_before": refs["refs/heads/main"], "r114_parent_before": refs[r114_ref],
        "force": False, "automatic_retry": False,
        "canonical_files_preverified": 370,
        "r116_manifest_sha256": R116_MANIFEST_SHA,
        "credential_transport": "stdin_then_sealed_memfd_askpass",
        "credential_persisted": False,
    })
    try:
        code, out, err = run(["push", URL, "HEAD:" + remote_ref], timeout=180)
        terminal = {"status": "PUSH_RETURNED_ZERO_VERIFY_PENDING" if code == 0 else "PUSH_NONZERO_RECONCILIATION_REQUIRED",
                    "exit_code": code, "stdout_sha256": sha(out), "stderr_sha256": sha(err),
                    "stdout_bytes": len(out), "stderr_bytes": len(err), "automatic_retry": False}
        receipt("PUSH_TERMINAL.json", terminal)
    except Exception as exc:
        receipt("PUSH_TERMINAL.json", {"status": "PUSH_UNKNOWN_NO_RETRY",
                                        "error_class": type(exc).__name__, "automatic_retry": False})
        code = -1
    code, refs_after_raw, _ = run(["ls-remote", "--heads", URL])
    if code:
        receipt("REMOTE_VERIFICATION.json", {"status": "REMOTE_REF_UNRECONCILED",
                                               "exit_code": code, "automatic_retry": False})
        return 2
    refs_after = {}
    for line in refs_after_raw.decode("utf-8", "strict").splitlines():
        digest, ref = line.split("\t", 1)
        if ref in refs_after:
            raise ValueError("duplicate remote ref after push")
        refs_after[ref] = digest
    if refs_after.get(remote_ref) != EXPECTED:
        receipt("REMOTE_VERIFICATION.json", {"status": "REMOTE_REF_NOT_EQUAL_TO_EXPECTED",
                                               "branch": remote_branch, "commit": refs_after.get(remote_ref),
                                               "automatic_retry": False})
        return 2
    receipt("REMOTE_VERIFICATION.json", {"status": "REMOTE_REF_EQUALS_LOCAL_COMMIT",
                                           "branch": remote_branch, "commit": EXPECTED,
                                           "main_unchanged": refs_after.get("refs/heads/main") == refs["refs/heads/main"],
                                           "refs_sha256": sha(refs_after_raw)})
    clone = Path(tempfile.mkdtemp(prefix="clone-verify-", dir=ROOT))
    clone_inode = clone.stat().st_ino
    code, _, _ = run(["clone", "--no-checkout", "--quiet", URL, str(clone)], cwd=ROOT, timeout=180)
    if code:
        receipt("CLONE_VERIFICATION.json", {"status": "REMOTE_REF_VERIFIED_CLONE_FAILED", "exit_code": code})
        return 2
    remote_tracking = "refs/remotes/origin/" + remote_branch
    code, cloned_head, _ = run(["rev-parse", remote_tracking], cwd=clone)
    if code or cloned_head.decode().strip() != EXPECTED:
        raise ValueError("fresh clone branch commit mismatch")
    code, manifest_bytes, _ = run(["show", remote_tracking + ":provenance/source_final_r59.json"], cwd=clone)
    if code or sha(manifest_bytes) != CANONICAL_SHA:
        raise ValueError("fresh clone canonical manifest mismatch")
    for rel, digest in json.loads(manifest_bytes).items():
        code, blob, _ = run(["show", remote_tracking + ":" + rel], cwd=clone, cap=50 * 1024 * 1024)
        if code or sha(blob) != digest:
            raise ValueError("fresh clone canonical source mismatch")
    code, r116_bytes, _ = run(["show", remote_tracking + ":research/R116/MANIFEST.sha256"], cwd=clone)
    if code or sha(r116_bytes) != R116_MANIFEST_SHA:
        raise ValueError("fresh clone R116 manifest mismatch")
    new_files = {}
    for line in r116_bytes.decode().splitlines():
        digest, rel = line.split("  ", 1)
        code, blob, _ = run(["show", remote_tracking + ":research/R116/" + rel], cwd=clone,
                            cap=2 * 1024 * 1024)
        if code or sha(blob) != digest:
            raise ValueError("fresh clone R116 file mismatch")
        new_files[rel] = digest
    code, tree, _ = run(["ls-tree", "-r", "--name-only", "-z", remote_tracking], cwd=clone)
    if code:
        raise ValueError("fresh clone tree could not be listed")
    tree_count = len([item for item in tree.split(b"\0") if item])
    if tree_count != 520:
        raise ValueError("fresh clone tree file count mismatch")
    if clone.parent != ROOT or clone.stat().st_ino != clone_inode or clone.is_symlink():
        raise ValueError("fresh clone cleanup target changed")
    shutil.rmtree(clone)
    clone = None
    code, _, _ = run(["fetch", "--no-tags", URL, remote_ref + ":refs/remotes/origin/" + remote_branch], timeout=90)
    if code:
        raise RuntimeError("remote tracking ref fetch failed")
    code, _, _ = run(["branch", "--set-upstream-to=origin/" + remote_branch, LOCAL_BRANCH])
    if code:
        raise RuntimeError("local upstream configuration failed after verified push")
    receipt("CLONE_VERIFICATION.json", {"status": "FRESH_CLONE_CONTENT_VERIFIED",
                                          "commit": EXPECTED, "canonical_files_verified": 370,
                                          "r116_files_verified": len(new_files), "tree_files": tree_count,
                                          "clone_removed": True})
    receipt("FINAL.json", {"status": "R116_BRANCH_PUSH_AND_FRESH_CLONE_VERIFIED",
                            "commit": EXPECTED, "branch": remote_branch,
                            "main_unchanged": refs_after.get("refs/heads/main") == refs["refs/heads/main"],
                            "canonical_files_verified": 370, "r116_files_verified": len(new_files),
                            "tree_files": tree_count, "force": False,
                            "credential_persisted": False})
    print(json.dumps({"status": "R116_BRANCH_PUSH_AND_FRESH_CLONE_VERIFIED",
                      "commit": EXPECTED, "branch": remote_branch,
                      "canonical_files_verified": 370, "r116_files_verified": len(new_files),
                      "tree_files": tree_count, "force": False}))
    return 0


try:
    raise SystemExit(main())
except Exception as exc:
    result = {"status": "STOPPED_INSPECT_RECEIPTS", "error_class": type(exc).__name__,
              "push_intent_exists": (ROOT / "PUSH_INTENT.json").exists(), "automatic_retry": False}
    print(json.dumps(result))
    raise SystemExit(1)
finally:
    if fd is not None:
        os.close(fd)
    if clone is not None:
        # Keep a failed clone for inspection; never erase an unverified target.
        pass
