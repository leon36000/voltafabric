import hashlib
import json
import os
import pathlib
import socket
import struct
import sys
import tempfile
import unittest

import voltafabric_launch as launcher


HIDDEN = 2560


def write_executable(path: pathlib.Path, body: str) -> pathlib.Path:
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(0o755)
    return path


def make_fixture(root: pathlib.Path, bridge_body: str, strata_body: str) -> pathlib.Path:
    pack = root / "pack.gguf"
    pack.write_bytes(b"P" * 64)
    index = "# strata native experts v3: fixture (n_expert 512, total 50292326400; synthetic)\n"
    index_hash = hashlib.sha256(index.encode()).hexdigest()
    model_experts = list(range(512))
    bindings = []
    for layer in range(48):
        for expert in range(512):
            bindings.append({
                "rank": 0, "layer": layer, "expert": expert, "local": expert, "epoch": 7,
                "index_sha256": index_hash, "blob_sha256": f"{expert:064x}",
            })
    checkpoint = {
        "schema_version": 1, "pack_path": str(pack), "index": index,
        "model_experts": model_experts, "bindings": bindings,
    }
    checkpoint_path = root / "checkpoint-manifest.json"
    checkpoint_path.write_text(json.dumps(checkpoint, separators=(",", ":")))
    copies = [{"expert": expert, "rank": 0, "local": expert} for expert in model_experts]
    layer_templates = []
    for layer in range(48):
        layer_templates.append({
            "hidden": HIDDEN, "experts": model_experts, "routes": [],
            "registry": {"epoch": 7, "copies": copies},
            "stamp": {"run_id": "", "model_revision": "", "plan_id": "",
                      "layer": layer, "batch": 0, "epoch": 7},
            "resident_order": {"0": model_experts},
        })
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps({"schema_version": 1, "layer_templates": layer_templates}, separators=(",", ":")))
    endpoint = write_executable(root / "rank-worker", "import sys\nsys.exit(0)\n")
    bridge = write_executable(root / "bridge", bridge_body)
    strata = write_executable(root / "strata", strata_body)
    ledger_paths = []
    for layer in range(48):
        directory = root / f"ledger-{layer}"
        directory.mkdir()
        ledger_paths.append(str(directory))
    manifest = {
        "schema_version": 1, "run_id": "r642-fixture", "epoch": 7,
        "first_batch": 11, "first_attempt": 1,
        "startup_timeout_ms": 30_000, "shutdown_timeout_ms": 5_000,
        "coordinator_timeout_ms": 1_000, "strata_timeout_ms": 1_000,
        "aggregate_resident_payload_limit": 48 * 1_200_000_000,
        "checkpoint_manifest_path": str(checkpoint_path), "plan_path": str(plan_path),
        "bridge_executable": str(bridge), "strata_executable": str(strata),
        "working_directory": str(root), "strata_args": ["--tokens", "1,2"],
        "endpoints": [{"rank": 0, "executable": str(endpoint)}],
        "ledger_paths": ledger_paths, "resident_payload_limits": [1_200_000_000] * 48,
    }
    manifest_path = root / "launch.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    return manifest_path


BRIDGE_BODY = r'''import json, socket, struct, sys

def read_exact(n):
    data = bytearray()
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk: raise SystemExit(20)
        data.extend(chunk)
    return bytes(data)

def send_json(value):
    sys.stdout.buffer.write(json.dumps(value, separators=(",", ":")).encode() + b"\\n")
    sys.stdout.buffer.flush()

init = json.loads(sys.stdin.buffer.readline().decode())
ranks = sorted(int(x) for x in init["template"]["resident_order"])
send_json({"state":"ready", "ranks":ranks})
if json.loads(sys.stdin.buffer.readline().decode()) != {"op":"enter_binary"}: raise SystemExit(21)
send_json({"state":"binary_ready"})
header = read_exact(52)
magic, op, batch, tokens, top_k, layer, epoch, rl, ml, pl, reserved, attempt = struct.unpack("<4sIQIIIQHHHHQ", header)
identity = read_exact(rl + ml + pl)
if magic != b"VFB3" or op != 2 or batch or tokens or top_k or attempt or reserved: raise SystemExit(22)
reply = struct.pack("<4sIQIIIQHHHHQ", b"VFR3", 2, 0, 0, 2560, layer, epoch, rl, ml, pl, 0, 0) + identity
sys.stdout.buffer.write(reply)
sys.stdout.buffer.flush()
'''.replace('b"\\\\n"', 'b"\\n"')


STRATA_BODY = r'''import os, socket, struct, sys
args = sys.argv[1:]
fd_text = args[args.index("--vf-session-fds") + 1]
fds = [int(x) for x in fd_text.split(",")]
assert len(fds) == 48 and len(set(fds)) == 48
opts = {args[i]:args[i+1] for i in range(len(args)-1) if args[i].startswith("--vf-")}
run = opts["--vf-run-id"].encode(); model = opts["--vf-model-revision"].encode(); plan = opts["--vf-plan-id"].encode()
epoch = int(opts["--vf-epoch"])
for layer, fd in enumerate(fds):
    sock = socket.socket(fileno=fd)
    sock.settimeout(2)
    hdr = struct.pack("<4sIQIIIQHHHHQ", b"VFB3", 2, 0, 0, 0, layer, epoch,
                      len(run), len(model), len(plan), 0, 0)
    sock.sendall(hdr + run + model + plan)
    data = bytearray()
    while len(data) < 52:
        data.extend(sock.recv(52 - len(data)))
    fields = struct.unpack("<4sIQIIIQHHHHQ", data)
    assert fields[0] == b"VFR3" and fields[1] == 2 and fields[5] == layer
    size = fields[7] + fields[8] + fields[9]
    echoed = bytearray()
    while len(echoed) < size:
        echoed.extend(sock.recv(size - len(echoed)))
    assert bytes(echoed) == run + model + plan
    sock.close()
'''


class LauncherTests(unittest.TestCase):
    def test_launch_passes_48_initialized_identity_bound_streams(self):
        previous = launcher.SC117_PACK_BYTES
        launcher.SC117_PACK_BYTES = 64  # synthetic fixture only; no model bytes loaded
        try:
            with tempfile.TemporaryDirectory(prefix="r642-launch-", dir=os.environ["VF_STRATA_LAUNCH_TEST_TMP_ROOT"]) as td:
                manifest = make_fixture(pathlib.Path(td), BRIDGE_BODY, STRATA_BODY)
                config = launcher._check_manifest(str(manifest))
                self.assertEqual(len(config["sessions"]), 48)
                self.assertTrue(config["model_revision"].startswith("sha256:"))
                self.assertTrue(config["plan_id"].startswith("sha256:"))
                self.assertEqual(launcher.launch(config), 0)
        finally:
            launcher.SC117_PACK_BYTES = previous

    def test_plan_template_mismatch_rejects_before_process_start(self):
        previous = launcher.SC117_PACK_BYTES
        launcher.SC117_PACK_BYTES = 64
        try:
            with tempfile.TemporaryDirectory(prefix="r642-reject-", dir=os.environ["VF_STRATA_LAUNCH_TEST_TMP_ROOT"]) as td:
                root = pathlib.Path(td)
                sentinel = root / "bridge-started"
                bridge_body = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('started')\n"
                manifest_path = make_fixture(root, bridge_body, STRATA_BODY)
                manifest = json.loads(manifest_path.read_text())
                plan_path = pathlib.Path(manifest["plan_path"])
                plan = json.loads(plan_path.read_text())
                plan["layer_templates"][17]["hidden"] = 1024
                plan_path.write_text(json.dumps(plan, separators=(",", ":")))
                with self.assertRaises(launcher.LaunchError):
                    launcher._check_manifest(str(manifest_path))
                self.assertFalse(sentinel.exists())
        finally:
            launcher.SC117_PACK_BYTES = previous


if __name__ == "__main__":
    unittest.main()
