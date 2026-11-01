import os
import pathlib
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "adapters/strata_native_bridge"))

import voltafabric_launch as launcher  # noqa: E402


class FakeCoordinator:
    def poll(self):
        return None


class FakeStrata:
    def wait(self):
        return 0


class StagePinSanitizationTests(unittest.TestCase):
    def test_launch_forces_child_off_without_mutating_parent_environment(self):
        sessions = []
        socket_pairs = []
        for layer in range(launcher.LAYERS):
            local, peer = launcher.socket.socketpair()
            socket_pairs.append((local, peer))
            stamp = {"run_id": "r652", "model_revision": "model", "plan_id": "plan"}
            sessions.append({"init": {"template": {"stamp": stamp}}})

        bridge_sessions = [
            launcher.BridgeSession(layer, FakeCoordinator(), pair[0], b"identity")
            for layer, pair in enumerate(socket_pairs)
        ]
        config = {
            "startup_timeout_ms": 30_000,
            "sessions": sessions,
            "strata_executable": "/fixture/strata",
            "strata_args": ["--fixture"],
            "working_directory": "/tmp",
            "epoch": 80,
            "first_batch": 1,
            "first_attempt": 1,
            "strata_timeout_ms": 1_000,
            "shutdown_timeout_ms": 1_000,
        }

        try:
            with mock.patch.dict(os.environ, {"STRATA_STAGE_PIN": "1", "VF_R652_KEEP": "present"}):
                with mock.patch.object(launcher, "_start_bridge", side_effect=bridge_sessions), \
                     mock.patch.object(launcher, "_finish_sessions", return_value=True), \
                     mock.patch.object(launcher.subprocess, "Popen", return_value=FakeStrata()) as popen:
                    self.assertEqual(launcher.launch(config), 0)
                    self.assertEqual(os.environ["STRATA_STAGE_PIN"], "1")

                kwargs = popen.call_args.kwargs
                self.assertEqual(kwargs["env"]["STRATA_STAGE_PIN"], "0")
                self.assertEqual(kwargs["env"]["VF_R652_KEEP"], "present")
                self.assertEqual(len(kwargs["pass_fds"]), launcher.LAYERS)
                self.assertTrue(kwargs["close_fds"])
        finally:
            for local, peer in socket_pairs:
                local.close()
                peer.close()


if __name__ == "__main__":
    unittest.main()
