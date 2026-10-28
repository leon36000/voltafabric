"""A53-R40: a CPU qualification step passes only with the JSON integer 0.

package_h1 and h1_preflight both accepted a CPU_FULL_PASS receipt whose steps compared
`step.get("exit_code") != 0`, which in Python is false for `false`, `0.0` and `-0.0`,
and crashed with AttributeError on a step that is not an object. verify.py always
writes integer exit codes, so any other spelling is a forged or damaged receipt and
must be refused by each guard with its own message.

Each case runs in a fresh fixture. The preflight cases pin the host architecture to
x86_64 (the host check is not under test and runs first), so they also discriminate
on an aarch64 host.
"""

from __future__ import annotations

import unittest
from unittest import mock

import test_a43_package_refusals as package_tests
import test_a43_preflight_refusals as preflight_tests

# Spellings that are not the integer 0. Before R40, `!= 0` let False, 0.0 and -0.0
# pass; "0", None, 1 and True were already refused and stay as controls.
NOT_ZERO = [False, 0.0, -0.0, "0", None, 1, True]


def _steps_cases():
    for value in NOT_ZERO:
        yield (
            f"exit_code={value!r}",
            [{"step": "a", "exit_code": 0}, {"exit_code": value}],
        )
    yield "missing exit_code", [{"step": "a", "exit_code": 0}, {"step": "b"}]
    yield "step is not an object", [{"step": "a", "exit_code": 0}, 0]
    yield "step is a list", [["exit_code", 0]]


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


class PackageExitCodeTests(unittest.TestCase):
    # Borrow the A43 fixture and helpers without inheriting (and re-running) its tests.
    _base = package_tests.PackageRefusalTests
    setUp = _base.setUp
    _fixture = _base._fixture
    _write_proofs = _base._write_proofs
    _refuse = _base._refuse
    _cpu = _base._cpu
    CPU_RECEIPT = _base.CPU_RECEIPT

    def test_integer_zero_steps_still_package(self):
        self._cpu(steps=[{"step": "a", "exit_code": 0}, {"step": "b", "exit_code": 0}])
        result = package_tests.package_h1.package(self.args)
        self.assertEqual(result["status"], "H1_PACKAGE_PREPARED_NO_GPU")


class X86Bundle(preflight_tests.Bundle):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.object(
            preflight_tests.h1_preflight.platform, "machine", return_value="x86_64"
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class PreflightExitCodeTests(X86Bundle):
    def test_integer_zero_steps_still_pass(self):
        preflight_tests.build(
            self.root,
            cpu=lambda c: c.update(steps=[{"exit_code": 0}, {"exit_code": 0}]),
        )
        with self.tools.installed():
            status, _checks, _details = preflight_tests.h1_preflight.preflight(
                self.root
            )
        self.assertEqual(status, "STATIC_PREFLIGHT_PASS_NO_GPU")


def _add_cases():
    for name, steps in _steps_cases():

        def package_case(self, s=steps):
            self._cpu(steps=s)
            self._refuse(self.CPU_RECEIPT)

        def preflight_case(self, s=steps):
            self.refused(
                "not a complete unchanged-source pass",
                cpu=lambda c: c.update(steps=s),
            )

        for cls, test in (
            (PackageExitCodeTests, package_case),
            (PreflightExitCodeTests, preflight_case),
        ):
            test.__name__ = f"test_refuses_{_slug(name)}"
            setattr(cls, test.__name__, test)


_add_cases()


if __name__ == "__main__":
    unittest.main()
