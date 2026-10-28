"""The native loader must distinguish dense Skinny from its required MoE API."""

import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from adapters import skinny_moe as adapter


MARLIN = "vllm.model_executor.kernels.linear.nvfp4.marlin"


class NativeProviderContractTests(unittest.TestCase):
    def test_dense_only_or_noncallable_moe_export_is_rejected_at_resolution(self):
        for native in (
            SimpleNamespace(gemm_qpn2=lambda: None),
            SimpleNamespace(gemm_qpn2_moe=None),
            SimpleNamespace(gemm_qpn2_moe=123),
        ):
            loader = mock.Mock(return_value=native)
            with (
                self.subTest(native=native),
                mock.patch.dict(
                    sys.modules, {MARLIN: SimpleNamespace(_get_skinny_ext=loader)}
                ),
                self.assertRaisesRegex(
                    RuntimeError, "does not export callable gemm_qpn2_moe"
                ),
            ):
                adapter._ext()
            loader.assert_called_once_with()

    def test_moe_callable_is_returned_without_invoking_a_kernel(self):
        kernel = mock.Mock()
        native = SimpleNamespace(gemm_qpn2_moe=kernel)
        with mock.patch.dict(
            sys.modules, {MARLIN: SimpleNamespace(_get_skinny_ext=lambda: native)}
        ):
            self.assertIs(adapter._ext(), native)
        kernel.assert_not_called()

    def test_native_loader_failure_is_preserved(self):
        error = RuntimeError("native loader failure")
        loader = mock.Mock(side_effect=error)
        with mock.patch.dict(
            sys.modules, {MARLIN: SimpleNamespace(_get_skinny_ext=loader)}
        ):
            with self.assertRaises(RuntimeError) as caught:
                adapter._ext()
        self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()
