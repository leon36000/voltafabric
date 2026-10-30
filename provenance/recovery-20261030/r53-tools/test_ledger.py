import json, tempfile, unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import namespace_ledger


class Ledger(unittest.TestCase):
    def reserve(self, *args):
        implementation = getattr(namespace_ledger, "reserve", None)
        self.assertTrue(
            callable(implementation), "EFFECTIVE_NAMESPACE_RESERVATION_ABSENT"
        )
        return implementation(*args)

    def test_suffix_alias_is_consumed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = "r53-abcd1234"
            ids = self.reserve(root, base, {"case": "duplicate"})
            self.assertEqual(set(ids), {base, base + "-duplicate-slot"})
            with self.assertRaises(FileExistsError):
                self.reserve(root, base + "-duplicate-slot", {"case": "normal"})
            with self.assertRaises(FileExistsError):
                self.reserve(root, base, {"case": "repeat"})

    def test_reverse_alias_preserves_partial_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = "r53-abcd1234"
            self.reserve(root, base + "-duplicate-slot", {"case": "normal"})
            with self.assertRaises(FileExistsError):
                self.reserve(root, base, {"case": "duplicate"})
            self.assertTrue((root / (base + ".json")).exists())
            with self.assertRaises(FileExistsError):
                self.reserve(root, base, {"case": "repeat"})

    def test_unknown_or_truncated_receipt_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = "r53-ledger-unknown"
            self.reserve(root, base, {"status": "UNKNOWN_NO_REPLAY"})
            p = root / (base + ".json")
            original = json.loads(p.read_text())
            self.assertEqual(original["status"], "RESERVED_UNKNOWN_NO_REPLAY")
            p.write_bytes(b"{")
            with self.assertRaises(FileExistsError):
                self.reserve(root, base, {"case": "repeat"})

    def test_concurrent_same_base_has_one_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Exercise the actual atomic filesystem create, no mocked lock.
            self.assertTrue(
                callable(getattr(namespace_ledger, "reserve", None)),
                "EFFECTIVE_NAMESPACE_RESERVATION_ABSENT",
            )

            def once(i):
                try:
                    self.reserve(root, "r53-ledger-concurrent", {"owner": i})
                    return True
                except FileExistsError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(once, range(2)))
            self.assertEqual(sum(results), 1)

    def test_rejects_invalid_names_before_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for value in [
                "../outside",
                "r14-legacy-test",
                "r53-short",
                "r53-space here",
                "r53-" + ("x" * 61),
            ]:
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        self.reserve(root, value, {})
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
