"""A45: refusals the `if ..: raise` sweep could not reach.

guard_schemata.py --sites py-handlers catalogs two kinds of site. One is a guard
whose raise follows other statements; its operands are disabled as before. The
other is a `raise X(..)` written in an `except` body, which translates a low-level
error (OSError, JSONDecodeError, TimeoutExpired, ...) into the module's own refusal;
VF_MUTANT=ID makes it re-raise the original instead. Each test below drives one
site with an input that reaches it alone and pins the translated type, the full
anchored message and, for the preflight, the receipt status. An untranslated
exception, or one of a parent type with another message, then fails the test.

The ids in the comments are the py-handlers catalog ids. CPU only: no CUDA
runtime, library or device is used, and the commands run are /bin/sh builtins
and sleep.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SCRIPT_ROOT = HERE.parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(HERE))
from scripts import h1_preflight  # noqa: E402
from scripts import hardware_canary  # noqa: E402
from scripts import resident_artifact as ra  # noqa: E402
import test_a43_preflight_refusals as a43  # noqa: E402
import test_a44_artifact_refusals as a44  # noqa: E402

PreflightError = h1_preflight.PreflightError
SH = shutil.which("sh") or "/bin/sh"


class Temp(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="a45-"))
        self.addCleanup(shutil.rmtree, self.root)

    def preflight_refused(self, pattern, status, fn, *args, **kwargs):
        with self.assertRaisesRegex(PreflightError, pattern) as caught:
            fn(*args, **kwargs)
        self.assertIs(type(caught.exception), PreflightError)
        self.assertEqual(caught.exception.status, status)
        return caught.exception


class PreflightGuards(Temp):
    """Guards whose raise follows an assignment (ids 1-3, 11, 15, 17)."""

    def elf(self, header):
        tools = a43.FakeTools()
        tools.header["vf"] = header
        target = self.root / "vf"
        target.write_bytes(b"stand-in")
        with tools.installed():
            return h1_preflight._check_elf("/fake/bin/readelf", target, "bin/vf")

    def test_elf_control_passes(self):
        needed, _, _ = self.elf(a43.HEADER)
        self.assertEqual(needed, {"libc.so.6", "libm.so.6"})

    def test_missing_machine_line(self):  # 1: `not machine`
        header = a43.HEADER.replace(
            "  Machine:                           Advanced Micro Devices X86-64\n", ""
        )
        self.assertNotIn("Machine", header)
        self.preflight_refused(
            r"^bin/vf ELF machine is not x86-64: unknown$", "FAILED", self.elf, header
        )

    def test_other_machine(self):  # 2: `"X86-64" not in ...`
        header = a43.HEADER.replace("Advanced Micro Devices X86-64", "AArch64")
        self.preflight_refused(
            r"^bin/vf ELF machine is not x86-64: AArch64$", "FAILED", self.elf, header
        )

    def test_unlisted_payload_file(self):  # 3: actual_files != set(hashes)
        a43.build(self.root)
        (self.root / "extra.txt").write_text("not in SHA256SUMS\n")
        tools = a43.FakeTools()
        with (
            mock.patch.object(h1_preflight.platform, "machine", return_value="x86_64"),
            tools.installed(),
        ):
            self.preflight_refused(
                r"^SHA256SUMS payload coverage mismatch; "
                r"unlisted=\['extra\.txt'\], missing=\[\]$",
                "FAILED",
                h1_preflight.preflight,
                self.root,
            )

    def test_failing_command(self):  # 11: proc.returncode != 0
        self.preflight_refused(
            r"^command failed \(3\): partial$",
            "FAILED",
            h1_preflight._run_bounded,
            [SH, "-c", "echo partial; exit 3"],
        )

    def test_deadline_with_open_output(self):  # 15: remaining <= 0
        # The child keeps its output open and silent, so only the deadline ends it.
        self.preflight_refused(
            r"^command timed out: sh$",
            "BLOCKED",
            h1_preflight._run_bounded,
            [SH, "-c", "exec sleep 5"],
            timeout=1,
        )

    def test_output_cap(self):  # 17: len(output) > MAX_TOOL_OUTPUT
        with mock.patch.object(h1_preflight, "MAX_TOOL_OUTPUT", 16):
            self.preflight_refused(
                r"^command output exceeded 16 bytes$",
                "FAILED",
                h1_preflight._run_bounded,
                [SH, "-c", "printf 0123456789abcdefXYZ"],
            )

    def test_bounded_command_control(self):
        self.assertEqual(
            h1_preflight._run_bounded([SH, "-c", "printf ok"], timeout=5), b"ok"
        )


class PreflightTranslations(Temp):
    """`except ..: raise PreflightError(..)` sites (ids 4-10, 12, 14, 16, 18)."""

    def test_missing_bundle_path(self):  # 4: (OSError, ValueError)
        self.preflight_refused(
            r"^missing or escaping bundle path: absent\.txt$",
            "FAILED",
            h1_preflight._bundle_file,
            self.root,
            "absent.txt",
        )

    def test_unstattable_bundle_file(self):  # 16: OSError from stat
        (self.root / "run_h1.sh").write_text("x\n")
        real_stat = Path.stat

        def stat(path, *, follow_symlinks=True):
            # Only the following stat fails, so the lstat of is_symlink() passes.
            if path.name == "run_h1.sh" and follow_symlinks:
                raise PermissionError("denied")
            return real_stat(path, follow_symlinks=follow_symlinks)

        with mock.patch.object(Path, "stat", stat):
            self.preflight_refused(
                r"^cannot stat bundle file run_h1\.sh: denied$",
                "FAILED",
                h1_preflight._check_no_symlinks_and_bounds,
                self.root,
            )

    def test_unreadable_sums(self):  # 5: OSError from read_bytes
        (self.root / "SHA256SUMS").mkdir()
        self.preflight_refused(
            r"^cannot read SHA256SUMS: \[Errno 21\] Is a directory: ",
            "FAILED",
            h1_preflight._parse_sha256sums,
            self.root / "SHA256SUMS",
            self.root,
        )

    def test_sums_not_utf8(self):  # 6: UnicodeDecodeError
        (self.root / "SHA256SUMS").write_bytes(b"\xff\n")
        self.preflight_refused(
            r"^SHA256SUMS is not UTF-8$",
            "FAILED",
            h1_preflight._parse_sha256sums,
            self.root / "SHA256SUMS",
            self.root,
        )

    def test_unhashable_file(self):  # 7: OSError while hashing
        target = self.root / "lib"
        target.mkdir()
        self.preflight_refused(
            "^" + re.escape(f"cannot hash {target}: [Errno 21] Is a directory: "),
            "FAILED",
            h1_preflight._hash_file,
            target,
        )

    def test_unreadable_json(self):  # 8: OSError from read_bytes
        (self.root / "PACKAGE.json").mkdir()
        self.preflight_refused(
            r"^cannot read PACKAGE\.json: \[Errno 21\] Is a directory: ",
            "FAILED",
            h1_preflight._read_json,
            self.root / "PACKAGE.json",
            "PACKAGE.json",
        )

    def test_invalid_json(self):  # 9: JSONDecodeError and UnicodeDecodeError
        for raw, detail in (
            (b"{", r"Expecting property name enclosed in double quotes"),
            (b"\xff", r"'utf-8' codec can't decode byte 0xff"),
        ):
            with self.subTest(raw=raw):
                (self.root / "PACKAGE.json").write_bytes(raw)
                self.preflight_refused(
                    r"^invalid PACKAGE\.json: " + detail,
                    "FAILED",
                    h1_preflight._read_json,
                    self.root / "PACKAGE.json",
                    "PACKAGE.json",
                )

    def test_command_cannot_start(self):  # 10: OSError from Popen
        missing = str(self.root / "no-such-tool")
        self.preflight_refused(
            "^" + re.escape(f"cannot execute {missing}: [Errno 2]"),
            "BLOCKED",
            h1_preflight._run_bounded,
            [missing],
        )

    def test_bundle_directory_missing(self):  # 12: OSError from resolve
        self.preflight_refused(
            r"^bundle directory unavailable: \[Errno 2\] No such file or directory: ",
            "FAILED",
            h1_preflight.preflight,
            self.root / "absent",
        )

    def test_receipt_created_between_check_and_open(self):  # 14: FileExistsError
        out = self.root / "receipt.json"
        out.write_text("earlier receipt\n")
        # The exists()/is_symlink() check has already passed when a racing
        # writer creates the file; only the exclusive open can still refuse.
        with (
            mock.patch.object(Path, "exists", lambda _p: False),
            mock.patch.object(Path, "is_symlink", lambda _p: False),
        ):
            self.preflight_refused(
                "^" + re.escape(f"refusing to overwrite output receipt: {out}") + "$",
                "FAILED",
                h1_preflight._write_receipt,
                out,
                {"status": "FAILED"},
            )
        self.assertEqual(out.read_text(), "earlier receipt\n")

    def test_silent_child_outlives_deadline(self):  # 18: TimeoutExpired
        # Output closes at once (EOF), but the process keeps running, so the
        # deadline is met in proc.wait() rather than in the read loop.
        self.preflight_refused(
            r"^command timed out: sh$",
            "BLOCKED",
            h1_preflight._run_bounded,
            [SH, "-c", "exec >&- 2>&-; exec sleep 5"],
            timeout=1,
        )


class CanaryTranslation(Temp):
    def test_output_outside_receipt_root(self):  # 19: ValueError from relative_to
        receipts = self.root / "receipts"
        receipts.mkdir()
        with mock.patch.dict(os.environ, {"VF_H1_RECEIPT_DIR": str(receipts)}):
            inside = hardware_canary.output_path(receipts / "h1.json")
            self.assertEqual(inside, (receipts / "h1.json").resolve())
            with self.assertRaisesRegex(
                ValueError,
                "^"
                + re.escape(
                    "--out must stay below the resolved project root: "
                    + str(receipts.resolve())
                )
                + "$",
            ):
                hardware_canary.output_path(self.root / "elsewhere" / "h1.json")


class ArtifactTranslations(a44.Base):
    """`except ..: raise _fail(..)` sites of resident_artifact (ids 25-28)."""

    def test_invalid_header_json(self):  # 25: JSONDecodeError
        self.refuse_file(
            "invalid Safetensors header JSON", a44.encode(raw_mut=lambda r: r[:-1])
        )

    def test_header_not_utf8(self):  # 25: UnicodeDecodeError
        blob = bytearray(a44.encode())
        at = blob.index(b"test-rev")
        blob[at : at + 1] = b"\xff"
        self.refuse_file("invalid Safetensors header JSON", bytes(blob))

    def test_final_symlink_is_not_followed(self):  # 26: OSError from os.open
        target = self.write(a44.encode())
        link = self.dir / "link.safetensors"
        link.symlink_to(target)
        ra.load_stack(target)
        self.refuse(
            "cannot open regular artifact without following final symlink",
            ra.load_stack,
            link,
        )

    def test_reader_unavailable(self):  # 27: ImportError
        path = self.write(a44.encode())
        with mock.patch.dict(sys.modules, {"safetensors": None}):
            self.refuse("safetensors.safe_open is unavailable", ra.load_stack, path)

    def test_reader_failure_is_wrapped(self):  # 28: Exception from the reader
        path = self.write(a44.encode())

        def broken(*_a, **_k):
            raise RuntimeError("reader exploded")

        with mock.patch("safetensors.safe_open", broken):
            self.refuse("Safetensors reader rejected file", ra.load_stack, path)


if __name__ == "__main__":
    unittest.main()
