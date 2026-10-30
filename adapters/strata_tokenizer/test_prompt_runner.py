import json
import base64
import os
import hashlib
import struct
import subprocess
import shutil
import tempfile
import sys
import unittest
from unittest.mock import patch

from adapters.strata_tokenizer.prompt_runner import _pinned_executable, _run_bounded, _verify_qsa_trace, run_prompt as _run_prompt
from adapters.strata_tokenizer.test_adapter import FakeTokenizer


def run_prompt(*args, **kwargs):
    # The old synthetic fixtures deliberately exercise legacy compatibility.
    kwargs.setdefault("require_bound_plan", False)
    return _run_prompt(*args, **kwargs)


class Tokenizer(FakeTokenizer):
    def encode(self, text, parse_special=False):
        return {"A": [0], "multi": [0, 1], "eight": [0] * 8, "long32": [0] * 32, "long64": [0] * 64, "long128": [0] * 128, "special": [3], "too many": [0] * 129}.get(text, [])


def receipt():
    return {"layer_count": 48, "run_id": "fresh", "workers_closed": 96, "eos_token_id": 3,
            "input_tokens": [0, 1], "generated_token_ids": [1, 2],
            "output_sha256": ["a" * 64] * 2, "logits_sha256": ["b" * 64] * 2}


class RunnerTests(unittest.TestCase):
    def test_qsa_trace_validates_length_digest_and_metadata(self):
        raw = b"\0" * (37380 * 4)
        trace = {"version": "vf-qsa-trace-v1-layer3-f32le", "layer": 3,
                 "positions": 1, "words_per_position": 37380,
                 "sha256": hashlib.sha256(raw).hexdigest(),
                 "base64": base64.b64encode(raw).decode("ascii")}
        record = {"input_tokens": [0], "qsa_trace": trace}
        _verify_qsa_trace(record)
        for key, invalid in (("positions", 2), ("layer", 7), ("sha256", "0" * 64),
                             ("base64", trace["base64"][:-4])):
            bad = dict(trace, **{key: invalid})
            with self.assertRaises(ValueError):
                _verify_qsa_trace({"input_tokens": [0], "qsa_trace": bad})

    def test_trace_is_required_for_trace_plan_and_refused_for_regular_plan(self):
        raw = b"\0" * (2 * 37380 * 4)
        trace = {"version": "vf-qsa-trace-v1-layer3-f32le", "layer": 3,
                 "positions": 2, "words_per_position": 37380,
                 "sha256": hashlib.sha256(raw).hexdigest(),
                 "base64": base64.b64encode(raw).decode("ascii")}
        value = dict(receipt(), prompt_tokens=[0], plan_id="r537-" + "a" * 64,
                     prompt_tokens_sha256=hashlib.sha256(struct.pack("<I", 0)).hexdigest(),
                     qsa_trace=trace)
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        def run():
            return _run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(run()["output"]["text"], "😀")
            del value["qsa_trace"]
            with self.assertRaisesRegex(ValueError, "QSA trace metadata"):
                run()
            value["qsa_trace"] = trace
            value["plan_id"] = "r531-" + "a" * 64
            with self.assertRaisesRegex(ValueError, "trace-bound plan"):
                run()

    def test_input_trace_layout_is_bound_to_new_plan(self):
        raw = b"\0" * (47620 * 4)
        trace = {"version": "vf-qsa-trace-v2-layer3-input-f32le", "layer": 3,
                 "positions": 1, "words_per_position": 47620,
                 "sha256": hashlib.sha256(raw).hexdigest(),
                 "base64": base64.b64encode(raw).decode("ascii")}
        record = {"input_tokens": [0], "qsa_trace": trace, "plan_id": "r544-" + "a" * 64}
        _verify_qsa_trace(record)
        record["plan_id"] = "r537-" + "a" * 64
        with self.assertRaisesRegex(ValueError, "trace metadata"):
            _verify_qsa_trace(record)

    def test_prepare_plan_uses_tokenizer_and_rejects_bad_plan_receipts(self):
        prompt_hash = hashlib.sha256(struct.pack("<I", 0)).hexdigest()
        good = {"plan_id": "r512-" + "a" * 64, "prompt_tokens_sha256": prompt_hash}
        state = {"body": json.dumps(good).encode()}
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, state["body"], b"")
        def prepare():
            return _run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8,
                               prepare_plan=True, runner=runner)
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(prepare(), dict(good, prompt_token_ids=[0]))
            self.assertEqual(calls[-1][1], "--plan-only")
            state["body"] = json.dumps(dict(good, plan_id="r520-" + "a" * 64)).encode()
            self.assertEqual(prepare()["plan_id"], "r520-" + "a" * 64)
            state["body"] = json.dumps(dict(good, plan_id="r523-" + "a" * 64)).encode()
            self.assertEqual(prepare()["plan_id"], "r523-" + "a" * 64)
            state["body"] = json.dumps(dict(good, plan_id="r525-" + "a" * 64)).encode()
            self.assertEqual(prepare()["plan_id"], "r525-" + "a" * 64)
            state["body"] = json.dumps(dict(good, plan_id="r527-" + "a" * 64)).encode()
            self.assertEqual(prepare()["plan_id"], "r527-" + "a" * 64)
            state["body"] = json.dumps(dict(good, plan_id="r528-" + "a" * 64)).encode()
            self.assertEqual(prepare()["plan_id"], "r528-" + "a" * 64)
            state["body"] = json.dumps(dict(good, plan_id="r529-" + "a" * 64)).encode()
            self.assertEqual(prepare()["plan_id"], "r529-" + "a" * 64)
            state["body"] = json.dumps(dict(good, plan_id="r531-" + "a" * 64)).encode()
            self.assertEqual(prepare()["plan_id"], "r531-" + "a" * 64)
            for bad in ({"plan_id": good["plan_id"]},
                        dict(good, prompt_tokens_sha256="0" * 64),
                        dict(good, plan_id="r512-" + "A" * 64),
                        dict(good, extra=1)):
                state["body"] = json.dumps(bad).encode()
                with self.assertRaises(ValueError):
                    prepare()
            state["body"] = b'{' + b' ' * 1100 + b'}'
            with self.assertRaisesRegex(ValueError, "prepared plan"):
                prepare()
            state["body"] = json.dumps(good).encode()
        with patch.dict("os.environ", {"VF_EXPECTED_PLAN_ID": "r512-" + "b" * 64}):
            with self.assertRaisesRegex(ValueError, "differs from expected plan"):
                prepare()

    def test_strict_receipt_rejects_forged_shorter_prompt(self):
        # Input starts with the actual two-token prompt, but the receipt treats
        # its second token as generated; the old prefix-only check accepted it.
        value = dict(receipt(), prompt_tokens=[0], plan_id="r520-" + "a" * 64,
                     prompt_tokens_sha256=hashlib.sha256(struct.pack("<II", 0, 1)).hexdigest())
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        with patch.dict("os.environ", {}, clear=True):
            for forged in ([0], [0, True], None, [0, 1, 2]):
                value["prompt_tokens"] = forged
                with self.assertRaisesRegex(ValueError, "receipt differs from prompt tokens"):
                    _run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
            value.pop("prompt_tokens")
            with self.assertRaisesRegex(ValueError, "receipt differs from prompt tokens"):
                _run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)

    def test_expected_plan_is_checked_before_and_after_engine(self):
        value = dict(receipt(), prompt_tokens=[0], plan_id="r512-" + "a" * 64,
                     prompt_tokens_sha256=hashlib.sha256(struct.pack("<I", 0)).hexdigest())
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        def run():
            return _run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        with patch.dict("os.environ", {"VF_EXPECTED_PLAN_ID": value["plan_id"]}):
            self.assertEqual(run()["output"]["text"], "😀")
        with patch.dict("os.environ", {"VF_EXPECTED_PLAN_ID": "r512-" + "b" * 64}):
            with self.assertRaisesRegex(ValueError, "differs from expected plan"):
                run()
        count = len(calls)
        with patch.dict("os.environ", {"VF_EXPECTED_PLAN_ID": "malformed"}):
            with self.assertRaisesRegex(ValueError, "invalid expected plan"):
                run()
        self.assertEqual(len(calls), count)

    def test_strict_plan_rejects_missing_unknown_malformed_and_missing_digest(self):
        value = dict(receipt(), prompt_tokens=[0], plan_id="r512-" + "a" * 64,
                     prompt_tokens_sha256=hashlib.sha256(struct.pack("<I", 0)).hexdigest())
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        def run():
            return _run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(run()["output"]["text"], "😀")
        value["plan_id"] = "r518-" + "a" * 64
        self.assertEqual(run()["output"]["text"], "😀")
        value["plan_id"] = "r520-" + "a" * 64
        self.assertEqual(run()["output"]["text"], "😀")
        value["plan_id"] = "r523-" + "a" * 64
        self.assertEqual(run()["output"]["text"], "😀")
        value["plan_id"] = "r525-" + "a" * 64
        self.assertEqual(run()["output"]["text"], "😀")
        value["plan_id"] = "r527-" + "a" * 64
        self.assertEqual(run()["output"]["text"], "😀")
        value["plan_id"] = "r528-" + "a" * 64
        self.assertEqual(run()["output"]["text"], "😀")
        value["plan_id"] = "r529-" + "a" * 64
        self.assertEqual(run()["output"]["text"], "😀")
        value["plan_id"] = "r531-" + "a" * 64
        self.assertEqual(run()["output"]["text"], "😀")
        for bad in (None, "r508-" + "a" * 64, "r512-" + "A" * 64,
                    "r512-" + "a" * 63, 123):
            value["plan_id"] = bad
            with self.assertRaisesRegex(ValueError, "bound plan identity"):
                run()
        del value["plan_id"]
        with self.assertRaisesRegex(ValueError, "bound plan identity"):
            run()
        value["plan_id"] = "r512-" + "a" * 64
        for bad in (None, "0" * 64):
            value["prompt_tokens_sha256"] = bad
            with self.assertRaisesRegex(ValueError, "prompt digest"):
                run()
        del value["prompt_tokens_sha256"]
        with self.assertRaisesRegex(ValueError, "prompt digest"):
            run()

    def test_prompt_digest_is_checked_when_present_or_required_by_plan(self):
        value = dict(receipt(), plan_id="r506-" + "a"*64,
                     prompt_tokens_sha256=hashlib.sha256(struct.pack("<I", 0)).hexdigest())
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        self.assertEqual(run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)["output"]["text"], "😀")
        for bad in ("0"*64, None):
            value["prompt_tokens_sha256"] = bad
            with self.assertRaisesRegex(ValueError, "prompt digest"):
                run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)
        del value["prompt_tokens_sha256"]
        with self.assertRaisesRegex(ValueError, "prompt digest"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)
        value["plan_id"] = "r508-" + "a"*64
        with self.assertRaisesRegex(ValueError, "prompt digest"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)
        value["plan_id"] = "r512-" + "a"*64
        with self.assertRaisesRegex(ValueError, "prompt digest"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)

    def test_eos_binding_is_passed_and_checked_before_returning_text(self):
        calls = []
        value = dict(receipt(), input_tokens=[0], generated_token_ids=[3],
                     output_sha256=["a" * 64], logits_sha256=["b" * 64])
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        result = run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)
        self.assertEqual(result["output"]["stop_reason"], "eos")
        self.assertEqual(calls[0][-2:], ["2", "3"])
        for invalid in (None, 2, True):
            altered = dict(value, eos_token_id=invalid)
            with self.assertRaisesRegex(ValueError, "tokenizer EOS"):
                run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8,
                           runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(altered).encode(), b""))
        missing = dict(value)
        del missing["eos_token_id"]
        with self.assertRaisesRegex(ValueError, "tokenizer EOS"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8,
                       runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(missing).encode(), b""))
        with self.assertRaisesRegex(ValueError, "invalid tokenizer EOS"):
            run_prompt("A", Tokenizer(), 0, "/engine", ["/input"]*8, runner=runner)

    def test_long_prompt_and_combined_position_bound(self):
        value = dict(receipt(), prompt_tokens=[0]*128, input_tokens=[0]*128,
                     requested_max_new_tokens=1, generated_token_ids=[0],
                     output_sha256=["a"*64]*128, logits_sha256=["b"*64])
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        result = run_prompt("long128", Tokenizer(), 3, "/engine", ["/input"]*8, max_new_tokens=1, runner=runner)
        self.assertEqual(result["output"]["text"], "A")
        with self.assertRaises(ValueError):
            run_prompt("long128", Tokenizer(), 3, "/engine", ["/input"]*8, max_new_tokens=2, runner=runner)
        self.assertEqual(len(calls), 1)
    def test_remote_runner_passes_lease_descriptor_or_refuses_before_spawn(self):
        value = dict(receipt(), prompt_tokens=[0], plan_id="r528-" + "a"*64,
                     prompt_tokens_sha256=hashlib.sha256(struct.pack("<I", 0)).hexdigest())
        calls=[]
        def runner(args, **kwargs):
            calls.append(kwargs)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        read_fd, write_fd = os.pipe()
        try:
            with patch.dict("os.environ", {"VF_RANK3_SSH_HOST":"example", "VF_SHARED_LEASE_FD":str(read_fd)}, clear=True):
                self.assertEqual(_run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)["output"]["text"], "😀")
                self.assertEqual(calls[-1]["pass_fds"], (read_fd,))
            with patch.dict("os.environ", {"VF_RANK3_SSH_HOST":"example"}, clear=True):
                with self.assertRaisesRegex(ValueError, "shared lease fd"):
                    _run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)
            self.assertEqual(len(calls), 1)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_bounded_capture_passes_only_requested_descriptor(self):
        read_fd, write_fd = os.pipe()
        try:
            program = 'import os,sys; os.fstat(int(sys.argv[1])); print("inherited")'
            good = _run_bounded([sys.executable, "-c", program, str(read_fd)], pass_fds=(read_fd,))
            self.assertEqual((good.returncode, good.stdout), (0, b"inherited\n"))
            absent = _run_bounded([sys.executable, "-c", program, str(read_fd)])
            self.assertNotEqual(absent.returncode, 0)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_bounded_capture_refuses_pipe_inherited_after_engine_exit(self):
        # The direct child exits successfully after printing plausible output;
        # its descendant retains the pipes but exits on its own shortly after.
        program = ("import os,time; p=os.fork(); "
                   "time.sleep(0.7) if p==0 else os.write(1,b'{}'); os._exit(0)")
        with patch("adapters.strata_tokenizer.prompt_runner.EXIT_PIPE_GRACE_SECONDS", 0.1, create=True):
            with self.assertRaisesRegex(RuntimeError, "output pipes remain open"):
                _run_bounded([sys.executable, "-c", program])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux executable descriptor required")
    def test_executable_snapshot_survives_path_replacement(self):
        directory = tempfile.mkdtemp(prefix="vf-r549-exec-")
        path = os.path.join(directory, "engine")
        replacement = os.path.join(directory, "replacement")
        shutil.copyfile("/usr/bin/true", path)
        shutil.copyfile("/usr/bin/false", replacement)
        os.chmod(path, 0o700)
        os.chmod(replacement, 0o700)
        with open(path, "rb") as f:
            expected = hashlib.file_digest(f, "sha256").hexdigest()
        with _pinned_executable(path, expected) as (executable, fds):
            os.replace(replacement, path)
            run = subprocess.run([executable], pass_fds=fds, capture_output=True)
        self.assertEqual(run.returncode, 0, "executed replacement instead of verified binary")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux executable descriptor required")
    def test_executable_snapshot_survives_in_place_change_and_is_sealed(self):
        directory = tempfile.mkdtemp(prefix="vf-r549-inplace-")
        path = os.path.join(directory, "engine")
        shutil.copyfile("/usr/bin/true", path)
        os.chmod(path, 0o700)
        with open(path, "rb") as f:
            expected = hashlib.file_digest(f, "sha256").hexdigest()
        with _pinned_executable(path, expected) as (executable, fds):
            with open("/usr/bin/false", "rb") as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            with self.assertRaises(OSError):
                os.write(fds[0], b"tamper")
            self.assertEqual(subprocess.run([executable], pass_fds=fds).returncode, 0)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            with _pinned_executable(path, expected):
                pass

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux executable descriptor required")
    def test_snapshot_descriptor_coexists_with_lease_and_unsealed_is_refused(self):
        directory = tempfile.mkdtemp(prefix="vf-r549-lease-")
        path = os.path.join(directory, "engine")
        shutil.copyfile("/usr/bin/true", path)
        os.chmod(path, 0o700)
        with open(path, "rb") as f:
            expected = hashlib.file_digest(f, "sha256").hexdigest()
        lease = os.open(path, os.O_RDONLY)
        try:
            calls = []
            def runner(args, **kwargs):
                calls.append(kwargs.get("pass_fds"))
                return subprocess.CompletedProcess(args, 0, json.dumps(receipt()).encode(), b"")
            with _pinned_executable(path, expected) as (executable, fds):
                with patch.dict("os.environ", {"VF_RANK2_SSH_HOST":"host", "VF_SHARED_LEASE_FD":str(lease)}, clear=True):
                    run_prompt("A", Tokenizer(), 3, executable, ["/input"] * 8, runner=runner, binary_fd=fds[0])
                self.assertEqual(calls, [(fds[0], lease)])
            with self.assertRaisesRegex(ValueError, "not sealed"):
                run_prompt("A", Tokenizer(), 3, f"/proc/self/fd/{lease}", ["/input"] * 8, runner=runner, binary_fd=lease)
        finally:
            os.close(lease)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux executable descriptor required")
    def test_executable_snapshot_rejects_script_symlink_and_missing_execute_permission(self):
        directory = tempfile.mkdtemp(prefix="vf-r549-refuse-")
        script = os.path.join(directory, "script")
        with open(script, "wb") as f:
            f.write(b"#!/bin/sh\nexit 0\n")
        os.chmod(script, 0o700)
        with open(script, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        with self.assertRaisesRegex(ValueError, "must be ELF"):
            with _pinned_executable(script, digest):
                pass
        target = os.path.join(directory, "engine")
        shutil.copyfile("/usr/bin/true", target)
        os.chmod(target, 0o600)
        with open(target, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        with self.assertRaisesRegex(ValueError, "invalid engine binary"):
            with _pinned_executable(target, digest):
                pass
        os.chmod(target, 0o700)
        link = os.path.join(directory, "link")
        os.symlink(target, link)
        with self.assertRaises(OSError):
            with _pinned_executable(link, digest):
                pass

    def test_bounded_capture_does_not_timeout_active_engine(self):
        program = "import time; time.sleep(0.35); print('done',end='')"
        with patch("adapters.strata_tokenizer.prompt_runner.EXIT_PIPE_GRACE_SECONDS", 0.1):
            result = _run_bounded([sys.executable, "-c", program])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"done")
        self.assertFalse(result.stdout_truncated)
        self.assertFalse(result.stderr_truncated)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_bounded_capture_refuses_inherited_stderr_alone(self):
        program = ("import os,time; p=os.fork(); "
                   "(os.close(1), time.sleep(0.7)) if p==0 else os.write(1,b'{}'); os._exit(0)")
        with patch("adapters.strata_tokenizer.prompt_runner.EXIT_PIPE_GRACE_SECONDS", 0.1):
            with self.assertRaisesRegex(RuntimeError, "output pipes remain open"):
                _run_bounded([sys.executable, "-c", program])

    def test_bounded_capture_drains_both_pipes_and_marks_overflow(self):
        program = ('import sys; sys.stdout.buffer.write(b"x"*200000); sys.stdout.flush(); '
                   'sys.stderr.buffer.write(b"y"*200000); sys.stderr.flush()')
        result = _run_bounded([sys.executable, "-c", program], stdout_limit=1024, stderr_limit=512)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"x"*1024)
        self.assertEqual(result.stderr, b"y"*512)
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.stderr_truncated)
        exact = _run_bounded([sys.executable, "-c", 'print("abc", end="")'], stdout_limit=3)
        self.assertEqual(exact.stdout, b"abc")
        self.assertFalse(exact.stdout_truncated)

    def test_large_complete_receipt_crosses_real_pipe(self):
        program = ('import json; r={"layer_count":48,"run_id":"fresh","workers_closed":96,'
                   '"prompt_tokens":[0]*8,"input_tokens":[0]*39,"generated_token_ids":[0]*32,'
                   '"requested_max_new_tokens":32,"eos_token_id":3,"output_sha256":["a"*64]*39,'
                   '"logits_sha256":["b"*64]*32,"padding":"x"*2000000}; print(json.dumps(r))')
        def runner(args, **kwargs):
            return _run_bounded([sys.executable, "-c", program], **kwargs)
        result = run_prompt("eight", Tokenizer(), 3, "/engine", ["/input"]*8,
                            max_new_tokens=32, runner=runner)
        self.assertEqual(result["output"]["text"], "A"*32)

    def test_truncated_success_is_not_decoded(self):
        value = json.dumps(receipt()).encode()
        def runner(args, **kwargs):
            result = subprocess.CompletedProcess(args, 0, value, b"")
            result.stdout_truncated = True
            return result
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)

    def test_truncated_error_is_never_misclassified(self):
        def runner(args, **kwargs):
            result = subprocess.CompletedProcess(args, 1, b"", b"prefill-greedy qualification refused: worker shutdown receipt\n")
            result.stderr_truncated = True
            return result
        with self.assertRaisesRegex(RuntimeError, "unclassified"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)

    def test_runs_once_and_returns_complete_text(self):
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(receipt()).encode(), b"")
        out = run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(out["output"]["text"], "😀")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-4:], ["48", "0", "2", "3"])

    def test_two_token_prompt_is_prefilled_before_generation(self):
        calls = []
        value = dict(receipt(), prompt_tokens=[0, 1], input_tokens=[0, 1, 0],
                     generated_token_ids=[0, 0], output_sha256=["a" * 64] * 3)
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        out = run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(calls[0][-3:], ["0,1", "2", "3"])
        self.assertEqual(out["output"]["text"], "AA")
        bad = dict(value, input_tokens=[0, 0, 0])
        with self.assertRaises(ValueError):
            run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8,
                       runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(bad).encode(), b""))

    def test_four_tokens_and_incomplete_receipt_are_checked(self):
        calls = []
        value = dict(receipt(), prompt_tokens=[0, 1], input_tokens=[0, 1, 0, 0, 0],
                     requested_max_new_tokens=4, generated_token_ids=[0, 0, 0, 0],
                     output_sha256=["a" * 64] * 5, logits_sha256=["b" * 64] * 4)
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        out = run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=4, runner=runner)
        self.assertEqual(out["output"]["text"], "AAAA")
        self.assertEqual(calls[0][-4:], ["48", "0,1", "4", "3"])
        short = dict(value, input_tokens=[0, 1], generated_token_ids=[0],
                     output_sha256=["a" * 64] * 2, logits_sha256=["b" * 64])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=4,
                       runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(short).encode(), b""))
        with self.assertRaises(ValueError):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=129, runner=runner)

    def test_32_outputs_and_39_positions_are_bounded_and_atomic(self):
        value = dict(receipt(), prompt_tokens=[0]*8, input_tokens=[0]*39,
                     requested_max_new_tokens=32, generated_token_ids=[0]*32,
                     output_sha256=["a" * 64]*39, logits_sha256=["b" * 64]*32)
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        out = run_prompt("eight", Tokenizer(), 3, "/engine", ["/input"]*8, max_new_tokens=32, runner=runner)
        self.assertEqual(out["output"]["text"], "A"*32)
        incomplete = dict(value, generated_token_ids=[0]*31, input_tokens=[0]*38,
                          output_sha256=["a"*64]*38, logits_sha256=["b"*64]*31)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            run_prompt("eight", Tokenizer(), 3, "/engine", ["/input"]*8, max_new_tokens=32,
                       runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(incomplete).encode(), b""))

    def test_128_outputs_and_positions_are_bounded_and_atomic(self):
        value = dict(receipt(), prompt_tokens=[0], input_tokens=[0]*128,
                     requested_max_new_tokens=128, generated_token_ids=[0]*128,
                     output_sha256=["a"*64]*128, logits_sha256=["b"*64]*128)
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        out = run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, max_new_tokens=128, runner=runner)
        self.assertEqual(out["output"]["text"], "A"*128)
        incomplete = dict(value, generated_token_ids=[0]*127, input_tokens=[0]*127,
                          output_sha256=["a"*64]*127, logits_sha256=["b"*64]*127)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, max_new_tokens=128,
                       runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(incomplete).encode(), b""))

    def test_eight_token_bound_and_prompt_bound(self):
        value = dict(receipt(), prompt_tokens=[0, 1], input_tokens=[0, 1] + [0]*7,
                     requested_max_new_tokens=8, generated_token_ids=[0]*8,
                     output_sha256=["a" * 64] * 9, logits_sha256=["b" * 64] * 8)
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        out = run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=8, runner=runner)
        self.assertEqual(out["output"]["text"], "A" * 8)
        prefill = dict(receipt(), prompt_tokens=[0]*8, input_tokens=[0]*8,
                       requested_max_new_tokens=1, generated_token_ids=[0],
                       output_sha256=["a" * 64] * 8, logits_sha256=["b" * 64])
        out = run_prompt("eight", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=1,
                         runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(prefill).encode(), b""))
        self.assertEqual(out["prompt_token_ids"], [0]*8)

    def test_refuses_prompt_before_spawning_and_never_retries_failure(self):
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 1, b"partial", b"failure")
        for prompt in ("", "too many", "special"):
            with self.assertRaises(ValueError):
                run_prompt(prompt, Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(RuntimeError, "do not replay"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(len(calls), 1)

    def test_rejects_receipt_for_different_prompt(self):
        bad = dict(receipt(), input_tokens=[4, 1])
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(bad).encode(), b"")
        with self.assertRaises(ValueError):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)

    def test_failure_diagnostic_is_allowlisted_and_never_returns_partial_output(self):
        cases = [
            (b"prefill-greedy qualification refused: INVALID_INPUT: no rank has expert payload capacity\n", "rank_payload_capacity"),
            (b"prefill-greedy qualification refused: INVALID_INPUT: expert payload budget exceeded\n", "global_payload_capacity"),
            (b"prefill-greedy qualification refused: INVALID_INPUT: eviction target cannot be met safely\n", "eviction_target"),
            (b"prefill-greedy qualification refused: INVALID_INPUT: eviction limit\n", "eviction_limit"),
            (b"prefill-greedy qualification refused: worker shutdown receipt\n", "worker_shutdown"),
            (b"prefill-greedy qualification refused: INVALID_INPUT: no rank has expert payload capacity SECRET\n", "unclassified"),
            (b"private path or secret", "unclassified"),
        ]
        for stderr, category in cases:
            calls = []
            def runner(args, **kwargs):
                calls.append(args)
                return subprocess.CompletedProcess(args, 1, b"partial secret", stderr)
            with self.subTest(category=category), self.assertRaises(RuntimeError) as caught:
                run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
            self.assertIn(category, str(caught.exception))
            self.assertNotIn("secret", str(caught.exception).lower())
            self.assertEqual(len(calls), 1)

    def test_real_subprocess_reports_known_failure_without_stdout(self):
        def runner(args, **kwargs):
            program = ('import sys; print("private partial output"); '
                       'print("prefill-greedy qualification refused: INVALID_INPUT: no rank has expert payload capacity", file=sys.stderr); '
                       'sys.exit(1)')
            return subprocess.run([sys.executable, "-c", program], **kwargs)
        with self.assertRaises(RuntimeError) as caught:
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertIn("rank_payload_capacity", str(caught.exception))
        self.assertNotIn("private", str(caught.exception))

    def test_signal_status_is_preserved_without_exposing_stderr(self):
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, -9, b"secret stdout", b"secret stderr")
        with self.assertRaises(RuntimeError) as caught:
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertIn("unclassified, exit=-9", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
