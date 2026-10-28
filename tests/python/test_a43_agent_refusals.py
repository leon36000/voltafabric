"""Phase A43: pin each `or` operand of the refusal guards in scripts/vf_agents.py.

Every case feeds an input for which exactly one operand of one guard is true and
asserts the exception type and its exact message, so a neighbouring check that
refuses with a different message cannot mask a disabled operand.

No network: Gateway traffic goes to an in-process fake opener; nothing reaches a
real host. Every directory a test writes (including `.runs/`) lives under a
temporary root.
"""

from collections.abc import Mapping
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import vf_agents  # noqa: E402
from vf_agents import (  # noqa: E402
    MAX_RESPONSE,
    Gateway,
    Ledger,
    RequestFailure,
    parse_proposal,
    safe_context,
    validate_queue,
)

ALIAS = "proxy/exact-alias"


def task(name="t1", **changes):
    t = {
        "id": name,
        "model_key": "reviewer",
        "objective": "Inspect one contract.",
        "files": ["src/a.rs"],
        "write_set": [],
    }
    t.update(changes)
    return t


def proposal(**changes):
    p = {
        "summary": "A bounded proposal.",
        "findings": [],
        "patch": "",
        "commands_to_run": ["cargo test --locked"],
        "verification": "NOT_EXECUTED",
    }
    p.update(changes)
    return p


def finding(**changes):
    f = {"severity": "info", "path": "src/a.rs", "line": 1, "claim": "x"}
    f.update(changes)
    return f


class FrozenMapping(Mapping):
    """A read-only mapping that is not a dict: it passes every key/value check, so
    only an `isinstance(..., dict)` operand can refuse it."""

    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self.body if n < 0 else self.body[:n]


def completion_body(usage=None, content=None):
    body = {
        "model": "upstream-actual",
        "choices": [
            {
                "message": {
                    "content": json.dumps(proposal()) if content is None else content
                }
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return json.dumps(body).encode()


class FakeOpener:
    """In-process stand-in for the urllib opener; records requests, never connects."""

    def __init__(self, models_body=None, chat_body=None):
        self.models_body = (
            json.dumps({"data": [{"id": ALIAS}]}).encode()
            if models_body is None
            else models_body
        )
        self.chat_body = (
            completion_body({"prompt_tokens": 20, "completion_tokens": 30})
            if chat_body is None
            else chat_body
        )
        self.requests = []

    def open(self, req, timeout=None):
        self.requests.append((req.full_url, req.data))
        if req.full_url.endswith("/models"):
            return FakeResponse(self.models_body)
        if req.full_url.endswith("/chat/completions"):
            return FakeResponse(self.chat_body)
        raise AssertionError("unexpected fake gateway path " + req.full_url)

    def posts(self):
        return [r for r in self.requests if r[1] is not None]


class RefusalCase(unittest.TestCase):
    def assertRefused(self, exc_type, message, fn, *args, **kwargs):
        with self.assertRaises(exc_type) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(str(ctx.exception), message)
        return ctx.exception


class PathRefusalTests(RefusalCase):
    def test_invalid_project_relative_path(self):
        # 343 non-string, 344 empty, 345 backslash, 346 colon.
        for name in (5, "", "src\\a.rs", "src/a:b.rs"):
            with self.subTest(name=name):
                self.assertRefused(
                    ValueError, "invalid project-relative path", vf_agents._path, name
                )

    def test_non_canonical_or_absolute_path(self):
        # 347 absolute, 348 parent component, 349 normalised-away component.
        for name in ("/src/a.rs", "src/../a.rs", "src//a.rs", "src/./a.rs"):
            with self.subTest(name=name):
                self.assertRefused(
                    ValueError,
                    "path must be canonical and relative",
                    vf_agents._path,
                    name,
                )

    def test_private_path_components(self):
        # 378 blocked dir (case-folded), 379 .env prefix, 380 secret, 381 credential,
        # 382 id_rsa prefix; every name has an allowed extension.
        for name in (
            "target/a.rs",
            "NODE_MODULES/a.json",
            ".envrc.txt",
            "my_secret.json",
            "src/Credentials.toml",
            "id_rsa.txt",
        ):
            with self.subTest(name=name):
                self.assertRefused(
                    ValueError,
                    "private path excluded from worker context",
                    vf_agents._path,
                    name,
                )

    def test_unsupported_extension(self):
        # 350
        for name in ("src/a.exe", "Makefile"):
            with self.subTest(name=name):
                self.assertRefused(
                    ValueError, "unsupported source extension", vf_agents._path, name
                )


class ContextRefusalTests(RefusalCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src/a.rs").write_text("fn main() {}\n")

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, paths, **kwargs):
        self.assertRefused(
            ValueError,
            "invalid bounded context list",
            safe_context,
            self.root,
            paths,
            **kwargs,
        )

    def test_paths_must_be_a_list(self):  # 351
        self.refused(("src/a.rs",))

    def test_paths_must_be_nonempty(self):  # 352
        self.refused([])

    def test_at_most_64_paths(self):  # 353
        names = []
        for i in range(65):
            (self.root / f"src/f{i}.rs").write_text("x")
            names.append(f"src/f{i}.rs")
        self.assertEqual(len(safe_context(self.root, names[:64])), 64)
        self.refused(names)

    def test_every_path_must_be_a_string(self):  # 354
        self.refused(["src/a.rs", 5])

    def test_max_bytes_bounds(self):  # 356
        for max_bytes in (0, 262145):
            with self.subTest(max_bytes=max_bytes):
                self.refused(["src/a.rs"], max_bytes=max_bytes)

    def test_symlink_to_outside_file(self):  # 383
        with tempfile.TemporaryDirectory() as d:
            outside = Path(d) / "outside.rs"
            outside.write_text("fn leaked() {}\n")
            (self.root / "src/link.rs").symlink_to(outside)
            self.assertRefused(
                ValueError,
                "context escapes root or is not a file",
                safe_context,
                self.root,
                ["src/link.rs"],
            )

    def test_directory_is_not_a_file(self):  # 384
        (self.root / "src/dir.rs").mkdir()
        self.assertRefused(
            ValueError,
            "context escapes root or is not a file",
            safe_context,
            self.root,
            ["src/dir.rs"],
        )

    def test_nul_byte_in_valid_utf8(self):
        # 386: NUL is valid UTF-8, so only the binary check refuses it (the existing
        # b'\xff\x00' case is also refused by the UTF-8 decode).
        (self.root / "src/b.rs").write_bytes(b"a\x00b")
        self.assertRefused(
            ValueError,
            "context byte budget exceeded or binary file",
            safe_context,
            self.root,
            ["src/b.rs"],
        )


class QueueRefusalTests(RefusalCase):
    def test_queue_container(self):
        # 357 tuple; 358 empty and 513 entries.
        many = [task(f"t{i}") for i in range(513)]
        for tasks in ((task(),), [], many):
            with self.subTest(n=len(tasks), kind=type(tasks).__name__):
                self.assertRefused(
                    ValueError,
                    "queue must contain 1..512 bounded tasks",
                    validate_queue,
                    tasks,
                )
        self.assertEqual(len(validate_queue(many[:512])), 512)

    def test_task_must_be_a_dict(self):  # 387: right keys, but not a dict
        self.assertRefused(
            ValueError,
            "unknown or missing task fields",
            validate_queue,
            [FrozenMapping(task())],
        )

    def test_task_id_must_be_a_string(self):  # 389
        self.assertRefused(
            ValueError, "invalid/duplicate task id", validate_queue, [task(5)]
        )

    def test_model_key(self):
        # 392 non-string, 393 pattern mismatch.
        for key in (5, "../reviewer", ""):
            with self.subTest(key=key):
                self.assertRefused(
                    ValueError,
                    "invalid model role key",
                    validate_queue,
                    [task(model_key=key)],
                )

    def test_objective(self):
        # 394 non-string, 396 longer than 12000.
        self.assertEqual(len(validate_queue([task(objective="x" * 12000)])), 1)
        for objective in (5, "x" * 12001):
            with self.subTest(objective=str(objective)[:8]):
                self.assertRefused(
                    ValueError,
                    "objective must be bounded and nonempty",
                    validate_queue,
                    [task(objective=objective)],
                )

    def test_task_path_lists(self):
        # 433 not a list, 434 more than 64, 435 non-string entry, 436 duplicates;
        # checked for both files and write_set.
        many = [f"src/f{i}.rs" for i in range(65)]
        for field in ("files", "write_set"):
            for names in (
                ("src/a.rs",),
                many,
                ["src/a.rs", 5],
                ["src/a.rs", "src/a.rs"],
            ):
                with self.subTest(field=field, names=str(names)[:30]):
                    self.assertRefused(
                        ValueError,
                        "invalid task paths",
                        validate_queue,
                        [task(**{field: names})],
                    )
        self.assertEqual(
            len(validate_queue([task(files=many[:64], write_set=many[:64])])), 1
        )

    def test_explicit_context_required(self):  # 397
        self.assertRefused(
            ValueError,
            "explicit source context required",
            validate_queue,
            [task(files=[])],
        )


class ProposalRefusalTests(RefusalCase):
    def test_proposal_must_be_a_dict(self):  # 359
        self.assertRefused(
            ValueError,
            "worker must return the strict proposal-only schema",
            parse_proposal,
            FrozenMapping(proposal()),
        )

    def test_summary_and_patch(self):
        # 362 summary type, 363 summary length, 364 patch type, 365 patch length.
        parse_proposal(proposal(summary="x" * 12000, patch="x" * 300000))
        for change in (
            {"summary": 5},
            {"summary": "x" * 12001},
            {"patch": 5},
            {"patch": "x" * 300001},
        ):
            with self.subTest(
                field=list(change)[0], value=str(list(change.values())[0])[:8]
            ):
                self.assertRefused(
                    ValueError,
                    "invalid bounded summary/patch",
                    parse_proposal,
                    proposal(**change),
                )

    def test_findings_and_commands_lists(self):
        # 366 findings type, 367 >128 findings, 368 commands type, 369 >32 commands.
        parse_proposal(proposal(findings=[finding()] * 128, commands_to_run=["x"] * 32))
        cases = (
            {"findings": ()},
            {"findings": [finding()] * 129},
            {"commands_to_run": ("x",)},
            {"commands_to_run": ["x"] * 33},
        )
        for change in cases:
            with self.subTest(change=str(change)[:40]):
                self.assertRefused(
                    ValueError,
                    "invalid findings/commands list",
                    parse_proposal,
                    proposal(**change),
                )

    def test_command_text(self):  # 370
        parse_proposal(proposal(commands_to_run=["x" * 4096]))
        for commands in ([5], ["x" * 4097]):
            with self.subTest(command=str(commands[0])[:8]):
                self.assertRefused(
                    ValueError,
                    "invalid command text",
                    parse_proposal,
                    proposal(commands_to_run=commands),
                )

    def test_invalid_finding(self):
        # 398 non-dict mapping, 399 extra key, 400 unknown severity (valid path/line).
        cases = (
            FrozenMapping(finding()),
            dict(finding(), extra=1),
            finding(severity="great"),
        )
        for f in cases:
            with self.subTest(finding=repr(f)[:40]):
                self.assertRefused(
                    ValueError,
                    "invalid finding",
                    parse_proposal,
                    proposal(findings=[f]),
                )

    def test_finding_evidence_location(self):
        # 401 bool/float line, 402 line < 1, 403 claim type, 404 claim length.
        parse_proposal(proposal(findings=[finding(claim="x" * 12000)]))
        cases = (
            {"line": True},
            {"line": 1.0},
            {"line": 0},
            {"claim": 5},
            {"claim": "x" * 12001},
        )
        for change in cases:
            with self.subTest(change=str(change)[:30]):
                self.assertRefused(
                    ValueError,
                    "invalid finding evidence location",
                    parse_proposal,
                    proposal(findings=[finding(**change)]),
                )


class LedgerRefusalTests(RefusalCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def test_ledger_limits(self):
        # 405 max_requests type, 406 range, 407 max_output_tokens type, 408 < 1.
        cases = (
            {"max_requests": True},
            {"max_requests": 8.0},
            {"max_requests": 0},
            {"max_requests": 4097},
            {"max_output_tokens": True},
            {"max_output_tokens": 10.0},
            {"max_output_tokens": 0},
        )
        for kwargs in cases:
            with self.subTest(**{k: repr(v) for k, v in kwargs.items()}):
                self.assertRefused(
                    ValueError, "invalid ledger limits", Ledger, self.path, **kwargs
                )
        self.assertFalse(
            self.path.exists(), "a refused ledger must not create its database"
        )

    def test_invalid_reservation(self):
        # 409 task id pattern, 410 empty fingerprint, 411 bool/float tokens, 412 range.
        ledger = Ledger(self.path)
        cases = (
            ("../x", "h", 128),
            ("", "h", 128),
            ("a" * 65, "h", 128),
            ("a", "", 128),
            ("a", "h", True),
            ("a", "h", 128.0),
            ("a", "h", 0),
            ("a", "h", 65537),
        )
        for args in cases:
            with self.subTest(args=repr(args)[:30]):
                self.assertRefused(
                    ValueError, "invalid reservation", ledger.reserve, *args
                )
        self.assertEqual(ledger.rows(), [])

    def test_send_requires_reserved_state(self):  # 442
        ledger = Ledger(self.path)
        self.assertRefused(
            ValueError, "invalid send state transition", ledger.sent, "ghost"
        )
        ledger.reserve("a", "h", 100)
        ledger.sent("a")
        self.assertRefused(
            ValueError, "invalid send state transition", ledger.sent, "a"
        )

    def test_terminal_status_allowlist(
        self,
    ):  # 413: task is in 'sent', so only the status is wrong
        ledger = Ledger(self.path)
        ledger.reserve("a", "h", 100)
        ledger.sent("a")
        for status in ("sent", "reserved", "COMPLETED"):
            with self.subTest(status=status):
                self.assertRefused(
                    ValueError, "invalid terminal status", ledger.finish, "a", status
                )
        self.assertEqual(ledger.rows()[0]["state"], "sent")


class GatewayConstructionRefusalTests(RefusalCase):
    def test_invalid_gateway_url(self):
        # 414 scheme, 415 no host, 417 password without username, 419 fragment.
        for url in (
            "ftp://proxy.example/v1",
            "https:///v1",
            "https://:pw@proxy.example",
            "https://proxy.example/#frag",
        ):
            with self.subTest(url=url):
                self.assertRefused(
                    ValueError,
                    "invalid explicit gateway URL",
                    Gateway,
                    url,
                    "dummy-test-key",
                )

    def test_http_requires_private_literal_even_with_opt_in(self):  # 445
        for url in ("http://8.8.8.8", "http://proxy.example"):
            with self.subTest(url=url):
                self.assertRefused(
                    ValueError,
                    "HTTP allowed only by explicit opt-in on a literal private/loopback address",
                    Gateway,
                    url,
                    "dummy-test-key",
                    allow_local_http=True,
                )

    def test_key_and_timeout(self):
        # 421 non-string key, 422 empty key, 423 timeout outside (0, 300].
        cases = (
            {"key": b"dummy-test-key"},
            {"key": 5},
            {"key": ""},
            {"timeout": 0},
            {"timeout": -1},
            {"timeout": 300.5},
        )
        Gateway("https://proxy.example", "dummy-test-key", timeout=300)
        for change in cases:
            kwargs = {"key": "dummy-test-key", **change}
            with self.subTest(change=repr(change)):
                self.assertRefused(
                    ValueError,
                    "missing existing gateway key or invalid timeout",
                    Gateway,
                    "https://proxy.example",
                    **kwargs,
                )


class GatewayResponseRefusalTests(RefusalCase):
    def gateway(self, **opener_kwargs):
        gateway = Gateway("https://proxy.example", "dummy-test-key")
        gateway.opener = FakeOpener(**opener_kwargs)
        return gateway

    def assertFailure(self, message, uncertain, fn, *args):
        exc = self.assertRefused(RequestFailure, message, fn, *args)
        self.assertIs(exc.uncertain, uncertain)

    def test_request_byte_limit_blocks_before_sending(self):  # 424
        gateway = self.gateway()
        big = [{"role": "user", "content": "x" * (512 * 1024)}]
        self.assertRefused(
            ValueError, "request byte limit", gateway.propose, ALIAS, big, 128
        )
        self.assertEqual(gateway.opener.requests, [])

    def test_response_too_large(self):  # 425: oversized but otherwise valid JSON
        body = json.dumps({"data": []}).encode() + b" " * MAX_RESPONSE
        self.assertFailure(
            "RESPONSE_TOO_LARGE", True, self.gateway(models_body=body).models
        )

    def test_response_must_be_an_object(self):  # 426
        self.assertFailure(
            "INVALID_RESPONSE_OBJECT", True, self.gateway(models_body=b"[]").models
        )

    def test_model_discovery_list(self):
        # 427 data not a list, 428 more than 8192 rows.
        rows = [{"id": f"m{i}"} for i in range(8193)]
        self.assertEqual(
            len(
                self.gateway(
                    models_body=json.dumps({"data": rows[:8192]}).encode()
                ).models()
            ),
            8192,
        )
        for body in ({"data": {}}, {}, {"data": rows}):
            with self.subTest(body=str(body)[:20]):
                gateway = self.gateway(models_body=json.dumps(body).encode())
                self.assertFailure("INVALID_MODEL_DISCOVERY", False, gateway.models)

    def test_model_ids(self):
        # 446 row not an object, 447 id not a string, 448 empty id.
        for row in (["id"], {"id": 5}, {"id": ""}):
            with self.subTest(row=row):
                gateway = self.gateway(
                    models_body=json.dumps({"data": [{"id": ALIAS}, row]}).encode()
                )
                self.assertFailure("INVALID_MODEL_ID", False, gateway.models)

    def test_usage_receipt(self):
        # 449 usage not an object, 450 missing/bool/negative/float token counts.
        cases = (
            [],
            5,
            {"prompt_tokens": 1},
            {"prompt_tokens": -1, "completion_tokens": 1},
            {"prompt_tokens": 1, "completion_tokens": True},
            {"prompt_tokens": 1.0, "completion_tokens": 1},
        )
        for usage in cases:
            with self.subTest(usage=usage):
                gateway = self.gateway(chat_body=completion_body(usage))
                self.assertFailure(
                    "INVALID_USAGE_RECEIPT", True, gateway.propose, ALIAS, [], 128
                )

    def test_completion_tokens_over_reservation(self):  # 451
        ok = self.gateway(
            chat_body=completion_body({"prompt_tokens": 1, "completion_tokens": 128})
        )
        self.assertEqual(ok.propose(ALIAS, [], 128)["usage"]["completion_tokens"], 128)
        gateway = self.gateway(
            chat_body=completion_body({"prompt_tokens": 1, "completion_tokens": 129})
        )
        self.assertFailure(
            "PROVIDER_EXCEEDED_OUTPUT_RESERVATION",
            True,
            gateway.propose,
            ALIAS,
            [],
            128,
        )


class ExactMessageTests(RefusalCase):
    """Guards that test_agents.py already kills with type-only assertRaises(ValueError).
    These pin the exact message with one operand true, so a later refusal cannot
    stand in for them."""

    def test_context_duplicates_and_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src").mkdir()
            (root / "src/a.rs").write_text("fn main() {}\n")
            # 355
            self.assertRefused(
                ValueError,
                "invalid bounded context list",
                safe_context,
                root,
                ["src/a.rs", "src/a.rs"],
            )
            # 385: plain UTF-8 text without NUL, just over the byte budget.
            self.assertRefused(
                ValueError,
                "context byte budget exceeded or binary file",
                safe_context,
                root,
                ["src/a.rs"],
                max_bytes=4,
            )

    def test_proposal_schema(self):
        # 360 extra key, 360 missing key, 361 claimed verification.
        missing = proposal()
        del missing["patch"]
        for value in (
            proposal(tool_calls=["shell"]),
            missing,
            proposal(verification="PASSED"),
        ):
            with self.subTest(value=sorted(value)):
                self.assertRefused(
                    ValueError,
                    "worker must return the strict proposal-only schema",
                    parse_proposal,
                    value,
                )

    def test_queue_task_fields_and_ids(self):
        missing = task()
        del missing["write_set"]
        # 388 extra and missing fields.
        for t in (dict(task(), shell="x"), missing):
            with self.subTest(fields=sorted(t)):
                self.assertRefused(
                    ValueError, "unknown or missing task fields", validate_queue, [t]
                )
        # 390 pattern mismatch, 391 duplicate id.
        for tasks in ([task("../x")], [task("a"), task("a")]):
            with self.subTest(ids=[t["id"] for t in tasks]):
                self.assertRefused(
                    ValueError, "invalid/duplicate task id", validate_queue, tasks
                )
        # 395 whitespace-only objective.
        self.assertRefused(
            ValueError,
            "objective must be bounded and nonempty",
            validate_queue,
            [task(objective=" \n\t")],
        )
        # 437 two tasks proposing the same write.
        self.assertRefused(
            ValueError,
            "conflicting proposed writers; separate the waves",
            validate_queue,
            [task("a", write_set=["src/x.rs"]), task("b", write_set=["src/x.rs"])],
        )

    def test_ledger_state_machine(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.sqlite"
            Ledger(path, max_requests=3)
            # 438
            self.assertRefused(
                ValueError,
                "existing run budget cannot silently change",
                Ledger,
                path,
                max_requests=30,
            )
            ledger = Ledger(path, max_requests=3)  # same limits: reopening is allowed
            ledger.reserve("a", "h", 100)
            # 439
            self.assertRefused(
                ValueError,
                "task already reserved; inspect receipt, do not silently replay",
                ledger.reserve,
                "a",
                "other",
                100,
            )
            # 443: not yet sent; then already terminal.
            self.assertRefused(
                ValueError,
                "terminal update requires sent state",
                ledger.finish,
                "a",
                "completed",
            )
            ledger.sent("a")
            ledger.finish("a", "unknown")
            self.assertRefused(
                ValueError,
                "terminal update requires sent state",
                ledger.finish,
                "a",
                "completed",
            )
        with tempfile.TemporaryDirectory() as d:
            # 440: request count exhausted while tokens remain.
            ledger = Ledger(
                Path(d) / "ledger.sqlite", max_requests=1, max_output_tokens=1000
            )
            ledger.reserve("a", "h", 100)
            self.assertRefused(
                ValueError,
                "run reservation budget exhausted",
                ledger.reserve,
                "b",
                "h",
                100,
            )
        with tempfile.TemporaryDirectory() as d:
            # 441: tokens exhausted while requests remain.
            ledger = Ledger(
                Path(d) / "ledger.sqlite", max_requests=10, max_output_tokens=200
            )
            ledger.reserve("a", "h", 128)
            self.assertRefused(
                ValueError,
                "run reservation budget exhausted",
                ledger.reserve,
                "b",
                "h",
                128,
            )

    def test_gateway_url_policy(self):
        # 416 username, 418 query.
        for url in ("https://user@proxy.example", "https://proxy.example/?key=x"):
            with self.subTest(url=url):
                self.assertRefused(
                    ValueError,
                    "invalid explicit gateway URL",
                    Gateway,
                    url,
                    "dummy-test-key",
                )
        # 420 (hostname is case-folded).
        for url in (
            "https://openrouter.ai/api",
            "https://API.OpenAI.com/v1",
            "https://api.anthropic.com",
            # A43 fix: the root-dot spelling and provider subdomains were accepted.
            "https://openrouter.ai./api",
            "https://eu.api.openai.com/v1",
            "https://api.anthropic.com./v1",
        ):
            with self.subTest(url=url):
                self.assertRefused(
                    ValueError,
                    "direct provider endpoint forbidden; use the existing LiteLLM gateway",
                    Gateway,
                    url,
                    "dummy-test-key",
                )
        # The match is label-bounded: a lookalike domain is not a provider.
        Gateway("https://notopenai.com/v1", "dummy-test-key")
        # 444: private literal, but no opt-in.
        for url in ("http://127.0.0.1:9", "http://localhost:9", "http://10.0.0.1"):
            with self.subTest(url=url):
                self.assertRefused(
                    ValueError,
                    "HTTP allowed only by explicit opt-in on a literal private/loopback address",
                    Gateway,
                    url,
                    "dummy-test-key",
                )


class JsonFileRefusalTests(RefusalCase):
    def test_oversized_config_or_queue(self):  # 371: valid JSON, only too large
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "queue.json"
            path.write_bytes(b"[]" + b" " * (2 * 1024 * 1024 - 1))
            self.assertRefused(
                ValueError, "config/queue too large", vf_agents._json_file, path
            )


class MainRefusalTests(RefusalCase):
    """Runs vf_agents.main() in-process; the opener is a FakeOpener, so even a
    mutated run that gets past a guard never leaves the process."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "root"
        (self.root / "src").mkdir(parents=True)
        (self.root / "src/a.rs").write_text("fn a() {}")
        self.cfg = {
            "aliases": {"reviewer": ALIAS},
            "workers": 2,
            "max_requests": 4,
            "max_reserved_output_tokens": 1024,
            "max_tokens": 128,
            "context_byte_limit": 1024,
        }
        self.config = Path(self.tmp.name) / "config.json"
        self.queue = Path(self.tmp.name) / "queue.json"
        self.write_config()
        self.queue.write_text(json.dumps([task()]))
        self.opener = FakeOpener()

    def tearDown(self):
        self.tmp.cleanup()

    def write_config(self, cfg=None):
        self.config.write_text(json.dumps(self.cfg if cfg is None else cfg))

    def main(self, *extra):
        argv = [
            "vf_agents.py",
            "run",
            "--root",
            str(self.root),
            "--config",
            str(self.config),
            "--queue",
            str(self.queue),
            *extra,
        ]
        env = {
            "VF_LITELLM_URL": "http://127.0.0.1:9",
            "LITELLM_API_KEY": "dummy-test-key",
        }
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict(os.environ, env),
            mock.patch.object(
                vf_agents.urllib.request, "build_opener", lambda *a, **k: self.opener
            ),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            code = vf_agents.main()
        return code, out.getvalue()

    LIVE = ("--execute", "--authorized-spend", "--allow-local-http")

    def test_fixture_dry_run_and_live_succeed(self):
        code, out = self.main()
        self.assertEqual((code, json.loads(out)["mode"]), (0, "DRY_RUN_NO_NETWORK"))
        self.assertEqual(self.opener.requests, [])
        code, out = self.main(*self.LIVE)
        self.assertEqual((code, json.loads(out)["mode"]), (0, "LIVE_PROPOSALS_ONLY"))
        self.assertEqual(len(self.opener.posts()), 1)

    def test_config_schema(self):
        # 372 a JSON list of exactly the expected keys, 373 extra key, 374 aliases not an object.
        cases = (
            list(self.cfg),
            dict(self.cfg, extra=1),
            dict(self.cfg, aliases=[["reviewer", ALIAS]]),
        )
        for cfg in cases:
            with self.subTest(cfg=str(cfg)[:40]):
                self.write_config(cfg)
                self.assertRefused(ValueError, "invalid config schema", self.main)

    def test_bounded_config_values(self):
        # 429 non-int (bool/float), 430 out of range; each field alone.
        bounds = {
            "workers": (1, 32),
            "max_requests": (1, 4096),
            "max_reserved_output_tokens": (1, 100000000),
            "max_tokens": (1, 65536),
            "context_byte_limit": (1, 262144),
        }
        for name, (low, high) in bounds.items():
            for value in (True, float(low), low - 1, high + 1):
                with self.subTest(name=name, value=value):
                    self.write_config(dict(self.cfg, **{name: value}))
                    self.assertRefused(
                        ValueError, "invalid bounded config: " + name, self.main
                    )

    def test_live_requires_authorized_spend(self):  # 375
        self.assertRefused(
            ValueError,
            "live calls require existing authorization and --authorized-spend; verify proxy financial limits first",
            self.main,
            "--execute",
            "--allow-local-http",
        )
        self.assertEqual(self.opener.requests, [])
        self.assertFalse((self.root / ".runs").exists())

    def test_run_id_pattern(self):  # 376: '../x' still resolves inside the root
        self.assertRefused(
            ValueError, "invalid run id", self.main, *self.LIVE, "--run-id", "../x"
        )
        self.assertEqual(self.opener.requests, [])
        self.assertFalse((self.root / "x").exists())

    def test_run_directory_escape(self):  # 377
        with tempfile.TemporaryDirectory() as d:
            outside = Path(d)
            (self.root / ".runs").symlink_to(outside, target_is_directory=True)
            self.assertRefused(
                ValueError, "run directory escapes project", self.main, *self.LIVE
            )
            self.assertFalse((outside / "first-wave").exists())
        self.assertEqual(self.opener.requests, [])

    def test_symlinked_ledger_or_report(self):  # 431
        for fixed_name in ("ledger.sqlite", "report.json"):
            with (
                self.subTest(fixed_name=fixed_name),
                tempfile.TemporaryDirectory() as d,
            ):
                run_dir = self.root / ".runs/first-wave"
                run_dir.mkdir(parents=True, exist_ok=True)
                target = Path(d) / "target"
                link = run_dir / fixed_name
                link.symlink_to(target)
                try:
                    self.assertRefused(
                        ValueError,
                        "symlinked ledger/report is forbidden",
                        self.main,
                        *self.LIVE,
                    )
                    self.assertFalse(target.exists())
                finally:
                    link.unlink()
        self.assertEqual(self.opener.requests, [])

    def test_alias_must_be_discovered(self):  # 432
        self.cfg["aliases"]["reviewer"] = "invented"
        self.write_config()
        self.assertRefused(
            ValueError,
            "model alias unresolved or unavailable for reviewer",
            self.main,
            *self.LIVE,
        )
        self.assertEqual(self.opener.posts(), [])
        self.assertEqual(len(self.opener.requests), 1)


if __name__ == "__main__":
    unittest.main()
