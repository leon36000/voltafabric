#!/usr/bin/env python3
"""Guard-clause sweep by mutant schemata: one build, one runtime switch per clause.

rust_mutations.py builds one isolated binary per hand-written mutant, which is right
for a few dozen cases and far too slow for every guard. Here every top-level `||`
operand of an `if COND { return Err(..) }` statement in src/ becomes

    ((operand) && !__vf_mutant(ID))

in a copy of the tree, so VF_MUTANT=ID disables exactly that clause and nothing is
rebuilt between mutants. Stage one runs the whole Rust suite per clause; stage two
(--gate-python) runs the whole CPU gate for each Rust survivor, so the Python
oracles and the c-abi-validation suite also get a chance to notice. Each stage
first runs a control on the unmutated copy under the same limits and stops unless
it passes. Every run is confined to a per-user systemd scope with a memory cap: an
address-space ulimit would also break AddressSanitizer in the gate's FFI step.

With --sites reasons the copy instead rewrites each refusal that cannot be removed,
the closure of an `.ok_or_else(|| ..)` or the argument of a match-arm `=> Err(..)`,
so that VF_MUTANT=ID returns a placeholder error instead. Success paths are
untouched; a mutant dies only if some test reaches that refusal and checks its code
or message, which a bare is_err() does not. --sites reach makes the same sites
panic instead, which separates a refusal no test reaches from one whose reason no
test checks. --sites c-clauses applies the clause rewrite to the C launch
validator (cuda/qpn2_moe_validate.c, which the SM70 library also calls before every
launch); its first stage runs the workspace with c-abi-validation and the C driver
under AddressSanitizer, since the default Rust suite never compiles the validator.
--sites py-raises rewrites the `or` operands of `if ..: raise` guards in the Python
modules on the delivery path (preflight, canary, packager, adapter, agents); its
first stage runs tests/python and tests/adapter with the canonical interpreter.
--sites py-handlers covers what that leaves out, with ids of its own: the operands
of guards whose raise follows other statements, and each translating raise in an
`except` body, which VF_MUTANT=ID turns into a bare re-raise of the original.
--sites sh-launchers covers the two bash launchers of the H1 bundle: each
`: "${NAME:?..}"` check (removed, or weakened to `?` so an empty value passes),
each option of `set -euo pipefail`, and each pattern of a case arm that exits;
the launcher reads VF_MUTANT itself, and the first stage is the Python one.
--sites map-reasons and map-reach, with ids of their own, do for each `.map_err(..)`
translation what reasons and reach do for ok_or_else: the translated error is
replaced by a placeholder of the same type (crate Error or std::io::Error), or the
site panics.
--sites ret-reasons and ret-reach, with ids of their own, do the same for the error
of each `return Err(..)` and each block-tail `Err(..)` that the reasons catalog
does not hold (it takes the match-arm ones): A40 removes the guards in front of most
of them, which proves a test reaches the refusal but not that it checks the reason.
--sites c-codes and c-reach, with ids of their own, do the same for each refusal
`return VF_QPN2_<code>;` of the C launch validator: c-codes returns a code no caller
knows (-99) instead, c-reach aborts there; the first stage is the c-clauses one.
--sites py-reasons and py-reach, with ids of their own, take each `raise <call>(..)` of
the Python delivery-path modules (not the `SystemExit(main())` exits): py-reasons keeps
the exception's class but replaces its message and every instance attribute (a string
becomes "VF_MUTANT", a bool flips), py-reach raises a BaseException no `except
Exception` can absorb. --skip-gate stops after the first stage (no gate control).

The canonical tree is only read. A surviving clause is not automatically a gap: it
may be unreachable or backed by another clause, which a reason-pinned test settles.
"""

import argparse
import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SWITCH = """
#[doc(hidden)]
pub fn __vf_mutant(id: u32) -> bool {
    static ACTIVE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *ACTIVE.get_or_init(|| std::env::var("VF_MUTANT").ok().and_then(|v| v.parse().ok()))
        == Some(id)
}
"""
SKIPPED = {
    "target",
    ".tools",
    ".runs",
    ".worktrees",
    "__pycache__",
    ".ruff_cache",
    "evidence",
}
# Some canonical tests read this receipt; it is the only evidence carried over.
NEEDED_EVIDENCE = ("evidence/continuation_a23/CUDA_COMPILED.json",)
CRATE_ROOTS = ("src/lib.rs", "src/main.rs", "src/bin/vf-rank-worker.rs")
COSMETIC_STEPS = {"rustfmt", "clippy", "c_abi_clippy", "python_format", "python_lint"}


def skip_literal(text, i):
    """Index just past a string/char literal or line comment at i, else i."""
    if text.startswith("//", i):
        return text.index("\n", i)
    if text[i] == '"':
        j = i + 1
        while text[j] != '"':
            j += 2 if text[j] == "\\" else 1
        return j + 1
    if text[i] == "'":
        m = re.match(r"'(\\.|[^\\'])'", text[i:])
        if m:
            return i + m.end()
    return i


def scan_condition(text, start):
    """Return (index of the body brace, operand spans), or (None, []) if no guard."""
    depth, i, spans, operand = 0, start, [], start
    while True:
        j = skip_literal(text, i)
        if j != i:
            i = j
            continue
        c = text[i]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
            if depth < 0:
                return None, []
        elif depth == 0 and text.startswith("=>", i):
            return None, []
        elif c == "{":
            if depth == 0:
                spans.append((operand, i))
                return i, spans
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0 and text.startswith("||", i):
            spans.append((operand, i))
            i += 2
            operand = i
            continue
        i += 1


def guards(text):
    """Yield (start, body brace, operand spans) for refusal guards outside tests."""
    limit = text.find("#[cfg(test)]")
    limit = len(text) if limit < 0 else limit
    for m in re.finditer(r"(?<![\w.])if\s+(?!let\b)", text):
        if m.start() >= limit:
            break
        before = text[text.rfind("\n", 0, m.start()) + 1 : m.start()]
        if before.strip() not in ("", "} else", "else"):
            continue  # match-arm guards, macro patterns and if-expressions
        brace, spans = scan_condition(text, m.end())
        if brace is not None and text[brace + 1 : brace + 400].lstrip().startswith(
            "return Err("
        ):
            yield m.end(), brace, spans


MAP_SITES = ("map-reasons", "map-reach")
RET_SITES = ("ret-reasons", "ret-reach")
MAP_SWITCH = """
/// Calls f(e); the bound lets a closure's parameter type come from e.
#[doc(hidden)]
pub fn __vf_apply<E, F>(e: E, f: impl FnOnce(E) -> F) -> F {
    f(e)
}
#[doc(hidden)]
pub trait __VfReplace {
    fn __vf_replace(self) -> Self;
}
impl __VfReplace for Error {
    fn __vf_replace(self) -> Self {
        Error::new("VF_MUTANT", "refusal reason replaced")
    }
}
impl __VfReplace for std::io::Error {
    fn __vf_replace(self) -> Self {
        std::io::Error::other("VF_MUTANT refusal reason replaced")
    }
}
"""


def closing_paren(text, i):
    """Index of the parenthesis closing the one at i."""
    depth = 0
    while True:
        j = skip_literal(text, i)
        if j != i:
            i = j
            continue
        if text[i] in "([{":
            depth += 1
        elif text[i] in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1


def reason_sites(text):
    """Yield (start, end) of each refusal reason outside tests: the body of an
    `.ok_or_else(|| ..)` closure or the argument of a match-arm `=> Err(..)`."""
    limit = text.find("#[cfg(test)]")
    limit = len(text) if limit < 0 else limit
    for m in re.finditer(r"\.ok_or_else\(|=>\s*(?:return\s+)?Err\(", text):
        if m.start() >= limit:
            break
        open_paren = m.end() - 1
        inner = text[open_paren + 1 : closing_paren(text, open_paren)]
        start = open_paren + 1
        if m.group(0).startswith(".ok_or_else"):
            if not inner.lstrip().startswith("||"):
                continue
            start += inner.index("||") + 2
        body = text[start : open_paren + 1 + len(inner)]
        lead = len(body) - len(body.lstrip())
        yield start + lead, start + lead + len(body.strip())


def map_err_sites(text):
    """Yield (start, end) of the argument of each `.map_err(..)` outside tests."""
    limit = text.find("#[cfg(test)]")
    limit = len(text) if limit < 0 else limit
    for m in re.finditer(r"\.map_err\(", text):
        if m.start() >= limit:
            break
        open_paren = m.end() - 1
        inner = text[open_paren + 1 : closing_paren(text, open_paren)]
        lead = len(inner) - len(inner.lstrip())
        yield open_paren + 1 + lead, open_paren + 1 + lead + len(inner.strip())


def ret_sites(text):
    """Yield (start, end) of the argument of each `return Err(..)` (not after `=>`)
    and each `Err(..)` opening a statement after `{` or `;` outside tests; an Err
    followed by `=>`, `=` or a match guard is a pattern, not a refusal."""
    limit = text.find("#[cfg(test)]")
    limit = len(text) if limit < 0 else limit
    i = 0
    while i < limit:
        j = skip_literal(text, i)
        if j != i:
            i = j
            continue
        if not text.startswith("Err(", i) or re.match(r"[\w:]", text[i - 1]):
            i += 1
            continue
        open_paren = i + 3
        close = closing_paren(text, open_paren)
        prev = text[:i].rstrip()
        after = text[close + 1 :].lstrip()
        returned = re.search(r"(?<![\w=])return$", prev) and not re.search(
            r"=>\s*return$", prev
        )
        pattern = re.match(r"=>|=(?!=)|if\b", after)
        if (returned or prev.endswith(("{", ";"))) and not pattern:
            inner = text[open_paren + 1 : close]
            lead = len(inner) - len(inner.lstrip())
            yield open_paren + 1 + lead, open_paren + 1 + lead + len(inner.strip())
        i = open_paren + 1


C_VALIDATOR = "cuda/qpn2_moe_validate.c"
C_SWITCH = """
static int vf_mutant(int id) {
  static int active = -2;
  if (active == -2) {
    const char *value = getenv("VF_MUTANT");
    active = value ? atoi(value) : -1;
  }
  return active == id;
}
"""
# The C driver the gate runs under AddressSanitizer, rebuilt from the rewritten copy.
C_DRIVER = (
    "gcc -std=c11 -Wall -Wextra -Werror -pedantic -fsanitize=address,undefined "
    "tests/qpn2_moe_validate.c cuda/qpn2_moe_validate.c -o target/validate_ffi "
    "&& target/validate_ffi"
)


def c_guards(text):
    """Yield operand spans of each `if (..)` whose consequent returns."""
    for m in re.finditer(r"(?<![\w])if \(", text):
        open_paren = m.end() - 1
        close = closing_paren(text, open_paren)
        after = text[close + 1 : close + 80].lstrip()
        if not re.match(r"(return\b|\{\s*free\(seen\);\s*return\b)", after):
            continue
        spans, depth, operand, i = [], 0, open_paren + 1, open_paren + 1
        while i < close:
            c = text[i]
            if c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
            elif depth == 0 and text.startswith("||", i):
                spans.append((operand, i))
                i += 2
                operand = i
                continue
            i += 1
        spans.append((operand, close))
        yield spans


C_CODES = ("c-codes", "c-reach")
C_SITES = ("c-clauses", *C_CODES)


def c_returns(text):
    """Yield (start, end) of the code of each refusal `return VF_QPN2_<code>;` in
    vf_qpn2_moe_validate (VF_QPN2_OK is success, the helpers return booleans)."""
    start = text.index("int vf_qpn2_moe_validate(")
    for m in re.finditer(r"\breturn (VF_QPN2_[A-Z]+);", text[start:]):
        if m.group(1) != "VF_QPN2_OK":
            yield start + m.start(1), start + m.end(1)


def rewrite_c(work, catalog, next_id, sites="c-clauses"):
    path = work / C_VALIDATOR
    text = path.read_text()
    edits = []
    for a, b in c_returns(text) if sites in C_CODES else ():
        code = text[a:b]
        catalog.append(
            {
                "id": next_id,
                "file": C_VALIDATOR,
                "line": text[:a].count("\n") + 1,
                "code": code,
            }
        )
        mutant = "-99" if sites == "c-codes" else f"(abort(), {code})"
        edits.append((a, b, f"(vf_mutant({next_id}) ? {mutant} : {code})"))
        next_id += 1
    for spans in c_guards(text) if sites == "c-clauses" else ():
        for a, b in spans:
            operand = text[a:b]
            lead = len(operand) - len(operand.lstrip())
            core = operand.strip()
            catalog.append(
                {
                    "id": next_id,
                    "file": C_VALIDATOR,
                    "line": text[: a + lead].count("\n") + 1,
                    "operand": " ".join(core.split()),
                }
            )
            edits.append(
                (a + lead, a + lead + len(core), f"(({core}) && !vf_mutant({next_id}))")
            )
            next_id += 1
    for a, b, replacement in sorted(edits, reverse=True):
        text = text[:a] + replacement + text[b:]
    marker = "#include <stdlib.h>\n"
    path.write_text(text.replace(marker, marker + C_SWITCH, 1))


# Python refusals on the delivery path: preflight, canary, packager, adapter, agents.
PY_TARGETS = (
    "scripts/h1_preflight.py",
    "scripts/hardware_canary.py",
    "scripts/package_h1.py",
    "adapters/skinny_moe.py",
    "scripts/vf_agents.py",
    # Appended in A44 so the A43 ids 1-451 keep their meaning.
    "scripts/resident_artifact.py",
    "scripts/full_expert_canary.py",
)
PY_REASON_SITES = ("py-reasons", "py-reach")
PY_SITES = ("py-raises", "py-handlers", *PY_REASON_SITES)
PY_REASON_SWITCH = """


class _VfMutantReached(BaseException):
    pass


def _vf_reason(mutant_id, exc):
    import os

    if os.environ.get("VF_MUTANT") != str(mutant_id):
        return exc
    if REACH:
        raise _VfMutantReached(mutant_id)
    exc.args = ("VF_MUTANT refusal reason replaced",)
    for name, value in list(vars(exc).items()):
        setattr(exc, name, (not value) if isinstance(value, bool) else "VF_MUTANT")
    return exc
"""
PY_SWITCH = """


def _vf_mutant(mutant_id):
    import os

    return os.environ.get("VF_MUTANT") == str(mutant_id)
"""


def python_guard(node, sites):
    """py-raises: the body starts with raise; py-handlers: a later body statement does."""
    if not isinstance(node, ast.If):
        return False
    if isinstance(node.body[0], ast.Raise):
        return sites == "py-raises"
    return sites == "py-handlers" and any(isinstance(s, ast.Raise) for s in node.body)


def rewrite_python(work, catalog, next_id, sites="py-raises"):
    """Disable each operand of `if ..: raise` guards; positions come from the AST.

    With sites="py-handlers", also make each translating `raise X(..)` written
    directly in an `except` body re-raise the original exception instead. A bare
    `raise` there already propagates the original and is not a site.
    """
    for rel in PY_TARGETS:
        path = work / rel
        source = path.read_bytes()
        starts = [0]
        for line in source.splitlines(keepends=True):
            starts.append(starts[-1] + len(line))
        edits = []
        tree = ast.parse(source)
        if sites in PY_REASON_SITES:
            next_id = reason_raises(rel, source, starts, tree, catalog, next_id, edits)
        for node in ast.walk(tree) if sites not in PY_REASON_SITES else ():
            if sites == "py-handlers" and isinstance(node, ast.ExceptHandler):
                for stmt in node.body:
                    if not (isinstance(stmt, ast.Raise) and stmt.exc is not None):
                        continue
                    a = starts[stmt.lineno - 1]
                    at = a + stmt.col_offset
                    b = starts[stmt.end_lineno - 1] + stmt.end_col_offset
                    if source[a:at].strip():
                        raise SystemExit(f"{rel}:{stmt.lineno}: raise shares its line")
                    catalog.append(
                        {
                            "id": next_id,
                            "file": rel,
                            "line": stmt.lineno,
                            "kind": "translation",
                            "handler": ast.unparse(node.type) if node.type else "",
                            "raise": " ".join(source[at:b].decode().split()),
                        }
                    )
                    pad = " " * stmt.col_offset
                    edits.append(
                        (a, a, f"{pad}if _vf_mutant({next_id}):\n{pad}    raise\n")
                    )
                    next_id += 1
                continue
            if not python_guard(node, sites):
                continue
            if any(
                isinstance(n, ast.Name) and n.id == "__name__"
                for n in ast.walk(node.test)
            ):
                continue  # the script entry point, not a refusal
            test = node.test
            operands = (
                test.values
                if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or)
                else [test]
            )
            for operand in operands:
                a = starts[operand.lineno - 1] + operand.col_offset
                b = starts[operand.end_lineno - 1] + operand.end_col_offset
                core = source[a:b].decode()
                entry_row = {
                    "id": next_id,
                    "file": rel,
                    "line": operand.lineno,
                    "operand": " ".join(core.split()),
                }
                if sites == "py-handlers":
                    entry_row["kind"] = "guard"
                catalog.append(entry_row)
                edits.append((a, b, f"(({core}) and not _vf_mutant({next_id}))"))
                next_id += 1
        # The switch must exist before a script's entry point calls main().
        entry = [
            n
            for n in tree.body
            if isinstance(n, ast.If)
            and any(isinstance(m, ast.Name) and m.id == "__name__" for m in ast.walk(n))
        ]
        switch = PY_SWITCH
        if sites in PY_REASON_SITES:
            reach = "True" if sites == "py-reach" else "False"
            switch = PY_REASON_SWITCH.replace("if REACH:", f"if {reach}:")
        if entry:
            at = starts[entry[0].lineno - 1]
            edits.append((at, at, switch.lstrip("\n") + "\n\n"))
        for a, b, replacement in sorted(edits, reverse=True):
            source = source[:a] + replacement.encode() + source[b:]
        path.write_bytes(source if entry else source + switch.encode())
    return next_id


def reason_raises(rel, source, starts, tree, catalog, next_id, edits):
    """py-reasons / py-reach: wrap the exception of each `raise <call>(..)` inside a
    function in _vf_reason(id, ..). A module-level raise would run before the switch
    is defined and is refused; `raise SystemExit(..)` is the script exit, not a site."""
    inside = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inside.update(id(n) for n in ast.walk(fn))
    raises = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
    ]
    for node in sorted(raises, key=lambda n: (n.lineno, n.col_offset)):
        exc = node.exc
        if isinstance(exc.func, ast.Name) and exc.func.id == "SystemExit":
            continue
        if id(node) not in inside:
            raise SystemExit(f"{rel}:{node.lineno}: module-level raise")
        a = starts[exc.lineno - 1] + exc.col_offset
        b = starts[exc.end_lineno - 1] + exc.end_col_offset
        core = source[a:b].decode()
        catalog.append(
            {
                "id": next_id,
                "file": rel,
                "line": node.lineno,
                "raise": " ".join(core.split()),
            }
        )
        edits.append((a, b, f"_vf_reason({next_id}, {core})"))
        next_id += 1
    return next_id


# Refusals of the H1 shell launchers (A47). The arm rewrite relies on bash's `;;&`.
SH_TARGETS = ("handoff/h1_bundle_run.sh", "handoff/hardware_canary.sh")
SCRIPT_SITES = (*PY_SITES, "sh-launchers")
SH_REQUIRED = re.compile(r'^(\s*): "\$\{(\w+):\?([^}]*)\}"\s*$')
SH_OPTIONS = "set -euo pipefail"
SH_CASE = re.compile(r"^\s*case .* in\s*$")
SH_ARM = re.compile(r"^(\s*)([^()]+)\)\s*(.*\bexit\b.*?)[\s;]*;;\s*$")


def shell_patterns(text):
    """Split a case-arm pattern list on `|` outside quotes and brackets."""
    parts, depth, quote, start = [], 0, None, 0
    for i, c in enumerate(text):
        if quote:
            quote = None if c == quote else quote
        elif c in "\"'":
            quote = c
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
        elif c == "|" and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def rewrite_shell(work, catalog, next_id):
    """Make each launcher refusal switchable by VF_MUTANT, read by bash itself.

    A required-parameter check gives two sites: removed, and `:?` weakened to `?`.
    `set -euo pipefail` gives one site per option, switched off right after it. A
    case arm that exits is split into one arm per pattern, each ending in `;;&`, so
    a switched-off pattern falls through to the later arms exactly as if deleted.
    Any other `exit`, `:?` or `set` stops the rewrite rather than going uncounted.
    """

    def site(rel, line, **fields):
        nonlocal next_id
        catalog.append({"id": next_id, "file": rel, "line": line, **fields})
        next_id += 1
        return next_id - 1

    for rel in SH_TARGETS:
        path = work / rel
        lines = path.read_text().splitlines()
        if lines[0] != "#!/usr/bin/env bash":
            raise SystemExit(f"{rel}: the arm rewrite needs bash")
        out, in_case = [], False
        for n, line in enumerate(lines, 1):
            required = SH_REQUIRED.match(line)
            arm = SH_ARM.match(line) if in_case else None
            if required:
                pad, name, message = required.groups()
                gone = site(rel, n, kind="required", operand=f"${{{name}:?}}")
                empty = site(rel, n, kind="nonempty", operand=f"${{{name}:?}} -> ?")
                out.append(
                    f'{pad}case "${{VF_MUTANT:-}}" in {gone}) ;; '
                    f'{empty}) : "${{{name}?{message}}}" ;; '
                    f'*) : "${{{name}:?{message}}}" ;; esac'
                )
            elif line == SH_OPTIONS:
                ids = [
                    site(rel, n, kind="option", operand=o)
                    for o in ("-e", "-u", "-o pipefail")
                ]
                out += [
                    line,
                    f'case "${{VF_MUTANT:-}}" in {ids[0]}) set +e ;; '
                    f"{ids[1]}) set +u ;; {ids[2]}) set +o pipefail ;; esac",
                ]
            elif arm:
                pad, patterns, command = arm.groups()
                for pattern in shell_patterns(patterns):
                    i = site(
                        rel, n, kind="pattern", operand=pattern, arm=patterns.strip()
                    )
                    out.append(
                        f'{pad}{pattern}) if [ "${{VF_MUTANT:-}}" != {i} ]; '
                        f"then {command}; fi ;;&"
                    )
            elif re.search(r"\bexit\b|:\?|^\s*set\s", line):
                raise SystemExit(f"{rel}:{n}: refusal form not covered: {line.strip()}")
            else:
                out.append(line)
            in_case = (in_case or bool(SH_CASE.match(line))) and not re.match(
                r"^\s*esac\b", line
            )
        path.write_text("\n".join(out) + "\n")
    return next_id


def build_copy(source, work, sites="clauses"):
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(
        source, work, ignore=lambda _d, names: [n for n in names if n in SKIPPED]
    )
    for rel in NEEDED_EVIDENCE:
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, work / rel)
    catalog, next_id = [], 1
    if sites in C_SITES:
        rewrite_c(work, catalog, next_id, sites)
        return catalog
    if sites in PY_SITES:
        rewrite_python(work, catalog, next_id, sites)
        return catalog
    if sites == "sh-launchers":
        rewrite_shell(work, catalog, next_id)
        return catalog
    for path in sorted((work / "src").rglob("*.rs")):
        rel = path.relative_to(work).as_posix()
        text = path.read_text()
        root = rel in CRATE_ROOTS and rel != "src/lib.rs"
        switch = "voltafabric::__vf_mutant" if root else "crate::__vf_mutant"
        error = "voltafabric::Error" if root else "crate::Error"
        edits, taken = [], []
        for a, b in map_err_sites(text) if sites in MAP_SITES else ():
            core = text[a:b]
            catalog.append(
                {
                    "id": next_id,
                    "file": rel,
                    "line": text[:a].count("\n") + 1,
                    "map_err": " ".join(core.split()),
                }
            )
            prefix = "voltafabric" if root else "crate"
            replaced = (
                f"{prefix}::__VfReplace::__vf_replace(__vf_r)"
                if sites == "map-reasons"
                else 'panic!("VF_MUTANT_REACHED")'
            )
            edits.append(
                (
                    a,
                    b,
                    f"|__vf_e| {{ let __vf_r = {prefix}::__vf_apply(__vf_e, {core}); "
                    f"if {switch}({next_id}) {{ {replaced} }} else {{ __vf_r }} }}",
                )
            )
            next_id += 1
        for a, b in ret_sites(text) if sites in RET_SITES else ():
            core = text[a:b]
            catalog.append(
                {
                    "id": next_id,
                    "file": rel,
                    "line": text[:a].count("\n") + 1,
                    "err": " ".join(core.split()),
                }
            )
            prefix = "voltafabric" if root else "crate"
            mutant = (
                'panic!("VF_MUTANT_REACHED")'
                if sites == "ret-reach"
                else f"{prefix}::__VfReplace::__vf_replace(__vf_r)"
            )
            if core.startswith("serde::de::Error::custom("):
                # A Deserialize error of the caller's type, inferred from the branch.
                mutant = (
                    'panic!("VF_MUTANT_REACHED")'
                    if sites == "ret-reach"
                    else 'serde::de::Error::custom("VF_MUTANT refusal reason replaced")'
                )
                replaced = f"if {switch}({next_id}) {{ {mutant} }} else {{ {core} }}"
            else:
                # Bound first: an if-expression error would otherwise nest as
                # `else { if .. }`, which clippy -D warnings refuses in the gate.
                replaced = (
                    f"{{ let __vf_r = {core}; if {switch}({next_id}) "
                    f"{{ {mutant} }} else {{ __vf_r }} }}"
                )
            edits.append((a, b, replaced))
            next_id += 1
        if sites in ("reasons", "reach"):
            for a, b in reason_sites(text):
                core = text[a:b]
                line = text[:a].count("\n") + 1
                catalog.append(
                    {
                        "id": next_id,
                        "file": rel,
                        "line": line,
                        "reason": " ".join(core.split()),
                    }
                )
                replaced = (
                    f'{error}::new("VF_MUTANT", "refusal reason replaced")'
                    if sites == "reasons"
                    else 'panic!("VF_MUTANT_REACHED")'
                )
                edits.append(
                    (a, b, f"if {switch}({next_id}) {{ {replaced} }} else {{ {core} }}")
                )
                next_id += 1
        for start, brace, spans in guards(text) if sites == "clauses" else ():
            if any(start < b and a < brace for a, b in taken):
                raise SystemExit(f"overlapping guards in {rel} near byte {start}")
            taken.append((start, brace))
            for a, b in spans:
                operand = text[a:b]
                lead = len(operand) - len(operand.lstrip())
                core = operand.strip()
                line = text[: a + lead].count("\n") + 1
                edits.append(
                    (
                        a + lead,
                        a + lead + len(core),
                        f"(({core}) && !{switch}({next_id}))",
                    )
                )
                catalog.append(
                    {
                        "id": next_id,
                        "file": rel,
                        "line": line,
                        "operand": " ".join(core.split()),
                    }
                )
                next_id += 1
        spans = sorted((a, b) for a, b, _ in edits)
        if any(b > c for (_, b), (c, _) in zip(spans, spans[1:])):
            raise SystemExit(f"overlapping sites in {rel}")
        for a, b, replacement in sorted(edits, reverse=True):
            text = text[:a] + replacement + text[b:]
        if rel in CRATE_ROOTS:
            # The rewrite adds parentheses rustc would otherwise warn about.
            text = "#![allow(unused_parens)]\n" + text
        if rel == "src/lib.rs":
            text += SWITCH + (MAP_SWITCH if sites in MAP_SITES + RET_SITES else "")
        path.write_text(text)
    return catalog


def scope(memory_max):
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={memory_max}",
        "-p",
        "MemorySwapMax=0",
    ]


def stage_one(sites, python=None):
    if sites in SCRIPT_SITES:
        suites = " && ".join(
            f'"{python}" -m unittest discover -s {suite}'
            for suite in ("tests/python", "tests/adapter")
        )
        return ["bash", "-c", suites]
    if sites in C_SITES:
        cargo = (
            "cargo test --locked --offline --workspace --all-targets "
            "--features c-abi-validation"
        )
        return ["bash", "-c", f"{cargo} && {C_DRIVER}"]
    return ["cargo", "test", "--locked", "--offline", "--all-targets"]


def run_rust(work, wrap, env, mutant, log, timeout, sites="clauses", python=None):
    env = dict(env)
    env.pop("VF_MUTANT", None)
    if mutant is not None:
        env["VF_MUTANT"] = str(mutant)
    argv = wrap + stage_one(sites, python)
    started = time.monotonic()
    # Its own session, so that a timeout stops the whole run: killing only the direct
    # child left test binaries and the workers they spawn running into the next runs.
    p = subprocess.Popen(
        argv,
        cwd=work,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = p.communicate(timeout=timeout)
        code, timed_out = p.returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            # A process outside the group still holds the pipes: keep no output.
            p.kill()
            p.wait()
            stdout, stderr = "", ""
        code, timed_out = None, True
    out = (stdout or "") + (stderr or "")
    log.write_text(out)
    return {
        "exit_code": code,
        "timed_out": timed_out,
        "tests_passed": sum(map(int, re.findall(r"ok\. (\d+) passed", out))),
        "failing_tests": sorted(
            set(re.findall(r"^test (\S+) \.\.\. FAILED", out, re.M))
        ),
        "seconds": round(time.monotonic() - started, 1),
    }


def stage_one_killed(run):
    """A mutant dies in stage one only when its run finished and failed. A run that hit
    --timeout (exit_code None) executed no verdict on the site: it is not a kill."""
    return not run["timed_out"] and run["exit_code"] != 0


def run_gate(work, wrap, env, python, mutant, run_id):
    target = work / "evidence/runs" / run_id
    if target.exists():
        shutil.rmtree(target)
    env = dict(env)
    env.pop("VF_MUTANT", None)
    if mutant is not None:
        env["VF_MUTANT"] = str(mutant)
    argv = wrap + [python, "scripts/verify.py", "--run-id", run_id]
    subprocess.run(
        argv, cwd=work, env=env, capture_output=True, text=True, timeout=7200
    )
    summary = json.loads((target / "summary.json").read_text())
    failed = [s["step"] for s in summary["steps"] if s["exit_code"] != 0]
    return summary["status"], failed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True, help="rewritten copy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--sites",
        choices=(
            "clauses",
            "reasons",
            "reach",
            *MAP_SITES,
            *RET_SITES,
            *C_SITES,
            *SCRIPT_SITES,
        ),
        default="clauses",
        help="clauses: disable each || operand of an if-guard; reasons: replace the "
        "error of each ok_or_else closure and match-arm Err with a placeholder; "
        "reach: make those same sites panic, to tell unreached from unchecked; "
        "c-clauses: disable each || operand of the C launch validator's checks; "
        "py-raises: disable each `or` operand of `if ..: raise` in the Python "
        "delivery-path modules (needs --gate-python); py-handlers: the same for "
        "guards whose raise follows other statements, and make each translating "
        "raise in an except body re-raise the original exception instead; "
        "sh-launchers: switch off each refusal of the two H1 bash launchers "
        "(needs --gate-python); map-reasons / map-reach: replace the error of each "
        ".map_err(..) with a placeholder of the same type, or panic there; "
        "ret-reasons / ret-reach: the same for each return Err(..) and block-tail "
        "Err(..) outside the reasons catalog; c-codes / c-reach: make each refusal "
        "return of the C launch validator return -99, or abort, instead; "
        "py-reasons / py-reach: replace the message and attributes of each Python "
        "raise of the delivery path, or raise an uncatchable marker there",
    )
    ap.add_argument("--only", default="", help="comma-separated clause ids")
    ap.add_argument(
        "--gate-python", help="adapter interpreter for stage two (as given)"
    )
    ap.add_argument("--memory-max", default="8G")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per Rust run")
    ap.add_argument("--catalog-only", action="store_true")
    ap.add_argument(
        "--skip-gate",
        action="store_true",
        help="stop after the first stage: no gate control and no gate per survivor",
    )
    a = ap.parse_args()
    source = a.source.resolve(strict=True)
    a.out.mkdir(parents=True, exist_ok=False)
    catalog = build_copy(source, a.work, a.sites)
    (a.out / "CATALOG.json").write_text(json.dumps(catalog, indent=1) + "\n")
    print(f"{len(catalog)} {a.sites}", flush=True)
    if a.catalog_only:
        return 0
    wrap = scope(a.memory_max)
    if subprocess.run(wrap + ["true"], capture_output=True).returncode != 0:
        print("ENVIRONMENT_ERROR: per-user systemd scope unavailable", flush=True)
        return 4
    env = dict(os.environ, CARGO_TARGET_DIR=str(a.work / "target"))
    env.setdefault("RUST_TEST_THREADS", "8")
    env.pop("TMPDIR", None)
    fmt = subprocess.run(["cargo", "fmt", "--all"], cwd=a.work, env=env)
    build = subprocess.run(
        ["cargo", "build", "--locked", "--offline", "--all-targets"],
        cwd=a.work,
        env=env,
        capture_output=True,
        text=True,
    )
    (a.out / "BUILD.log").write_text(build.stdout + build.stderr)
    if a.sites in SCRIPT_SITES and not (fmt.returncode or build.returncode):
        if not a.gate_python:
            print(f"{a.sites} needs --gate-python for the Python suites", flush=True)
            return 2
        # The Python suites need a release vf, as verify.py provides them.
        release = subprocess.run(
            ["cargo", "build", "--locked", "--offline", "--release"],
            cwd=a.work,
            env=env,
            capture_output=True,
            text=True,
        )
        ruff = subprocess.run(
            [a.gate_python, "-m", "ruff", "format", *PY_TARGETS],
            cwd=a.work,
            capture_output=True,
            text=True,
        )
        with open(a.out / "BUILD.log", "a") as log:
            log.write(release.stdout + release.stderr + ruff.stdout + ruff.stderr)
        if release.returncode or ruff.returncode:
            print("schemata copy does not build; no verdict is valid", flush=True)
            return 2
        env.update(
            VF_BINARY=str(a.work / "target/release/vf"),
            PYTHONPATH=str(a.work),
            PYTHONDONTWRITEBYTECODE="1",
        )
    if fmt.returncode or build.returncode:
        print("schemata copy does not build; no verdict is valid", flush=True)
        return 2
    summary = {
        "schema_version": 1,
        "tool": "scripts/guard_schemata.py",
        "argv": sys.argv,
        "limits": wrap,
        "sites": a.sites,
        "clauses": len(catalog),
    }
    control = run_rust(
        a.work,
        wrap,
        env,
        None,
        a.out / "CONTROL.log",
        a.timeout,
        a.sites,
        a.gate_python,
    )
    summary["rust_control"] = control
    print("RUST CONTROL", control, flush=True)
    if control["exit_code"] != 0:
        summary["status"] = "CONTROL_FAILED"
        (a.out / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
        return 3
    wanted = {int(x) for x in a.only.split(",") if x} or {c["id"] for c in catalog}
    rows = []
    for c in catalog:
        if c["id"] not in wanted:
            continue
        r = run_rust(
            a.work,
            wrap,
            env,
            c["id"],
            a.out / f"M{c['id']}.log",
            a.timeout,
            a.sites,
            a.gate_python,
        )
        r.update(c, rust_killed=stage_one_killed(r))
        rows.append(r)
        print(
            f"M{c['id']} {c['file']}:{c['line']} killed={r['rust_killed']}", flush=True
        )
    summary["stage_two"] = "skipped (--skip-gate)" if a.skip_gate else "run"
    if a.gate_python and not a.skip_gate:
        status, failed = run_gate(a.work, wrap, env, a.gate_python, None, "gcontrol")
        summary["gate_control"] = {"status": status, "failing_steps": failed}
        print("GATE CONTROL", status, failed, flush=True)
        if status != "CPU_FULL_PASS":
            summary["status"] = "GATE_CONTROL_FAILED"
            summary["mutants"] = rows
            (a.out / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
            return 3
        for r in rows:
            if r["rust_killed"]:
                continue
            status, failed = run_gate(
                a.work, wrap, env, a.gate_python, r["id"], f"g{r['id']}"
            )
            real = [s for s in failed if s not in COSMETIC_STEPS]
            r.update(
                gate_status=status, gate_failing_steps=failed, gate_killed=bool(real)
            )
            print(f"G{r['id']} killed={r['gate_killed']} steps={failed}", flush=True)
    survivors = [
        r["id"] for r in rows if not r["rust_killed"] and not r.get("gate_killed")
    ]
    summary.update(
        status="SURVIVORS" if survivors else "ALL_KILLED",
        run=len(rows),
        rust_killed=sum(r["rust_killed"] for r in rows),
        gate_killed=sum(bool(r.get("gate_killed")) for r in rows),
        survivors=survivors,
        cosmetic_steps_excluded=sorted(COSMETIC_STEPS),
        mutants=rows,
    )
    (a.out / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(
        json.dumps({k: summary[k] for k in ("clauses", "run", "survivors")}), flush=True
    )
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
