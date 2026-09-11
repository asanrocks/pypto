# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Typing-debt ratchet for the collector strictness overlay.

The overlay (``pyrightconfig.collector.json``) follows the former collector
and flow source lineage across ``language/tracer`` and ``language/constructs``.
Pyright cannot ratchet *explicit* escape hatches: ``reportExplicitAny`` does not exist in
pyright (it is a basedpyright rule). This script counts them instead:

- ``Any`` annotations (the dynamic escape),
- ``cast(...)`` calls (an unchecked claim),
- ``# type: ignore`` comments (a suppressed diagnostic).

``object`` is deliberately not counted: at the collector boundary it is the
*by-design* spelling (host values and trace-protocol objects cross as
``object``; only results produced by the Frame protocol are ``V``). Counting it would
punish the correct code.

Budgets are per-file ceilings. A file not listed has a zero budget: new
files enter clean, and lowering a budget is always allowed — raising one
requires justifying it next to the entry. Zero-cast/zero-ignore outside the
named confinement helpers is the end state.
"""

from __future__ import annotations

import re
import sys
import tokenize
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# (any, cast, ignore) ceilings for the overlay trees. Every entry above zero
# is a known, documented pocket — see the comment next to each entry. Zero
# everywhere outside the named pockets is the end state.
ANY_RE = re.compile(r"\bAny\b")
CAST_RE = re.compile(r"\bcast\s*\(")
IGNORE_RE = re.compile(r"#\s*type:\s*ignore")

# Token types carrying no code: comments and string literals (docstrings
# included — prose mentioning "Any" is documentation, not debt).
_FSTRING_TOKENS = tuple(
    t
    for t in (
        getattr(tokenize, "FSTRING_START", None),
        getattr(tokenize, "FSTRING_MIDDLE", None),
        getattr(tokenize, "FSTRING_END", None),
    )
    if t is not None
)
_NOISE_TOKENS = frozenset({tokenize.COMMENT, tokenize.STRING, *_FSTRING_TOKENS})

BUDGETS: dict[str, tuple[int, int, int]] = {
    "python/pypto/language/tracer/collector/__init__.py": (0, 0, 0),
    # 3 casts = the target-router confinement (one per CST target family:
    # _route_assign_target, _route_del_target, _as_line) — the documented
    # libcst stub-gap claims.
    "python/pypto/language/tracer/collector/rewriter.py": (0, 3, 0),
    "python/pypto/language/tracer/collector/interface.py": (0, 0, 0),
    # ForConstruct.activate_target's opaque ctx (the language-side
    # TraceContext cannot be named from the collector); 1 cast = begin_for's
    # host-path return — the Frame's decline is the runtime proof the value
    # is the host iterable the generated ``for`` iterates, but the parameter
    # must stay ``object`` (a staged trace-iterable descriptor is not
    # iterable).
    "python/pypto/language/tracer/collector/interpreter.py": (3, 1, 0),
    # The Sema-source duck-typing heuristic and the instrumented-call
    # boundary (the explicit factory/overload split is recorded follow-up).
    "python/pypto/language/tracer/collector/driver.py": (8, 2, 0),
    # The former flow/dsl.py budget is split exactly across loops and scopes.
    # loops.py: 1 Any = the _Attrs alias (the machine-only ForStmt attr
    # mapping); 1 cast = RangeIterator.__next__'s traced-yield (T) vs
    # host-counter (RangeArg) bridge — tracing binds the Scalar loop var
    # through __trace_iter__, never through __next__; the remaining 5 Any
    # are the r10-base additions (the former flow/dsl.py split).
    "python/pypto/language/constructs/loops.py": (6, 1, 0),
    # +1 Any: the deferred spmd carrier stores the inner InCore scope-attr
    # tuples (the same heterogeneous attr pocket the immediate form threads
    # straight into begin_scope) until the body's first index read decides.
    # +5 Any: the cache-policy declaration channel — the shared attr-merge
    # helper's attr pocket, and ``at``'s deferred-begin tuple (kind, begin
    # kwargs, attrs) held until the body's first statement opens the scope.
    # +1 cast: the scope-attr dump_vars read reifies the attr value (the
    # attr carries Vars by construction — see the protocol budget note for
    # the lineage).
    "python/pypto/language/constructs/scopes.py": (16, 1, 0),
    # ir.Submit attrs (the C++ attribute bridge) + the emit-closure ctx +
    # _SubmitResult's dynamic facade elements (__iter__ and __getitem__ —
    # the callee's declared-return-derived projections are unknowable
    # statically).
    "python/pypto/language/constructs/submit.py": (10, 0, 0),
    # func_attr's open-world attr values (host scalars, facades, enums, nested
    # lists) — the parser's dual infers them from AST syntax; tracing receives
    # them as runtime objects, so the coercion works the same open-world pocket.
    "python/pypto/language/constructs/func_attrs.py": (13, 0, 0),
    "python/pypto/language/constructs/optimizations.py": (2, 0, 0),
    "python/pypto/language/tracer/constructs.py": (4, 0, 0),
    # The former flow/sema.py budget follows its split between protocol and semantics;
    # semantics gains the scope-attr dump_vars read (the attr carries Vars by
    # construction — the parser, the marker call-merge, and merge_sticky_dump_vars
    # append only Vars; the attr value type is the opaque object bridge).
    "python/pypto/language/protocol.py": (3, 0, 0),
    "python/pypto/language/tracer/semantics.py": (6, 2, 0),
}

OVERLAY_PATHS = (
    "python/pypto/language/protocol.py",
    "python/pypto/language/tracer/collector",
    "python/pypto/language/tracer/semantics.py",
    "python/pypto/language/tracer/constructs.py",
    "python/pypto/language/constructs",
)


def _code_only(text: str) -> str:
    """The file's code with comments and string literals removed.

    Docstring prose ("Any later call") and commented-out code must not count
    as debt; tokenize is the only reliable separator.
    """
    return " ".join(
        tok.string
        for tok in tokenize.generate_tokens(StringIO(text).readline)
        if tok.type not in _NOISE_TOKENS
    )


def counts(path: Path) -> tuple[int, int, int]:
    text = path.read_text(encoding="utf-8")
    # Count Any/cast in code only — docstrings and comments mentioning "Any"
    # are documentation, not debt.
    code = _code_only(text)
    return (
        len(ANY_RE.findall(code)),
        len(CAST_RE.findall(code)),
        len(IGNORE_RE.findall(text)),
    )


def main() -> int:
    files: list[Path] = []
    for entry in OVERLAY_PATHS:
        path = REPO_ROOT / entry
        if path.is_file():
            files.append(path)
        else:
            files.extend(p for p in path.rglob("*.py") if "__pycache__" not in p.parts)
    files.sort()
    failed = False
    print(f"{'file':<45} {'Any':>4} {'cast':>5} {'ignore':>7}   budget(any/cast/ignore)")
    for path in files:
        rel = str(path.relative_to(REPO_ROOT))
        got = counts(path)
        budget = BUDGETS.get(rel, (0, 0, 0))
        mark = ""
        if any(g > b for g, b in zip(got, budget, strict=True)):
            mark = "  OVER BUDGET"
            failed = True
        print(f"{rel:<45} {got[0]:>4} {got[1]:>5} {got[2]:>7}   {budget[0]}/{budget[1]}/{budget[2]}{mark}")
    if failed:
        print(
            "\nFAIL: typing-debt budget exceeded — shrink the pocket or lower the"
            " budget with a justification next to the entry.",
            file=sys.stderr,
        )
        return 1
    print("\nPASS: all files within typing-debt budgets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
