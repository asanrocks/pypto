# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Traced IR spans: exact, from the instrumentor's row pins.

The rewriter emits ``rt._span(line, col)`` before every logical statement
row; the Sema pins it so the op and builder layers attach the kernel's
original source position instead of walking the Python stack (which
produced drifted lines for ops and, for statements, spans pointing at the
Sema's own source file).
"""

import ast
import pathlib

import pypto.language as pl
import pytest
from pypto.ir.utils import current_pinned_span
from pypto.language.tracing import semantics
from pypto.language.tracing.decorators.function import function_trace


@function_trace
def _spanned(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
    acc = x
    for i in range(4):
        acc = pl.add(acc, x)
    if True:
        acc = pl.mul(acc, x)
        y = pl.add(acc, x)
    return y


@function_trace
def _structurally_spanned(x: pl.Tensor[[8], pl.FP32], c: pl.Scalar[pl.INT32]) -> pl.Tensor[[8], pl.FP32]:
    acc = x
    with pl.at(level=pl.Level.CORE_GROUP):
        acc = pl.add(acc, x)
    for i in pl.range(4):
        acc = pl.add(acc, x)
    if c > 0:
        acc = pl.mul(acc, x)
    while c < 1:
        acc = pl.add(acc, x)
    return acc


def _kernel_source_facts() -> tuple[str, dict[str, int]]:
    """(filename, {'first': line of ``acc = x``, 'return': line of ``return y``})
    parsed from this test file, so the assertions track their own source."""
    source = pathlib.Path(__file__).read_text()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_spanned")
    # the function-level span points at the decorator line when present
    start = min([fn.lineno, *(d.lineno for d in fn.decorator_list)])
    return __file__, {"def": start, "first": fn.body[0].lineno, "return": fn.body[-1].lineno}


def _statement_spans(node, out: list) -> None:
    span = getattr(node, "span", None)
    if span is not None and hasattr(span, "begin_line"):
        out.append(span)
    for ch in getattr(node, "stmts", None) or []:
        _statement_spans(ch, out)
    body = getattr(node, "body", None)
    if body is not None and not isinstance(body, str):
        _statement_spans(body, out)


def _structural_stmt_lines(kernel, wanted: set[str]) -> dict[str, list[int]]:
    """Source lines of the traced ``wanted`` statement types, keyed by the
    matched type-name suffix (``InCoreScopeStmt`` counts as ``ScopeStmt``)."""
    found: dict[str, list[int]] = {}

    def walk(node) -> None:
        span = getattr(node, "span", None)
        name = type(node).__name__
        if span is not None and hasattr(span, "begin_line"):
            for w in wanted:
                if name.endswith(w):
                    found.setdefault(w, []).append(span.begin_line)
                    break
        for ch in getattr(node, "stmts", None) or []:
            walk(ch)
        body = getattr(node, "body", None)
        if body is not None and not isinstance(body, str):
            walk(body)

    walk(kernel)
    return found


def test_traced_statement_spans_point_at_the_kernel_source():
    filename, lines = _kernel_source_facts()
    spans: list = []
    _statement_spans(_spanned, spans)
    assert spans, "the traced function carries statement spans"
    for span in spans:
        assert str(span.filename) == filename, (
            f"statement span attributed to {span.filename}, not the kernel source"
        )
        # every span lies inside the kernel's line range — the decorator
        # line for the function-level span, the body range for statements
        # (the emitted expansion never invented positions outside it)
        assert lines["def"] <= span.begin_line <= lines["return"]


def test_traced_statement_spans_are_exact():
    _filename, lines = _kernel_source_facts()
    spans: list = []
    _statement_spans(_spanned, spans)
    begin = [s.begin_line for s in spans]
    # the first statement and the return carry their true source lines —
    # frame capture drifted these by the emitted-code expansion
    assert lines["first"] in begin
    assert lines["return"] in begin


def test_traced_structural_nodes_carry_the_pinned_location():
    """For/while/if/scope statements built through the direct builder
    surface carry the kernel's row pin.

    The trace frontend opens these regions without an explicit span; the
    builder's ``begin_*``/``end_*`` primitives resolve ``span=None`` through
    the pinned span, so the structural node keeps the user's source line
    instead of degrading to ``Span.unknown()``.
    """
    source = pathlib.Path(__file__).read_text()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_structurally_spanned")
    # the source lines of the ``for`` / ``if`` / ``while`` headers, straight
    # from this file's own AST. The scope is different: ``pl.at`` defers its
    # open until the first body statement lands (attrs are fixed at open),
    # so the pinned row at the open is the *first body line* of the ``with``.
    expected = {
        "ScopeStmt": [w.body[0].lineno for w in fn.body if isinstance(w, ast.With)],
        "ForStmt": [stmt.lineno for stmt in fn.body if isinstance(stmt, ast.For)],
        "IfStmt": [stmt.lineno for stmt in fn.body if isinstance(stmt, ast.If)],
        "WhileStmt": [stmt.lineno for stmt in fn.body if isinstance(stmt, ast.While)],
    }
    assert all(expected.values()), "the probe kernel spells every region once"

    found = _structural_stmt_lines(_structurally_spanned, {"ForStmt", "IfStmt", "WhileStmt", "ScopeStmt"})
    for kind, lines in expected.items():
        assert found.get(kind) == lines, (
            f"traced {kind} spans {found.get(kind)}, expected the kernel lines {lines}"
        )


def test_span_pin_is_set_during_and_cleared_after_a_trace():
    """A completing trace sets the pin while rows execute and restores it
    on finish; no stale pin leaks into later, unrelated IR construction.
    (A trace aborted by an exception keeps its last pin by design — spans
    are diagnostics-only and the process is on an error path.)"""
    _seen: list = []

    orig_note = semantics.PtoSema.note_span

    def recording_note(self, source_file, lineno, col):  # noqa: ANN001
        orig_note(self, source_file, lineno, col)
        _seen.append(current_pinned_span())

    semantics.PtoSema.note_span = recording_note
    try:
        before = current_pinned_span()
        function_trace(_raw_spanned)
    finally:
        semantics.PtoSema.note_span = orig_note
    assert _seen, "the trace pinned its rows"
    assert all(s is not None for s in _seen), "row pins are visible during execution"
    assert current_pinned_span() == before, "the pin was restored on finish"


def _raw_spanned(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
    y = pl.add(x, x)
    return y


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
