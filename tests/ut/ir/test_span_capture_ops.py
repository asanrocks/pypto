# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pinned spans win over the operator dunders' frame capture.

The Expr operator overloads (``__add__``, ``__rmul__``, ...) share the
builder's span discipline: a pinned span (the DSL parser per DSL call, the
trace interpreter per instrumented row) is fetched, and only an unpinned
context falls back to walking the Python stack.
"""

from pypto.ir import IRBuilder
from pypto.ir.utils import use_parser_span
from pypto.pypto_core import DataType, ir


def _int_var():
    b = IRBuilder()
    b.function("f", ir.Span.unknown())
    return b.var("x", ir.ScalarType(DataType.INT32))


def test_pinned_span_wins_for_expr_dunder_add():
    a = _int_var()
    pin = ir.Span("pinned.py", 42, 4)
    with use_parser_span(pin):
        call = a + 1
    assert str(call.span.filename).endswith("pinned.py")
    assert call.span.begin_line == 42


def test_pinned_span_wins_for_expr_dunder_rmul():
    a = _int_var()
    pin = ir.Span("pinned.py", 7, 0)
    with use_parser_span(pin):
        call = 2 * a
    assert str(call.span.filename).endswith("pinned.py")
    assert call.span.begin_line == 7


def test_unpinned_context_still_captures_the_caller():
    a = _int_var()
    call = a + 1
    assert str(call.span.filename).endswith("test_span_capture_ops.py")
