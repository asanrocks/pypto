# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pinned spans win over the builder's frame capture.

The DSL parser pins a span per DSL call and the trace interpreter pins one
per instrumented row; without the pin check, ``_capture_call_span`` grabbed
two frames up — framework plumbing in traced execution, attributing IR
statements to the wrong file.
"""

from pypto.ir import IRBuilder
from pypto.ir.utils import use_parser_span
from pypto.pypto_core import DataType, ir


def test_pinned_span_wins_for_builder_var():
    b = IRBuilder()
    pin = ir.Span("pinned.py", 42, 4)
    with b.function("f", ir.Span.unknown()):
        with use_parser_span(pin):
            v = b.var("x", ir.ScalarType(DataType.INT32))
    assert str(v.span.filename).endswith("pinned.py")
    assert v.span.begin_line == 42


def test_pinned_span_wins_for_builder_let():
    b = IRBuilder()
    pin = ir.Span("pinned.py", 7, 0)
    with b.function("f", ir.Span.unknown()):
        base = b.var("x", ir.ScalarType(DataType.INT32))
        with use_parser_span(pin):
            bound = b.let("y", base)
    assert str(bound.span.filename).endswith("pinned.py")
    assert bound.span.begin_line == 7
