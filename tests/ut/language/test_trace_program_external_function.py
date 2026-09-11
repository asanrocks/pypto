# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tracing coverage for standalone functions captured by program methods."""

import pypto.language as pl
from pypto import ir


def test_program_captures_standalone_function_call():
    """A program method can call a standalone function captured in its closure."""

    @pl.function
    def scale(x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        tile = pl.load(x, [0, 0], [16, 16])
        return pl.store(tile, [0, 0], x)

    @pl.program
    class Program:
        @pl.function
        def entry(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            return scale(x)

    assert Program.get_function("scale") is scale
    entry = Program.get_function("entry")
    assert entry is not None

    calls: list[ir.Call] = []

    class _CallCollector(ir.IRVisitor):
        def visit_call(self, op):
            calls.append(op)
            super().visit_call(op)

    _CallCollector().visit_function(entry)
    assert [call.op.name for call in calls if isinstance(call.op, ir.GlobalVar)] == ["scale"]
