# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""R13 conformance for the assembly intrinsic boundary.

The retained ``pl.yield_`` alias is shared source syntax and is anchored
through the normal raw harness. The ``pl.intrin.*`` assembly channels are a
tracing-only dialect extension; the parser rejects those spellings, so their
coverage is an acceptance pin rather than a false conformance anchor.
"""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto.language.diagnostics import ParserError
from pypto.language.parser import function_parser, program_parser
from pypto.language.tracer.decorators.function import function_trace, trace_program

from ._harness import conformance


def k_yield_alias(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
    if x > 0:
        pl.yield_()
    return x


def test_deprecated_yield_alias_anchor_raw_equal() -> None:
    """The compatibility ``pl.yield_()`` spelling still has one parser and
    tracing implementation, and the zero-result staged-if shape is raw-equal."""
    with pytest.warns(DeprecationWarning, match="pl.yield_"):
        conformance(k_yield_alias, function_trace)


def k_intrin_channels(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
    seed = pl.const(0, pl.INT32)
    (carried,) = pl.intrin.carry(seed)
    for _ in pl.range(2):
        carried = pl.add(carried, x)
        pl.intrin.yield_(carried)
    (result,) = pl.intrin.leak()
    return result


def k_intrin_free_var() -> pl.Tensor[[16], pl.FP32]:
    value = pl.intrin.free_var(pl.Tensor[[16], pl.FP32], name="external")
    return value


def k_intrin_transmute(x: pl.Tensor[[2, 3, 4], pl.FP32]) -> pl.Tile[[6, 4], pl.FP32, pl.Mem.Vec]:
    return pl.intrin.transmute(
        pl.tile.load(x, [0, 0, 0], [2, 3, 4], target_memory=pl.Mem.Vec),
        pl.Tile[[6, 4], pl.FP32, pl.Mem.Vec],
    )


@pytest.mark.parametrize(
    "kernel",
    [k_intrin_channels, k_intrin_free_var, k_intrin_transmute],
    ids=["carry-yield-leak", "free-var", "transmute"],
)
def test_assembly_intrin_has_no_parser_spelling(kernel) -> None:
    """The assembly namespace is tracing-owned, per the migration dialect
    contract; each traced function is rejected by the AST parser."""
    function_trace(kernel)
    with pytest.raises(ParserError, match="intrin"):
        function_parser(kernel)


class IntrinSpmdDirect:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
    ) -> pl.Tensor[[16], pl.FP32]:
        return pl.store(pl.load(x, [0], [16]), [0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
    ) -> pl.Tensor[[16], pl.FP32]:
        result, tid = pl.intrin.spmd_direct(self.kernel, x, out, core_num=4)
        return result


class IntrinSpmdDirectCall:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
    ) -> pl.Tensor[[16], pl.FP32]:
        return pl.store(pl.load(x, [0], [16]), [0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
    ) -> pl.Tensor[[16], pl.FP32]:
        result = pl.intrin.spmd_direct_call(self.kernel, x, out, core_num=4)
        return result


@pytest.mark.parametrize(
    "program",
    [IntrinSpmdDirect, IntrinSpmdDirectCall],
    ids=["captured", "uncaptured"],
)
def test_direct_spmd_intrin_has_no_parser_spelling(program) -> None:
    """Captured and uncaptured direct SPMD calls are assembly-only forms."""
    trace_program(program)
    with pytest.raises(ParserError, match="intrin"):
        program_parser(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
