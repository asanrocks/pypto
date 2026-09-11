# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Assembly-only direct SPMD scope construction."""

import pypto.language as pl
import pytest
from pypto.language.tracer.decorators.function import trace_program
from pypto.pypto_core import ir


def _main(program):
    return next(function for function in program.functions.values() if function.name == "main")


def _top_level_stmts(function):
    return list(function.body.stmts) if isinstance(function.body, ir.SeqStmts) else [function.body]


def _program(*, capture: bool):
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            tile = pl.load(a, [0], [64])
            return pl.store(tile, [0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            if capture:
                out, tid = pl.intrin.spmd_direct(self.kernel, a, out, core_num=4, name_hint="direct")
            else:
                out = pl.intrin.spmd_direct_call(self.kernel, a, out, core_num=4, name_hint="direct")
            return out

    return trace_program(Prog)


def test_spmd_direct_emits_captured_direct_scope():
    """The assembly intrinsic emits Spmd(Call), not InCore or Submit."""
    main = _main(_program(capture=True))
    stmts = _top_level_stmts(main)
    scope = next(stmt for stmt in stmts if isinstance(stmt, ir.SpmdScopeStmt))

    assert "task_id_var" in scope.attrs
    assert scope.name_hint == "direct"
    assert isinstance(scope.core_num, ir.ConstInt)
    assert scope.core_num.value == 4
    assert not isinstance(scope.body, ir.InCoreScopeStmt)
    assert isinstance(scope.body, ir.AssignStmt)
    assert isinstance(scope.body.value, ir.Call)
    assert isinstance(scope.body.value.op, ir.GlobalVar)
    assert scope.body.value.op.name == "kernel"
    assert not any(isinstance(stmt.value, ir.Submit) for stmt in stmts if isinstance(stmt, ir.AssignStmt))


def test_spmd_direct_call_emits_uncaptured_direct_scope():
    """The no-capture sibling leaves scheduling metadata absent."""
    main = _main(_program(capture=False))
    (scope,) = [stmt for stmt in _top_level_stmts(main) if isinstance(stmt, ir.SpmdScopeStmt)]

    assert "task_id_var" not in scope.attrs
    assert isinstance(scope.body, ir.AssignStmt)
    assert isinstance(scope.body.value, ir.Call)
    assert not isinstance(scope.body, ir.InCoreScopeStmt)


def test_spmd_direct_keeps_dependencies_on_the_outer_scope():
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            tile = pl.load(a, [0], [64])
            return pl.store(tile, [0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[64], pl.FP32],
            out0: pl.Out[pl.Tensor[[64], pl.FP32]],
            out1: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            out0, direct_tid = pl.intrin.spmd_direct(self.kernel, a, out0, core_num=4)
            out1, _ = pl.intrin.spmd_direct(self.kernel, a, out1, core_num=4, deps=[direct_tid])
            return out1

    stmts = _top_level_stmts(_main(trace_program(Prog)))
    scopes = [stmt for stmt in stmts if isinstance(stmt, ir.SpmdScopeStmt)]
    assert len(scopes) == 2
    assert scopes[1].attrs["manual_dep_edges"] == [scopes[0].attrs["task_id_var"]]
    assert not isinstance(scopes[1].body, ir.InCoreScopeStmt)


def test_spmd_direct_rejects_non_device_callee():
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y, _ = pl.intrin.spmd_direct(self.helper, x, core_num=4)
            return y

    with pytest.raises(TypeError, match="InCore/AIC/AIV"):
        trace_program(Prog)
