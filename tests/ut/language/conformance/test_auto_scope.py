# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The ``auto_scope=`` retirement test.

The ``@collect_todo`` on ``pl.function(auto_scope=)`` retires when the
traced orchestration main's scope-insertion IR is structurally equal to
the AST anchor's ``PTO2_SCOPE`` shape, in both modes:

- **default** (``auto_scope=True``): the pass-side
  ``materialize_runtime_scopes`` owns AUTO placement — it reads
  ``FunctionType.Orchestration`` and inserts the AUTO
  ``RuntimeScopeStmt`` (function body + for/if bodies).
- **opt-out** (``auto_scope=False``): ``fold_auto_scope`` (the entrance
  foundation's materialization rule) records
  ``attrs["auto_scope"] = False``, the pass skips insertion, and the
  hand-placed ``with pl.scope():`` survives.

Neither frontend inserts scopes at parse time — the AST anchor also
relies on the pass — so the conformance is asserted **post-pass** via
``assert_structural_equal`` vs the AST anchor.

The func_type gap that blocked retirement — the AST
``@pl.function`` returning class methods unchanged, so
``trace_program``'s raw-method branch re-marked them ``Opaque`` — is
closed by ``_mark_from_raw_method`` recovering the stashed
``_pl_function_class`` (``@pl.function(type=Orchestration)`` on a plain
program class now threads, matching the tracing frontend's ``Marker`` path).
"""

import pypto.language as pl
from pypto import ir
from pypto.language.parser import program_parser
from pypto.language.tracer.decorators.function import trace_program
from pypto.pypto_core import passes


def _scope_insertion_pair(cls):
    """Build both frontends' program and run the scope-insertion pair:
    ``derive_call_directions`` (the pass's pre-condition — it stamps
    ``arg_directions`` on calls, which ``materialize_runtime_scopes``
    pre-verification requires) then the pass itself. The Default pipeline
    runs the same pair (plus more); this keeps the assertion on exactly
    the scope-insertion IR, without dragging in the
    unrelated roundtrip-instrument gap (``pl.compat.ref`` after
    InitMemRef — a pre-existing baseline failure)."""

    def run(prog):
        prog = passes.derive_call_directions()(prog)
        return passes.materialize_runtime_scopes()(prog)

    return run(trace_program(cls)), run(program_parser(cls))


def test_auto_scope_default_matches_ast_anchor():
    """Default mode: the pass inserts the AUTO scope for the traced main
    exactly as for the AST anchor (the func_type gap would make this fail —
    an Opaque main gets no scope)."""

    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return x

    traced, ast_ = _scope_insertion_pair(Prog)
    ir.assert_structural_equal(traced, ast_, enable_auto_mapping=True)
    # Belt-and-suspenders: the pass really inserted the AUTO scope.
    main = next(f for f in traced.functions.values() if f.name == "main")
    assert main.func_type == ir.FunctionType.Orchestration
    assert "pl.scope()" in str(main)  # the pass really inserted the AUTO scope


def test_auto_scope_opt_out_matches_ast_anchor():
    """Opt-out mode: ``auto_scope=False`` skips the pass insertion and the
    hand-placed ``pl.scope()`` survives — identical on both frontends."""

    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            with pl.scope():
                return x

    traced, ast_ = _scope_insertion_pair(Prog)
    ir.assert_structural_equal(traced, ast_, enable_auto_mapping=True)
    main = next(f for f in traced.functions.values() if f.name == "main")
    assert main.attrs.get("auto_scope") is False


def test_auto_scope_threads_subworker_function_type():
    """The func_type gap regression: the raw-method branch of
    ``trace_program`` must recover the AST decorator's classification — an
    InCore sub-worker stays InCore, an Orchestration main stays
    Orchestration (previously both re-marked Opaque)."""

    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return self.kernel(x)

    traced = trace_program(Prog)
    types = {f.name: f.func_type for f in traced.functions.values()}
    assert types == {"kernel": ir.FunctionType.InCore, "main": ir.FunctionType.Orchestration}


def test_auto_scope_wraps_loop_bodies():
    """The pass wraps for/if bodies too: a loop-carried orchestration main
    matches the AST anchor post-pass (the ``RuntimeScopeStmt`` is
    transparent to SSA — the pass runs dead last)."""

    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            acc = x
            for i in pl.range(4):
                acc = self.kernel(acc)
            return acc

    traced, ast_ = _scope_insertion_pair(Prog)
    ir.assert_structural_equal(traced, ast_, enable_auto_mapping=True)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
