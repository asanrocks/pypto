# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Acceptance for the authoring runtime (``@pl.authoring.recipe``, ``@pl.authoring.body``).

The conformance gates of the runtime's contract
(``docs/en/dev/language/07-authoring-recipes.md``):

* **Splice semantics** — ``body.splice(...)`` is the borrowed collection:
  no function, no call, values cross back to the caller.
* **Callee compositions** — ``body.inline(...)`` / ``body.incore(...)`` are
  push / collect / finalize / call compositions: the program gains exactly
  one callee per (body, argument types), typed Inline / InCore.
* **Root parity** — a root recipe (invoke) reproduces the entrance
  foundation's root trace structurally.
* **The invariants** — one collection per unit, LIFO finalization, no
  unfinalized units at completion, stale handles, refused nesting, host
  calls, derived-signature discipline.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from pypto import backend
from pypto.backend import BackendType
from pypto.language.authoring import FunctionArtifact, ParamSpec
from pypto.language.tracing import entry as entrance
from pypto.language.tracing.entry.annotation import derive
from pypto.language.tracing.entry.declaration import Infer, fold_auto_scope
from pypto.pypto_core import ir

F32 = torch.float32


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure the backend the compile-for-test lane needs (the jit conftest's)."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    yield
    backend.reset_for_testing()


def _tensor() -> torch.Tensor:
    return torch.randn(64, 64, dtype=F32)


def _out() -> torch.Tensor:
    return torch.empty(64, 64, dtype=F32)


def _printed(program: ir.Program, entry: str) -> str:
    """A program's printed functions, with the entry's name normalized away."""
    printed = "\n".join(f"### {gv.name}\n{ir.python_print(fn)}" for gv, fn in program.functions.items())
    return printed.replace(entry, "ENTRY")


def _function(program: ir.Program, name: str) -> ir.Function:
    by_name = {gv.name: fn for gv, fn in program.functions.items()}
    assert name in by_name, f"{name} not in {sorted(by_name)}"
    return by_name[name]


# -----------------------------------------------------------------------------------------------------------
# The entrance contract
# -----------------------------------------------------------------------------------------------------------


class TestRecipeEntrance:
    def test_a_host_call_is_rejected(self):
        @pl.authoring.recipe
        def noop(ctx):
            return None

        with pytest.raises(TypeError, match="requires an active trace"):
            noop()

    def test_the_decorator_requires_a_plain_function(self):
        with pytest.raises(TypeError, match="expects a plain function"):
            pl.authoring.recipe(42)  # pyright: ignore[reportArgumentType] - a non-callable is the case under test

    def test_the_recipe_value_crosses_back_to_the_caller(self):
        @pl.authoring.recipe
        def twice(ctx, x):
            return pl.add(x, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = twice(a)
            return out

        bridged, _ = entry._bridge_args((_tensor(), _out()))
        program = entry._build_program(bridged)
        printed = _printed(program, "entry")
        assert "twice" not in {gv.name for gv in program.functions}
        assert "add" in printed


# -----------------------------------------------------------------------------------------------------------
# Splice — the borrowed collection
# -----------------------------------------------------------------------------------------------------------


class TestSpliceBody:
    def test_splice_is_the_expansion_semantics(self):
        @pl.authoring.body
        def scaled(x, k):
            return pl.mul(x, k)

        @pl.jit_trace
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = scaled.splice(a, 2.0)
            return out

        bridged, _ = entry._bridge_args((_tensor(), _out()))
        program = entry._build_program(bridged)
        assert sorted(gv.name for gv in program.functions) == ["entry"]
        printed = ir.python_print(_function(program, "entry"))
        assert "scaled" not in printed
        assert "mul" in printed

    def test_a_host_splice_is_rejected(self):
        @pl.authoring.body
        def scaled(x, k):
            return pl.mul(x, k)

        with pytest.raises(TypeError, match="requires an active trace"):
            scaled.splice(1.0, 2.0)

    def test_a_direct_body_call_names_the_placements(self):
        @pl.authoring.body
        def scaled(x, k):
            return pl.mul(x, k)

        with pytest.raises(TypeError, match="place it explicitly"):
            scaled(1.0, 2.0)

    def test_annotations_play_no_role(self):
        """A Python-only annotation still accepts a facade argument."""

        @pl.authoring.body
        def scaled(x: int | pl.Scalar | None, k: "float") -> "list[int]":  # pyright: ignore[reportArgumentType]
            return pl.mul(x, k)  # pyright: ignore[reportArgumentType] - the nonsense annotation is the point

        @pl.jit_trace
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = scaled.splice(a, 2.0)
            return out

        program = entry.compile_for_test(_tensor(), _out())
        assert "scaled" not in {gv.name for gv in program.functions}

    def test_a_host_literal_cannot_cross_staged_control_flow(self):
        @pl.authoring.body
        def accumulate(x):
            acc = 0
            for k in pl.range(4):
                acc = acc + k
            return acc

        @pl.jit_trace
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = pl.add(a, accumulate.splice(a))
            return out

        with pytest.raises(TypeError, match="staged"):
            entry.compile_for_test(_tensor(), _out())

    def test_a_value_rebound_in_a_scoped_loop_writes_through(self):
        """A caller value rebound inside a loop in a `pl.at` scope reaches the caller.

        The alias is what makes this work: a fresh `let` for the body's `o`
        would leave the enclosing loop with no carried channel to derive, and
        `ConvertToSSA` would emit an unversioned free var the verifier rejects.
        The emitted program is the hand-written body's, which is the point: a
        splice behaves like its text pasted at the placement.
        """

        @pl.authoring.body
        def accumulate_into(x, o):
            with pl.at(level=pl.Level.CORE_GROUP):
                for _k in pl.pipeline(2, stage=2):
                    o = pl.add(o, 1.0)
            return o

        @pl.jit_trace
        def via_splice(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            o = accumulate_into.splice(x, o)
            return o

        @pl.jit_trace
        def handwritten(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                for _k in pl.pipeline(2, stage=2):
                    o = pl.add(o, 1.0)
            return o

        spliced = via_splice.compile_for_test(_tensor(), _out())
        by_hand = handwritten.compile_for_test(_tensor(), _out())
        assert _printed(spliced, "via_splice") == _printed(by_hand, "handwritten")

    def test_auto_scope_false_matches_the_inline_spelling(self):
        """`splice(auto_scope=False)` and `inline(auto_scope=False)` emit one program.

        The body's ops must not gain compiler-placed scopes — the policy
        `tests/st/runtime/ops/test_matmul_nz.py` needs from its layer body.
        Like that site, these bodies write through their buffer and return
        nothing; a discarded splice return would stay behind as a no-op
        statement. The programs agree after the Default pipeline (the inline
        pass splices the callee at its call site).
        """

        def inline_program():
            @pl.jit_trace.inline(auto_scope=False)
            def body(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Tensor[[64, 64], pl.FP32]):
                with pl.at(level=pl.Level.CORE_GROUP):
                    pl.store(pl.add(pl.load(x, [0, 0], [64, 64]), 1.0), [0, 0], o)

            @pl.jit_trace
            def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
                body(a, out)
                return out

            return entry.compile_for_test(_tensor(), _out())

        def spliced_program():
            @pl.authoring.body
            def body(x, o):
                with pl.at(level=pl.Level.CORE_GROUP):
                    pl.store(pl.add(pl.load(x, [0, 0], [64, 64]), 1.0), [0, 0], o)

            @pl.jit_trace
            def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
                body.splice(a, out, auto_scope=False)
                return out

            return entry.compile_for_test(_tensor(), _out())

        assert ir.structural_equal(inline_program(), spliced_program(), enable_auto_mapping=False)

    def test_each_occurrence_collects_fresh(self):
        @pl.authoring.body
        def bump(x, k):
            y = pl.add(x, k)
            return pl.mul(y, k)

        @pl.jit_trace
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                one = bump.splice(a, 1.0)
                two = bump.splice(one, 2.0)
                out[:, :] = pl.add(one, two)
            return out

        bridged, _ = entry._bridge_args((_tensor(), _out()))
        program = entry._build_program(bridged)
        printed = ir.python_print(_function(program, "entry"))
        # Two occurrences of a two-op body: both scalar ops collected twice
        # (``adds``/``muls`` — the scalar-operand fold of add/mul).
        assert printed.count("adds") == 2
        assert printed.count("muls") == 2
        assert "bump" not in printed


# -----------------------------------------------------------------------------------------------------------
# The callee compositions
# -----------------------------------------------------------------------------------------------------------


class TestCalleeCompositions:
    def test_inline_pushes_collects_finalizes_and_calls(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = squared.inline(a)
            return out

        bridged, _ = entry._bridge_args((_tensor(), _out()))
        program = entry._build_program(bridged)
        names = sorted(gv.name for gv in program.functions)
        assert names == ["entry", "squared"]
        helper = _function(program, "squared")
        assert helper.func_type == ir.FunctionType.Inline
        printed = ir.python_print(_function(program, "entry"))
        assert "squared(" in printed

    def test_incore_types_the_callee(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            out[:, :] = squared.incore(a)
            return out

        bridged, _ = entry._bridge_args((_tensor(), _out()))
        program = entry._build_program(bridged)
        helper = _function(program, "squared")
        assert helper.func_type == ir.FunctionType.InCore
        assert helper.level == pl.Level.CHIP_DIE

    def test_incore_level_selects_the_specialized_kind(self):
        """An AIC/AIV level selects the specialized InCore function kind."""

        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            out[:, :] = squared.incore(a, level=pl.Level.AIV)
            return out

        bridged, _ = entry._bridge_args((_tensor(), _out()))
        program = entry._build_program(bridged)
        helper = _function(program, "squared")
        assert helper.func_type == ir.FunctionType.AIV
        assert helper.level == pl.Level.AIV

    def test_repeated_compositions_dedupe_to_one_callee(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                one = squared.inline(a)
                two = squared.inline(a)
                out[:, :] = pl.add(one, two)
            return out

        bridged, _ = entry._bridge_args((_tensor(), _out()))
        program = entry._build_program(bridged)
        assert sorted(gv.name for gv in program.functions) == ["entry", "squared"]

    def test_a_host_composition_is_rejected(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        with pytest.raises(TypeError, match="requires an active trace"):
            squared.inline(1.0)


# -----------------------------------------------------------------------------------------------------------
# The root composition
# -----------------------------------------------------------------------------------------------------------


class TestRootComposition:
    def _contract_from_decl(self, decl) -> pl.authoring.FunctionContract:
        returns = tuple(decl.returns) if decl.returns is not Infer else None
        return pl.authoring.FunctionContract(
            name=decl.name,
            function_type=decl.function_type,
            level=decl.level,
            role=decl.role,
            attrs=fold_auto_scope(decl.attrs, decl.auto_scope),
            params=tuple(ParamSpec(p.name, p.type, p.direction) for p in decl.params),
            returns=returns,
            strict_ssa=decl.strict_ssa,
            requires_runtime_binding=decl.requires_runtime_binding,
        )

    def test_a_root_recipe_reproduces_the_foundation_trace(self):
        T = pl.Tensor[[8, 8], pl.FP32]

        def kernel(a: T, b: T) -> T:  # pyright: ignore[reportInvalidTypeForm]
            return pl.add(a, b)

        plan = derive(kernel)
        reference = entrance.TracedCallable.from_callable(kernel, decl=plan.decl).invoke(
            args=plan.args, kwargs=plan.kwargs
        )

        @pl.authoring.recipe
        def entry_recipe(ctx, body):
            fn = ctx.current
            artifact = fn.finalize(fn.collect_body(body))
            return ctx.materialize_entry(artifact)

        artifact = entry_recipe.invoke(self._contract_from_decl(plan.decl), kernel)
        assert isinstance(artifact, FunctionArtifact)
        assert artifact.is_entry
        assert ir.structural_equal(artifact.function, reference, enable_auto_mapping=True)

    def test_invoke_refuses_an_active_trace(self):
        @pl.authoring.recipe
        def entry_recipe(ctx, body):
            fn = ctx.current
            return ctx.materialize_entry(fn.finalize(fn.collect_body(body)))

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                with pytest.raises(TypeError, match="cannot run inside an active trace"):
                    entry_recipe.invoke(pl.authoring.FunctionContract(name="x"))
                out[:, :] = pl.add(a, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))


# -----------------------------------------------------------------------------------------------------------
# Region composition — the shared session
# -----------------------------------------------------------------------------------------------------------


class TestRecipeRegionComposition:
    def test_a_recipe_composes_construct_regions(self):
        """A general recipe composes a construct region (``ctx.region``) and
        collects a body inside it — the construct tier's composition driving
        the same session state a construct recipe drives."""

        @pl.authoring.body
        def bump(x, k):
            y = pl.mul(x, k)
            return pl.add(y, k)

        @pl.authoring.recipe
        def scoped(ctx, body, x, k):
            with ctx.region(pl.scope()):
                return ctx.splice_body(body, x, k)

        @pl.jit_trace(auto_scope=False)
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            out[:, :] = scoped(bump, a, 2.0)
            return out

        bridged, _ = entry._bridge_args((_tensor(), _out()))
        program = entry._build_program(bridged)
        printed = ir.python_print(_function(program, "entry"))
        # The spliced body's own binding lands inside the composed region;
        # the returned expression binds at the caller's position.
        scope_body = printed.split("with pl.scope():")[1].split("out:")[0]
        assert "muls" in scope_body
        assert "adds" not in scope_body

    def test_a_recipe_leaving_a_region_open_is_loud(self):
        """A leaked region aborts the enclosing trace (failure is atomic):
        the recipe's regions close before it returns, and the one that did
        not takes the whole one-shot trace with it."""

        @pl.authoring.recipe
        def leaky(ctx):
            ctx.enter(pl.scope())

        @pl.jit_trace(auto_scope=False)
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            out[:, :] = leaky()
            return out

        with pytest.raises(TypeError, match=r"region\(s\) open"):
            entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))


# -----------------------------------------------------------------------------------------------------------
# The invariants
# -----------------------------------------------------------------------------------------------------------


class TestInvariants:
    def test_one_collection_per_unit(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        seen: dict[str, object] = {}

        @pl.authoring.recipe
        def bad(ctx, body, x):
            fn = ctx.push_function(
                pl.authoring.FunctionContract(name="dup", function_type=pl.FunctionType.Inline)
            )
            fn.collect_body(body, args=(x,))
            with pytest.raises(TypeError, match="one collection per unit"):
                fn.collect_body(body, args=(x,))
            fn.finalize()
            seen["ran"] = True
            return pl.add(x, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = bad(squared, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))
        assert seen["ran"] is True

    def test_finalize_is_once(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        @pl.authoring.recipe
        def bad(ctx, body, x):
            fn = ctx.push_function(
                pl.authoring.FunctionContract(name="dup", function_type=pl.FunctionType.Inline)
            )
            artifact = fn.finalize(fn.collect_body(body, args=(x,)))
            with pytest.raises(TypeError, match="already finalized"):
                fn.finalize()
            return ctx.emit_call(artifact, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = bad(squared, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_a_recipe_cannot_complete_owning_a_unit(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        @pl.authoring.recipe
        def leaky(ctx, body, x):
            fn = ctx.push_function(
                pl.authoring.FunctionContract(name="leak", function_type=pl.FunctionType.Inline)
            )
            fn.collect_body(body, args=(x,))
            return x  # never finalized

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                with pytest.raises(TypeError, match="unfinalized"):
                    out[:, :] = leaky(squared, a)
                out[:, :] = pl.add(a, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_finalize_is_lifo(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        @pl.authoring.recipe
        def bad(ctx, body, x):
            outer = ctx.push_function(
                pl.authoring.FunctionContract(name="outer", function_type=pl.FunctionType.Inline)
            )
            inner = ctx.push_function(
                pl.authoring.FunctionContract(name="inner", function_type=pl.FunctionType.Inline)
            )
            inner_collected = inner.collect_body(body, args=(x,))
            outer_collected = outer.collect_body(body, args=(x,))
            with pytest.raises(TypeError, match="LIFO"):
                outer.finalize(outer_collected)
            inner.finalize(inner_collected)
            outer.finalize(outer_collected)
            return pl.add(x, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = bad(squared, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_a_stale_handle_is_loud(self):
        stash: dict[str, Any] = {}

        @pl.authoring.recipe
        def stash_ctx(ctx):
            stash["ctx"] = ctx

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                stash_ctx()
                out[:, :] = pl.add(a, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))
        stale_ctx = stash["ctx"]
        with pytest.raises(RuntimeError, match="session has ended"):
            stale_ctx.push_function(pl.authoring.FunctionContract(name="late"))

    def test_derived_signatures_reject_host_values(self):
        @pl.authoring.body
        def scaled(x, k):
            return pl.mul(x, k)

        @pl.authoring.recipe
        def bad(ctx, body, x):
            fn = ctx.push_function(
                pl.authoring.FunctionContract(name="bad", function_type=pl.FunctionType.Inline)
            )
            with pytest.raises(TypeError, match="host value"):
                fn.collect_body(body, args=(x, 2.0))
            fn.discard()
            return pl.add(x, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = bad(scaled, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_nested_splice_is_refused(self):
        @pl.authoring.body
        def inner(x):
            return pl.mul(x, x)

        @pl.authoring.recipe
        def outer(ctx, x):
            # A recipe called from inside a splice body — its own splice is the
            # nested expansion, refused exactly as the entrance always refused.
            with pytest.raises(TypeError, match="already active"):
                ctx.splice_body(inner, args=(x,))
            return pl.add(x, x)

        @pl.authoring.body
        def caller(x):
            return outer(x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = caller.splice(a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_a_collected_unit_cannot_discard(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        @pl.authoring.recipe
        def bad(ctx, body, x):
            fn = ctx.push_function(
                pl.authoring.FunctionContract(name="keep", function_type=pl.FunctionType.Inline)
            )
            fn.collect_body(body, args=(x,))
            with pytest.raises(TypeError, match="collected a body"):
                fn.discard()
            fn.finalize()
            return pl.add(x, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = bad(squared, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_declared_abi_rejects_caller_values_at_collect(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        contract = pl.authoring.FunctionContract(
            name="decl",
            function_type=pl.FunctionType.Inline,
            params=(pl.authoring.ParamSpec("x", ir.TensorType([8, 8], pl.FP32)),),
        )

        @pl.authoring.recipe
        def bad(ctx, body, x):
            fn = ctx.push_function(contract)
            with pytest.raises(TypeError, match="declared-ABI unit binds its own"):
                fn.collect_body(body, args=(x,))
            fn.discard()
            return pl.add(x, x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = bad(squared, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))


# -----------------------------------------------------------------------------------------------------------
# Body collection semantics — the pl.authoring.body contract
# -----------------------------------------------------------------------------------------------------------


class TestBodyCollectionSemantics:
    def test_each_occurrence_gets_independent_locals(self):
        """A rebound local in one occurrence is invisible to the next — every
        collection runs in a fresh lexical frame."""

        @pl.authoring.body
        def step(x, k):
            acc = pl.add(x, k)
            acc = pl.mul(acc, k)  # the rebind dies with this occurrence's frame
            return acc

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                one = step.splice(a, 1.0)
                two = step.splice(a, 2.0)
                out[:, :] = pl.add(one, two)
            return out

        bridged, _ = entry._bridge_args((torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32)))
        program = entry._build_program(bridged)
        printed = ir.python_print(_function(program, "entry"))
        # Both occurrences collect the same two-op body; no state crosses.
        assert printed.count("adds") == 2
        assert printed.count("muls") == 2

    def test_closures_and_definition_site_globals_stay_live(self):
        scale = 3.0  # the closure cell the body reads

        def make_body():
            factor = 2.0  # the enclosing frame's cell

            @pl.authoring.body
            def weighted(x):
                return pl.mul(x, scale * factor)

            return weighted

        weighted = make_body()

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = weighted.splice(a)
            return out

        bridged, _ = entry._bridge_args((torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32)))
        program = entry._build_program(bridged)
        printed = ir.python_print(_function(program, "entry"))
        # The closure arithmetic folded to the host constant 6.0.
        assert "muls" in printed

    def test_a_body_may_compose_callees_inside_a_splice(self):
        """A spliced body runs instrumented: its own calls may place other
        bodies (a splice body pushing an inline callee is legal — only a
        nested *splice* is refused)."""

        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        @pl.authoring.body
        def weighted(x):
            return squared.inline(x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = weighted.splice(a)
            return out

        bridged, _ = entry._bridge_args((torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32)))
        program = entry._build_program(bridged)
        assert sorted(gv.name for gv in program.functions) == ["entry", "squared"]

    def test_a_self_splicing_body_hits_the_nesting_guard(self):
        """Direct self-splice recursion is the nested expansion — the splice's
        own guard refuses it before the body re-enters."""

        @pl.authoring.body
        def recursive(x):
            return recursive.splice(x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pytest.raises(TypeError, match="already active"):
                out[:, :] = recursive.splice(a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_a_self_composing_body_is_refused(self):
        """A body composing *itself* as a callee (inline inside its own
        collection) is the collecting guard's case: the trace would recurse
        with no staged condition to stop it."""

        @pl.authoring.body
        def recursive(x):
            return recursive.inline(x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pytest.raises(TypeError, match="cannot collect itself"):
                out[:, :] = recursive.splice(a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_a_terminated_region_rejects_region_composition(self):
        @pl.authoring.recipe
        def return_then_enter(ctx, out):
            ctx.control.emit_return(out)
            cm = ctx.enter(pl.scope())
            ctx.exit(cm)

        @pl.jit_trace(auto_scope=False)
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pytest.raises(TypeError, match="terminated by an emitted return"):
                return_then_enter(out)
            return out

        entry.compile_for_test(_small_tensor(), _small_out())

    def test_a_terminated_region_rejects_scope_opening(self):
        @pl.authoring.recipe
        def return_then_open(ctx, out):
            ctx.control.emit_return(out)
            region = ctx.open_scope(ir.ScopeKind.Runtime, attrs=[], defer=False, manual=True)
            ctx.exit(region)

        @pl.jit_trace(auto_scope=False)
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pytest.raises(TypeError, match="terminated by an emitted return"):
                return_then_open(out)
            return out

        entry.compile_for_test(_small_tensor(), _small_out())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# -----------------------------------------------------------------------------------------------------------
# Control capabilities — ctx.control
# -----------------------------------------------------------------------------------------------------------


class TestControlCapabilities:
    def test_emit_return_inside_a_staged_branch_is_a_conditional_path(self):
        @pl.authoring.recipe
        def maybe_return(ctx, x):
            ctx.control.emit_return(x)

        @pl.jit_trace
        def entry(
            a: pl.Tensor[[64, 64], pl.FP32],
            b: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                if flag:
                    maybe_return(a)
                out[:, :] = b
            return out

        t = torch.randn(64, 64, dtype=F32)
        o = torch.empty(64, 64, dtype=F32)
        bridged, _ = entry._bridge_args((t, t, o, True))
        program = entry._build_program(bridged)
        printed = ir.python_print(_function(program, "entry"))
        # The emitted return is one branch's path; the body continues after it.
        branch = printed.split("if flag:")[1].split("out:")[0]
        assert "return a" in branch
        assert "assemble" in printed

    def test_emit_return_at_top_level_terminates_the_region(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        @pl.authoring.recipe
        def bail(ctx, x):
            ctx.control.emit_return(x)

        @pl.authoring.recipe
        def late(ctx, body, x):
            fn = ctx.push_function(
                pl.authoring.FunctionContract(name="late", function_type=pl.FunctionType.Inline)
            )
            return ctx.emit_call(fn.finalize(fn.collect_body(body, args=(x,))), x)

        @pl.jit_trace
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                bail(a)
                with pytest.raises(TypeError, match="terminated"):
                    late(squared, a)
            return out

        # The raw build is the acceptance: the trace completed with the
        # emitted return and the rejected dead effect. (The pass pipeline's
        # outlined form of a top-level emitted return is a separate lowering
        # concern, not this contract.)
        bridged, _ = entry._bridge_args((torch.randn(64, 64, dtype=F32), torch.empty(64, 64, dtype=F32)))
        entry._build_program(bridged)

    def test_emit_yield_emits_the_staged_while_backedge(self):
        @pl.authoring.recipe
        def backedge(ctx):
            ctx.control.emit_yield()

        @pl.jit_trace
        def entry(a: pl.Tensor[[256, 64], pl.FP32], out: pl.Out[pl.Tensor[[256, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                i = pl.const(0, pl.INDEX)
                while i < 4:
                    off = i * 64
                    ta = pl.load(a, [off, 0], [64, 64])
                    pl.store(ta, [off, 0], out)
                    i = i + 1
                    backedge()
            return out

        t = torch.randn(256, 64, dtype=F32)
        o = torch.empty(256, 64, dtype=F32)
        bridged, _ = entry._bridge_args((t, o))
        program = entry._build_program(bridged)
        printed = ir.python_print(_function(program, "entry"))
        assert "while i < 4:" in printed
        assert "pl.yield_()" in printed

    def test_emit_break_and_continue_target_the_staged_loop(self):
        @pl.authoring.recipe
        def bail(ctx):
            ctx.control.emit_break()

        @pl.authoring.recipe
        def skip(ctx):
            ctx.control.emit_continue()

        @pl.jit_trace
        def entry(a: pl.Tensor[[256, 64], pl.FP32], out: pl.Out[pl.Tensor[[256, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                for i in pl.range(4):
                    off = i * 64
                    ta = pl.load(a, [off, 0], [64, 64])
                    pl.store(ta, [off, 0], out)
                    skip()
                    bail()
            return out

        t = torch.randn(256, 64, dtype=F32)
        o = torch.empty(256, 64, dtype=F32)
        bridged, _ = entry._bridge_args((t, o))
        program = entry._build_program(bridged)
        printed = ir.python_print(_function(program, "entry"))
        assert "continue" in printed
        assert "break" in printed

    def test_emit_break_on_a_host_loop_is_refused(self):
        @pl.authoring.recipe
        def bail(ctx):
            ctx.control.emit_break()

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                for name in ("one", "two"):  # a host loop: no staged target
                    with pytest.raises(TypeError, match="host loop"):
                        bail()
                out[:, :] = pl.add(a, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_emit_yield_with_no_target_is_loud(self):
        @pl.authoring.recipe
        def stray(ctx):
            ctx.control.emit_yield()

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                with pytest.raises(TypeError, match="no region accepts"):
                    stray()
                out[:, :] = pl.add(a, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_emit_break_and_continue_with_no_loop_open_are_loud(self):
        @pl.authoring.recipe
        def bail(ctx):
            ctx.control.emit_break()

        @pl.authoring.recipe
        def skip(ctx):
            ctx.control.emit_continue()

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                with pytest.raises(TypeError, match="no loop is open"):
                    bail()
                with pytest.raises(TypeError, match="no loop is open"):
                    skip()
                out[:, :] = pl.add(a, a)
            return out

        entry.compile_for_test(torch.randn(8, 8, dtype=F32), torch.empty(8, 8, dtype=F32))

    def test_emit_return_parity_with_the_native_return_in_a_collected_body(self):
        @pl.authoring.body
        def native_body(x):
            return pl.mul(x, x)

        @pl.authoring.recipe
        def ret(ctx, x):
            ctx.control.emit_return(x)

        @pl.authoring.body
        def emitted_body(x):
            ret(pl.mul(x, x))

        @pl.authoring.recipe
        def squared(ctx, body, x):
            fn = ctx.push_function(
                pl.authoring.FunctionContract(name="squared", function_type=pl.FunctionType.Inline)
            )
            return ctx.emit_call(fn.finalize(fn.collect_body(body, args=(x,))), x)

        @pl.jit_trace
        def native(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = squared(native_body, a)
            return out

        @pl.jit_trace
        def emitted(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = squared(emitted_body, a)
            return out

        printed = []
        for fn in (native, emitted):
            bridged, _ = fn._bridge_args((_tensor(), _out()))
            printed.append(_printed(fn._build_program(bridged), fn.__name__))
        # The emitted return lowers to the same IR as the native ``return``
        # in a collected body — entry and callee, byte for byte.
        assert printed[0] == printed[1]


# -----------------------------------------------------------------------------------------------------------
# The dispatch effect
# -----------------------------------------------------------------------------------------------------------


@pl.function(type=pl.FunctionType.InCore)
def _dispatch_leaf(
    x: pl.Tensor[[8, 8], pl.FP32], out: pl.Tensor[[8, 8], pl.FP32]
) -> pl.Tensor[[8, 8], pl.FP32]:
    """The submitted callee: a marked InCore kernel."""
    t = pl.add(pl.load(x, [0, 0], [8, 8]), pl.load(x, [0, 0], [8, 8]))
    return pl.store(t, [0, 0], out)


def _small_tensor() -> torch.Tensor:
    return torch.randn(8, 8, dtype=F32)


def _small_out() -> torch.Tensor:
    return torch.empty(8, 8, dtype=F32)


# -----------------------------------------------------------------------------------------------------------
# The dispatch effect
# -----------------------------------------------------------------------------------------------------------


@pl.function(type=pl.FunctionType.InCore)
def _dispatch_leaf(
    x: pl.Tensor[[8, 8], pl.FP32], out: pl.Tensor[[8, 8], pl.FP32]
) -> pl.Tensor[[8, 8], pl.FP32]:
    """The submitted callee: a marked InCore kernel."""
    t = pl.add(pl.load(x, [0, 0], [8, 8]), pl.load(x, [0, 0], [8, 8]))
    return pl.store(t, [0, 0], out)


def _small_tensor() -> torch.Tensor:
    return torch.randn(8, 8, dtype=F32)


def _small_out() -> torch.Tensor:
    return torch.empty(8, 8, dtype=F32)


class TestDispatchEffect:
    """``ctx.submit`` — the dispatch family's recipe spelling over the one effect."""

    def test_the_capability_traces_identically_to_the_stdlib_spelling(self):
        @pl.authoring.recipe
        def dispatch_one(ctx, callee, a, out):
            return ctx.submit(callee, a, out)

        @pl.jit_trace
        def via_capability(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            res, tid = dispatch_one(_dispatch_leaf, a, out)
            return out

        @pl.jit_trace
        def via_stdlib(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            res, tid = pl.submit(_dispatch_leaf, a, out)
            return out

        p1 = via_capability._build_program(via_capability._bridge_args((_small_tensor(), _small_out()))[0])
        p2 = via_stdlib._build_program(via_stdlib._bridge_args((_small_tensor(), _small_out()))[0])
        f1 = _function(p1, "via_capability")
        f2 = _function(p2, "via_stdlib")
        assert ir.structural_equal(f1, f2, enable_auto_mapping=True)
        assert "submit" in ir.python_print(f1)

    def test_the_spmd_launch_spec_crosses_the_capability(self):
        @pl.authoring.recipe
        def launch(ctx, callee, a, out):
            return ctx.submit(callee, a, out, core_num=4, sync_start=True)

        @pl.jit_trace
        def via_capability(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            res, tid = launch(_dispatch_leaf, a, out)
            return out

        @pl.jit_trace
        def via_stdlib(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            res, tid = pl.spmd_submit(_dispatch_leaf, a, out, core_num=4, sync_start=True)
            return out

        p1 = via_capability._build_program(via_capability._bridge_args((_small_tensor(), _small_out()))[0])
        p2 = via_stdlib._build_program(via_stdlib._bridge_args((_small_tensor(), _small_out()))[0])
        f1 = _function(p1, "via_capability")
        f2 = _function(p2, "via_stdlib")
        assert ir.structural_equal(f1, f2, enable_auto_mapping=True)
        printed = ir.python_print(f1)
        assert "core_num=4" in printed and "sync_start=True" in printed

    def test_the_capability_wires_dependencies_from_a_captured_task_id(self):
        @pl.authoring.recipe
        def dispatch_two(ctx, callee, a, out):
            first = ctx.bind("first_tid", ctx.submit(callee, a, out)[1])
            ctx.bind("second_tid", ctx.submit(callee, a, out, deps=[first])[1])

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            dispatch_two(_dispatch_leaf, a, out)
            return out

        p1 = entry._build_program(entry._bridge_args((_small_tensor(), _small_out()))[0])
        f1 = _function(p1, "entry")
        # The wiring, not the spelling: the second dispatch's one dep edge is
        # the first dispatch's producer TaskId (the ``first_tid`` binding).
        assert isinstance(f1.body, ir.SeqStmts)
        submits = [
            s.value for s in f1.body.stmts if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Submit)
        ]
        assert len(submits) == 2, ir.python_print(f1)
        assert len(submits[1].deps) == 1, ir.python_print(f1)
        dep = submits[1].deps[0]
        assert isinstance(dep, ir.Var) and dep.name_hint == "first_tid"
        assert "deps=[first_tid]" in ir.python_print(f1)

    def test_a_terminated_region_rejects_the_dispatch(self):
        @pl.authoring.recipe
        def return_then_dispatch(ctx, callee, a, out):
            ctx.control.emit_return(out)
            ctx.submit(callee, a, out)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pytest.raises(TypeError, match="terminated by an emitted return"):
                return_then_dispatch(_dispatch_leaf, a, out)
            return out

        entry.compile_for_test(_small_tensor(), _small_out())

    def test_an_unmarked_callee_is_loud_under_the_capability_name(self):
        @pl.authoring.recipe
        def dispatch_one(ctx, callee, a, out):
            return ctx.submit(callee, a, out)

        @pl.jit_trace
        def entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pytest.raises(TypeError, match="ctx.submit first argument must be a marked sub-kernel"):
                dispatch_one(42, a, out)  # pyright: ignore[reportArgumentType] - the refusal is the case
            return out

        entry.compile_for_test(_small_tensor(), _small_out())
