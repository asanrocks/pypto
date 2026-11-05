# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``@pl.jit_trace.body`` — a reusable statement fragment called like a function.

The kernel-author spelling of the splice placement. These tests pin both halves
of its contract: the expansion it performs (the same statements the caller
would have pasted, no callee), and its one design constraint — it is a *policy*
over the public authoring tier, so an external framework can build the same
spelling from ``pl.authoring`` without touching the tracer.
"""

import pypto.language as pl
import pytest
import torch
from pypto import ir

ROWS = 64
COLS = 64
TILE_ROWS = 16


def _tensor():
    return torch.zeros(ROWS, COLS, dtype=torch.float32)


def _out():
    return torch.zeros(ROWS, COLS, dtype=torch.float32)


def _program_text(kernel, *args) -> str:
    return ir.python_print(kernel.compile_for_test(*args))


class TestBodyExpansion:
    """A call expands the body's statements in the caller."""

    def test_a_body_call_is_the_pasted_statements(self):
        @pl.jit_trace.body
        def scale_block(a, out, row0, k):
            tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(tile, k), [row0, 0], out)

        @pl.jit_trace
        def via_body(
            a: pl.Tensor[[ROWS, COLS], pl.FP32],
            k: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                for chunk in pl.range(ROWS // TILE_ROWS):
                    row0 = chunk * TILE_ROWS
                    scale_block(a, out, row0, k)
            return out

        @pl.jit_trace
        def pasted(
            a: pl.Tensor[[ROWS, COLS], pl.FP32],
            k: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                for chunk in pl.range(ROWS // TILE_ROWS):
                    row0 = chunk * TILE_ROWS
                    tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
                    pl.store(pl.mul(tile, k), [row0, 0], out)
            return out

        bridged = _program_text(via_body, _tensor(), 2.0, _out())
        written_out = _program_text(pasted, _tensor(), 2.0, _out())
        assert bridged.replace("via_body", "P") == written_out.replace("pasted", "P"), (
            "a body call must lower to the statements it contains"
        )

    def test_no_callee_is_named_after_the_body(self):
        """The body is expanded: the only functions are the caller and whatever
        the caller's own scopes outline. A ``pl.at`` region becomes a device
        function, but the body never becomes one of its own."""

        @pl.jit_trace.body
        def step(a, out, row0):
            tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
            pl.store(tile, [row0, 0], out)

        @pl.jit_trace
        def entry(
            a: pl.Tensor[[ROWS, COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                step(a, out, 0)
            return out

        program = entry.compile_for_test(_tensor(), _out())
        names = sorted(gv.name for gv in program.functions)
        assert "step" not in names, names
        kinds = sorted(fn.func_type.name for _gv, fn in program.functions.items())
        assert kinds == ["AIV", "Orchestration"], kinds

    def test_a_body_may_return_a_staged_value(self):
        """The return crosses back into the caller's expression position."""

        @pl.jit_trace.body
        def scaled(x, k):
            return pl.mul(x, k)

        @pl.jit_trace
        def entry(
            a: pl.Tensor[[ROWS, COLS], pl.FP32],
            k: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [ROWS, COLS])
                pl.store(scaled(tile, k), [0, 0], out)
            return out

        printed = _program_text(entry, _tensor(), 2.0, _out())
        assert "mul" in printed
        assert "scaled" not in printed

    def test_a_body_cannot_call_another_body(self):
        """The tier admits one splice at a time; the error speaks body, not splice."""

        @pl.jit_trace.body
        def scaled(x, k):
            return pl.mul(x, k)

        @pl.jit_trace.body
        def doubled(x):
            return scaled(scaled(x, 2.0), 2.0)

        @pl.jit_trace
        def entry(
            a: pl.Tensor[[ROWS, COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [ROWS, COLS])
                pl.store(doubled(tile), [0, 0], out)
            return out

        with pytest.raises(TypeError, match="cannot be called from inside another body"):
            entry.compile_for_test(_tensor(), _out())

    def test_a_body_binds_its_caller_keywords(self):
        """Keyword arguments bind by the body's parameter names."""

        @pl.jit_trace.body
        def scaled(x, k):
            return pl.mul(x, k)

        @pl.jit_trace
        def entry(
            a: pl.Tensor[[ROWS, COLS], pl.FP32],
            k: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [ROWS, COLS])
                pl.store(scaled(x=tile, k=k), [0, 0], out)
            return out

        assert "mul" in _program_text(entry, _tensor(), 2.0, _out())


class TestBodyContract:
    """The declaration-time surface and its refusals."""

    def test_a_host_call_names_the_expansion_rule(self):
        @pl.jit_trace.body
        def scaled(x, k):
            return pl.mul(x, k)

        with pytest.raises(TypeError, match="expands statements"):
            scaled(1.0, 2.0)

    def test_auto_scope_is_declared_not_passed(self):
        """Placement policy belongs to the declaration: one meaning per fragment."""

        @pl.jit_trace.body
        def scaled(x, k):
            return pl.mul(x, k)

        @pl.jit_trace
        def entry(
            a: pl.Tensor[[ROWS, COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:, :] = scaled(a, 2.0, auto_scope=False)
            return out

        with pytest.raises(TypeError, match="auto_scope is declared on the body"):
            entry.compile_for_test(_tensor(), _out())

    def test_the_parenthesized_form_declares_the_policy(self):
        """``auto_scope=False`` on the declaration keeps a hand-scoped body legal
        when the caller declares no scope of its own."""

        @pl.jit_trace.body(auto_scope=False)
        def hand_scoped(a, out, k):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [ROWS, COLS])
                pl.store(pl.mul(tile, k), [0, 0], out)

        @pl.jit_trace
        def entry(
            a: pl.Tensor[[ROWS, COLS], pl.FP32],
            k: pl.Scalar[pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ):
            hand_scoped(a, out, k)
            return out

        program = entry.compile_for_test(_tensor(), 2.0, _out())
        kinds = sorted(fn.func_type.name for _gv, fn in program.functions.items())
        assert kinds == ["AIV", "Orchestration"], kinds

    def test_the_decorator_requires_a_plain_function(self):
        with pytest.raises(TypeError, match="expects a plain function"):
            pl.jit_trace.body("not a function")  # pyright: ignore[reportArgumentType]

    def test_the_body_keeps_the_kernels_metadata(self):
        """A trace, a diagnostic and a stack trace name the fragment, not a wrapper."""

        @pl.jit_trace.body
        def scaled(x, k):
            """Scale one tile."""
            return pl.mul(x, k)

        assert scaled.__name__ == "scaled"
        assert scaled.__doc__ == "Scale one tile."
        assert scaled.__wrapped__ is scaled._kernel
        assert repr(scaled) == "<traceable body scaled>"


class TestFrameworkParity:
    """The litmus test: the same spelling from the public authoring surface.

    ``@pl.jit_trace.body`` is a policy — ``@pl.authoring.recipe`` over
    ``ctx.splice_body`` — so a framework outside this repository can build it
    without private access. When this test fails, the stdlib has grown a
    privilege an external framework does not have.
    """

    @staticmethod
    def _framework_body(kernel):
        """A ``@body`` decorator written with the public ``pl.authoring`` surface only."""
        prepared = pl.authoring.body(kernel)

        @pl.authoring.recipe
        def expand(ctx, *args, **kwargs):
            return ctx.splice_body(prepared, *args, **kwargs)

        return expand

    def test_the_framework_spelling_lowers_to_the_same_program(self):
        def scale_block(a, out, row0, k):
            tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(tile, k), [row0, 0], out)

        framework_scale_block = self._framework_body(scale_block)
        stdlib_scale_block = pl.jit_trace.body(scale_block)

        def kernel(step):
            @pl.jit_trace
            def entry(
                a: pl.Tensor[[ROWS, COLS], pl.FP32],
                k: pl.Scalar[pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ):
                with pl.at(level=pl.Level.CORE_GROUP):
                    for chunk in pl.range(ROWS // TILE_ROWS):
                        row0 = chunk * TILE_ROWS
                        step(a, out, row0, k)
                return out

            return entry

        via_framework = _program_text(kernel(framework_scale_block), _tensor(), 2.0, _out())
        via_stdlib = _program_text(kernel(stdlib_scale_block), _tensor(), 2.0, _out())
        assert via_framework == via_stdlib

    def test_the_framework_spelling_expands_a_returned_value(self):
        def scaled(x, k):
            return pl.mul(x, k)

        framework_scaled = self._framework_body(scaled)

        @pl.jit_trace
        def entry(
            a: pl.Tensor[[ROWS, COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [ROWS, COLS])
                pl.store(framework_scaled(tile, 2.0), [0, 0], out)
            return out

        printed = _program_text(entry, _tensor(), _out())
        assert "mul" in printed
        assert "scaled" not in printed
