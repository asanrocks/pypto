# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``@pl.jit.graph`` — the Graph alias on the tracing jit (gap K6, task F5).

Two stories, one per lane the alias must serve:

- **standalone fragment** — fully annotation-shaped, the eager lane traces at
  decoration and the traced function equals the explicit
  ``@pl.function(type=pl.FunctionType.Graph)`` spelling;
- **dispatching fragment** — bare annotations, the lazy callee lane: called
  inside a jit entry's trace, the Graph variant must join the entry's program
  assembly *together with* the jit callee it dispatches. This is the orphan
  trap the gap's tracking note names: a bare ``@pl.function`` trace runs with
  no program assembly, so a Graph fragment stated that way would compile an
  entry whose dispatched callee never joined the program.
"""

import pypto.language as pl
import pytest
from pypto import backend
from pypto.backend import BackendType
from pypto.ir.printer import python_print
from pypto.pypto_core import ir, passes

# ---------------------------------------------------------------------------
# The standalone fragment: the eager lane
# ---------------------------------------------------------------------------


def test_standalone_fragment_traces_as_a_graph_function():
    """Fully annotation-shaped, the alias traces at decoration (the eager
    gate) and the result is the Graph function itself — the same object the
    explicit ``@pl.function(type=Graph)`` spelling produces."""

    @pl.jit.graph
    def band(
        w: pl.Tensor[[64, 64], pl.FP32],
        acc: pl.InOut[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            t = pl.load(w, [0, 0], [32, 32])
            c = pl.load(acc, [0, 0], [32, 32])
            pl.store(pl.add(t, c), [0, 0], acc)
        return acc

    assert isinstance(band, ir.Function)
    assert band.func_type == ir.FunctionType.Graph
    # A Graph body orchestrates tasks: chip level, orchestrator role.
    assert band.level == ir.Level.CHIP
    assert band.role == ir.Role.Orchestrator


def test_standalone_fragment_matches_the_explicit_spelling():
    """The alias's eager trace is structurally the explicit spelling's trace —
    the alias adds no IR of its own."""

    @pl.jit.graph
    def band(
        w: pl.Tensor[[64, 64], pl.FP32],
        acc: pl.InOut[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            t = pl.load(w, [0, 0], [32, 32])
            c = pl.load(acc, [0, 0], [32, 32])
            pl.store(pl.add(t, c), [0, 0], acc)
        return acc

    def make_reference():
        @pl.function(type=pl.FunctionType.Graph)
        def band(  # noqa: F811 - the reference, under the compared name
            w: pl.Tensor[[64, 64], pl.FP32],
            acc: pl.InOut[pl.Tensor[[32, 32], pl.FP32]],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.load(w, [0, 0], [32, 32])
                c = pl.load(acc, [0, 0], [32, 32])
                pl.store(pl.add(t, c), [0, 0], acc)
            return acc

        return band

    # Each frontend builds its own Var objects; the anchor compares shape +
    # names, never object identity (the conformance-harness regime).
    ir.assert_structural_equal(band, make_reference(), enable_auto_mapping=True)


def test_graph_alias_rejects_level():
    """Graph carries no level of its own (chip-derived), so ``level=`` is a
    loud TypeError — the same gate the other level-less aliases hold."""
    with pytest.raises(TypeError, match="does not accept level="):
        pl.jit.graph(level=ir.Level.CORE_GROUP)


# ---------------------------------------------------------------------------
# The dispatching fragment: the lazy callee lane (the orphan trap)
# ---------------------------------------------------------------------------


@pl.jit.incore
def scale_band(a: pl.Tensor, out: pl.InOut[pl.Tensor]) -> pl.Tensor:
    t = pl.load(a, [0, 0], [16, 16])
    out = pl.store(pl.add(t, t), [0, 0], out)
    return out


@pl.jit.graph
def record_band(a: pl.Tensor, out: pl.InOut[pl.Tensor]) -> pl.Tensor:
    """The Graph fragment that dispatches a jit kernel — the shape the
    standalone ``@pl.function`` spelling cannot serve (no program assembly,
    the callee would be orphaned)."""
    out = scale_band(a, out)
    return out


@pl.jit
def launch_band(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
    out = record_band(a, out)
    return out


def test_dispatching_fragment_joins_the_entry_program():
    """The entry's program carries all three functions: the Orchestration
    entry, the Graph fragment traced as a callee variant, and the InCore
    kernel the fragment dispatches. A dropped Graph or a missing callee is
    the orphan trap — the program would compile without the recorded
    topology or without its only task."""
    torch = pytest.importorskip("torch")

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    try:
        with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
            program = launch_band.compile_for_test(torch.zeros(16, 16), torch.zeros(16, 16))
    finally:
        backend.reset_for_testing()

    kinds = {fn.name: fn.func_type for fn in program.functions.values()}
    # The Default pipeline reclassifies the InCore callee to its AIV target —
    # the fragment's dispatched kernel is present, which is the assertion.
    assert kinds == {
        "launch_band": ir.FunctionType.Orchestration,
        "record_band": ir.FunctionType.Graph,
        "scale_band": ir.FunctionType.AIV,
    }, kinds

    # The printed form is what the st compile-side guard keys on: exactly one
    # Graph, reachable as a Call from the entry.
    src = python_print(program)
    assert src.count("type=pl.FunctionType.Graph") == 1, src
