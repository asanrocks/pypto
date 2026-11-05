# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Acceptance suite for the tracing-jit first wave.

Drives a small but representative corpus through the tracing frontend: five
real example entries plus two focused fixtures. CPU-side only: this proves
collection through tracing, program assembly with the expected callees, and the
presence of the required IR constructs. Numerical and dispatch checks belong to
the wave's device acceptance.

Each assertion is on the IR shape, not only on the golden or the function count.
A wrong result is not the only failure mode here: an unstaged loop can
host-unroll into identical arithmetic, and a missing callee can leave a smaller
program that still compiles. Asserting the construct is what distinguishes a
working frontend from a coincidentally correct output.
"""

import pypto.language as pl
import pytest
import torch

# The five example entries this suite traces. These are the real user-facing
# definitions, imported rather than copied, so helper closure comes with them.
from examples.beginner.elementwise import chunked_add  # type: ignore[reportMissingImports]
from examples.intermediate.fused_linear import (  # type: ignore[reportMissingImports]
    fused_linear_relu,
    fused_matmul_bias,
)
from examples.intermediate.softmax import tile_softmax  # type: ignore[reportMissingImports]
from pypto.pypto_core import ir, passes

from examples.utils.cross_function_calls import main_kernel  # type: ignore[reportMissingImports]
from tests.ut.language.tracing.jit._traced_corpus_fixtures import annotated_add, controlled_accumulate

F32 = torch.float32


# ---------------------------------------------------------------------------
# IR inspection helpers
# ---------------------------------------------------------------------------


class _Collect(ir.IRVisitor):
    """Collect the statements and calls of one subtree in traversal order."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[ir.Call] = []
        self.assigns: list[ir.AssignStmt] = []
        self.fors: list[ir.ForStmt] = []
        self.ifs: list[ir.IfStmt] = []

    def visit_call(self, op: ir.Call) -> None:
        self.calls.append(op)
        super().visit_call(op)

    def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
        self.assigns.append(op)
        super().visit_assign_stmt(op)

    def visit_for_stmt(self, op: ir.ForStmt) -> None:
        self.fors.append(op)
        super().visit_for_stmt(op)

    def visit_if_stmt(self, op: ir.IfStmt) -> None:
        self.ifs.append(op)
        super().visit_if_stmt(op)

    def op_names(self) -> list[str]:
        return [_op_kind(call.op.name) for call in self.calls]


class _VarNames(ir.IRVisitor):
    """Collect every Var name an expression mentions, at any depth."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def visit_var(self, op: ir.Var) -> None:
        self.names.append(op.name_hint)
        super().visit_var(op)


def _mentions(expr: ir.IRNode, var_name: str) -> bool:
    collector = _VarNames()
    if isinstance(expr, ir.Stmt):
        collector.visit_stmt(expr)
    else:
        collector.visit_expr(expr)  # type: ignore[arg-type]
    # SSA naming appends a suffix; compare exact first, then the base name.
    return any(name == var_name or name.split("__")[0] == var_name for name in collector.names)


def _op_kind(op_name: str) -> str:
    """Strip the storage-dialect prefix so assertions survive either lowering.

    Which of ``tile.*`` / ``buffer.*`` / ``tensor.*`` a semantic op lowers to
    depends on the selected pipeline (Buffer IR is on in the unit-test context),
    not on what the kernel wrote. The construct being asserted -- load, row_max,
    muls -- is the same either way.
    """
    for prefix in ("tile.", "buffer.", "tensor."):
        if op_name.startswith(prefix):
            return op_name[len(prefix) :]
    return op_name


def _const_int(expr: ir.Expr) -> int | None:
    return expr.value if isinstance(expr, ir.ConstInt) else None


def _const_float(expr: ir.Expr) -> float | None:
    return expr.value if isinstance(expr, ir.ConstFloat) else None


def _function(program: ir.Program, name: str) -> ir.Function:
    fn = program.get_function(name)
    have = sorted(f.name for f in program.functions.values())
    assert fn is not None, f"missing function {name!r}; have {have}"
    return fn


def _collect(stmt: ir.Stmt) -> _Collect:
    collector = _Collect()
    collector.visit_stmt(stmt)
    return collector


def _assign_producer(collector: _Collect, var_name: str) -> ir.Expr | None:
    """The value assigned to ``var_name`` in this subtree, if any."""
    for assign in collector.assigns:
        if assign.var.name_hint == var_name:
            return assign.value
    return None


def _calls_named(collector: _Collect, kind: str) -> list[ir.Call]:
    """Calls whose op kind is ``kind`` (``load``, ``row_max``, or a callee name)."""
    return [call for call in collector.calls if _op_kind(call.op.name) == kind]


# ---------------------------------------------------------------------------
# E1 — chunked vector addition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "args", "expected_functions"),
    [
        (
            chunked_add,
            lambda: [
                torch.randn(512, 128, dtype=F32),
                torch.randn(512, 128, dtype=F32),
                torch.empty(512, 128, dtype=F32),
            ],
            {"chunked_add", "chunked_add_incore_0"},
        ),
        (
            tile_softmax,
            lambda: [torch.randn(64, 64, dtype=F32), torch.empty(64, 64, dtype=F32)],
            {"tile_softmax", "tile_softmax_incore_0"},
        ),
        (
            fused_linear_relu,
            lambda: [torch.randn(64, 64, dtype=F32)] * 3 + [torch.empty(64, 64, dtype=F32)],
            {
                "fused_linear_relu",
                "_matmul_kernel_64x64",
                "_add_bias_relu_kernel_64x64",
            },
        ),
        (
            fused_matmul_bias,
            lambda: [torch.randn(64, 64, dtype=F32)] * 3 + [torch.empty(64, 64, dtype=F32)],
            {
                "fused_matmul_bias",
                "_matmul_kernel_64x64",
                "_add_bias_kernel_64x64",
            },
        ),
        (
            main_kernel,
            lambda: [torch.randn(128, 128, dtype=F32), torch.empty(128, 128, dtype=F32)],
            {"main_kernel", "main_kernel_incore_0", "main_kernel_incore_1"},
        ),
    ],
    ids=["chunked_add", "tile_softmax", "fused_linear_relu", "fused_matmul_bias", "main_kernel"],
)
def test_entry_assembles_its_expected_program(kernel, args, expected_functions):
    """Each entry compiles to exactly the callees its body reaches.

    An exact name set (rather than a count) catches both a dropped callee and a
    spurious fused one, which a ``len(functions) >= n`` check cannot.
    """
    program = kernel.compile_for_test(*args())
    assert {fn.name for fn in program.functions.values()} == expected_functions


def test_chunked_add_stages_the_loop():
    """E1: the tiled loop must be a staged loop, not host-unrolled into 4 copies."""
    program = chunked_add.compile_for_test(
        torch.randn(512, 128, dtype=F32), torch.randn(512, 128, dtype=F32), torch.empty(512, 128, dtype=F32)
    )
    incore = _collect(_function(program, "chunked_add_incore_0").body)

    assert len(incore.fors) == 1, "the tiled body must contain exactly one staged loop"
    loop = incore.fors[0]
    assert _const_int(loop.stop) == 4, "the loop must carry the tiled extent, not a host-unrolled body"
    assert loop.iter_args == [], "this loop carries no values"

    # The loop index must reach the load/store offsets; otherwise the loop is
    # staged but every iteration addresses the same tile.
    inner = _collect(loop.body)
    loads = _calls_named(inner, "load")
    stores = _calls_named(inner, "store")
    assert len(loads) == 2 and len(stores) == 1
    for call in loads:
        assert _mentions(call.args[1], loop.loop_var.name_hint), "load offset must depend on the loop index"
    for call in stores:
        assert _mentions(call.args[1], loop.loop_var.name_hint), "store offset must depend on the loop index"


# ---------------------------------------------------------------------------
# E2 — row softmax
# ---------------------------------------------------------------------------


def test_softmax_has_reductions_and_broadcast_division():
    """E2: row max/sum and the broadcast division they feed survive to lowered IR."""
    program = tile_softmax.compile_for_test(torch.randn(64, 64, dtype=F32), torch.empty(64, 64, dtype=F32))
    incore = _collect(_function(program, "tile_softmax_incore_0").body)
    names = incore.op_names()

    # Every stage of the algorithm must be present, not merely one reduction.
    for required in ("row_max", "row_expand_sub", "exp", "row_sum"):
        assert required in names, f"missing {required}; got {names}"
    division = _calls_named(incore, "row_expand_div")
    assert len(division) == 1, "softmax must close with one broadcast division"

    # The divisor must be the value row_sum produced, not a coincidentally
    # same-shaped tile.
    divisor_name = division[0].args[1].name_hint  # type: ignore[attr-defined]
    producer = _assign_producer(incore, divisor_name)
    assert isinstance(producer, ir.Call) and _op_kind(producer.op.name) == "row_sum"

    # The exponential must consume the max-shifted input.
    exp_call = _calls_named(incore, "exp")[0]
    shifted_name = exp_call.args[0].name_hint  # type: ignore[attr-defined]
    shifted_producer = _assign_producer(incore, shifted_name)
    assert isinstance(shifted_producer, ir.Call) and _op_kind(shifted_producer.op.name) == "row_expand_sub"


# ---------------------------------------------------------------------------
# E3/E4 — incore composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "entry", "first_callee", "second_callee"),
    [
        (fused_linear_relu, "fused_linear_relu", "_matmul_kernel_64x64", "_add_bias_relu_kernel_64x64"),
        (fused_matmul_bias, "fused_matmul_bias", "_matmul_kernel_64x64", "_add_bias_kernel_64x64"),
    ],
    ids=["fused_linear_relu", "fused_matmul_bias"],
)
def test_incore_composition_feeds_one_callee_into_the_next(kernel, entry, first_callee, second_callee):
    """E3/E4: the entry reaches both incore callees and links them by value."""
    program = kernel.compile_for_test(
        torch.randn(64, 64, dtype=F32),
        torch.randn(64, 64, dtype=F32),
        torch.randn(64, 64, dtype=F32),
        torch.empty(64, 64, dtype=F32),
    )
    orchestrator = _collect(_function(program, entry).body)

    first = _calls_named(orchestrator, first_callee)
    second = _calls_named(orchestrator, second_callee)
    assert len(first) == 1 and len(second) == 1, "each callee must be called exactly once"

    # The intermediate tensor produced by the matmul must be the value the
    # bias/activation kernel consumes; otherwise the two calls are independent.
    intermediate_name = second[0].args[0].name_hint  # type: ignore[attr-defined]
    producer = _assign_producer(orchestrator, intermediate_name)
    assert isinstance(producer, ir.Call) and producer.op.name == first_callee

    # The intermediate is created inside the entry, not appended as an output.
    assert _calls_named(orchestrator, "create"), "the intermediate must be allocated by the entry"
    assert len(second[0].args) == 3, "the fused tail still writes the caller's output"


def test_shared_matmul_helper_is_defined_once_across_both_callers():
    """E4: the shared helper is declared once and both callers reuse it."""
    for kernel in (fused_linear_relu, fused_matmul_bias):
        program = kernel.compile_for_test(
            torch.randn(64, 64, dtype=F32),
            torch.randn(64, 64, dtype=F32),
            torch.randn(64, 64, dtype=F32),
            torch.empty(64, 64, dtype=F32),
        )
        matmuls = [fn for fn in program.functions.values() if fn.name == "_matmul_kernel_64x64"]
        assert len(matmuls) == 1


# ---------------------------------------------------------------------------
# E5 — inline composition
# ---------------------------------------------------------------------------


def test_inline_helpers_are_spliced_with_their_arithmetic_intact():
    """E5: both helpers are inlined, and their operations and data dependency survive."""
    program = main_kernel.compile_for_test(torch.randn(128, 128, dtype=F32), torch.empty(128, 128, dtype=F32))

    assert not [fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline], (
        "inline helpers must not survive as functions"
    )

    first = _collect(_function(program, "main_kernel_incore_0").body)
    second = _collect(_function(program, "main_kernel_incore_1").body)

    adds = _calls_named(first, "adds")
    muls = _calls_named(second, "muls")
    assert len(adds) == 1 and len(muls) == 1
    assert _const_float(adds[0].args[1]) == 1.0, "add_helper's constant must survive the splice"
    assert _const_float(muls[0].args[1]) == 2.0, "mul_helper's constant must survive the splice"

    # The helpers' write-through must be preserved: the first callee writes the
    # intermediate output it was given, and the second consumes that tensor.
    assert _calls_named(first, "store"), "add_helper writes its intermediate result"
    assert _function(program, "main_kernel_incore_1").params[0].name_hint.startswith("intermediate")

    entry = _collect(_function(program, "main_kernel").body)
    entry_calls = [
        call for call in entry.calls if call.op.name in {"main_kernel_incore_0", "main_kernel_incore_1"}
    ]
    assert [call.op.name for call in entry_calls] == ["main_kernel_incore_0", "main_kernel_incore_1"]


# ---------------------------------------------------------------------------
# F1 — runtime control flow and host specialization
# ---------------------------------------------------------------------------


def _controlled_program(n: int = 3, scale: float = 2.0, *, buffer_ir: bool = True):
    """Compile F1 under an explicit Buffer IR choice.

    Buffer IR rewrites a loop-carried value into in-place buffer mutation, which
    erases the SSA ``iter_args``/``return_vars`` that show the carry is real. The
    carry assertion therefore pins ``enable_buffer_ir=False``; the dispatch and
    host-knob assertions run on the shipped default.
    """
    with passes.PassContext([], enable_buffer_ir=buffer_ir):
        return controlled_accumulate.compile_for_test(
            torch.randn(64, 64, dtype=F32), n, scale, torch.empty(64, 64, dtype=F32)
        )


def test_controlled_accumulate_stages_a_runtime_loop_carry_and_branch():
    """F1: the loop bound and branch stay runtime, and the accumulator is carried."""
    program = _controlled_program(buffer_ir=False)
    incore = _collect(_function(program, "controlled_accumulate_incore_0").body)

    assert len(incore.fors) == 1 and len(incore.ifs) == 1
    loop, branch = incore.fors[0], incore.ifs[0]

    # A host-unrolled loop would leave a constant extent and no iter arg.
    assert _const_int(loop.stop) is None, "the loop extent must stay the runtime scalar"
    assert _mentions(loop.stop, "n"), "the loop bound must be the scalar parameter"
    assert loop.kind == ir.ForKind.Sequential
    assert len(loop.iter_args) == 1 and len(loop.return_vars) == 1, "the accumulator must be carried"
    assert isinstance(loop.iter_args[0].initValue, ir.Var), "the carried value seeds from the loaded tile"

    assert not isinstance(branch.condition, (ir.ConstBool, ir.ConstInt)), (
        "the branch must stay a runtime conditional"
    )
    assert _mentions(branch.condition, "n"), "the branch must test the runtime scalar"


def test_controlled_accumulate_bakes_the_host_knob():
    """F1: ``scale`` is a host value at trace time, not a runtime parameter."""
    for scale in (2.0, 3.0):
        program = _controlled_program(scale=scale)
        incore_fn = _function(program, "controlled_accumulate_incore_0")

        base_names = [param.name_hint.split("__")[0] for param in incore_fn.params]
        assert base_names == ["x", "n", "out"], "the host knob must not become a parameter"

        muls = _calls_named(_collect(incore_fn.body), "muls")
        assert len(muls) == 1
        assert _const_float(muls[0].args[1]) == scale, "the host value must be baked into the trace"


def test_controlled_accumulate_n_is_a_runtime_index_scalar():
    """F1: ``n`` stays an INDEX parameter, so changing it needs no new artifact shape."""
    for n in (0, 1, 3):
        incore_fn = _function(_controlled_program(n=n), "controlled_accumulate_incore_0")
        scalar = incore_fn.params[1]
        assert isinstance(scalar.type, ir.ScalarType)
        assert scalar.type.dtype == pl.INDEX


# ---------------------------------------------------------------------------
# F2 — fully annotated entry
# ---------------------------------------------------------------------------


def test_annotated_add_keeps_its_declared_signature():
    """F2: the declaration, not the argument, fixes shapes, dtype and direction."""
    program = annotated_add.compile_for_test(
        torch.randn(64, 64, dtype=F32), 1.0, torch.empty(64, 64, dtype=F32)
    )
    for fn_name in ("annotated_add", "annotated_add_incore_0"):
        fn = _function(program, fn_name)
        assert len(fn.params) == 3
        tensor, scalar, out = fn.params

        assert isinstance(tensor.type, ir.TensorType)
        assert [_const_int(dim) for dim in tensor.type.shape] == [64, 64]
        assert tensor.type.dtype == pl.FP32

        assert isinstance(scalar.type, ir.ScalarType)
        assert scalar.type.dtype == pl.FP32

        assert fn.param_directions[0] == ir.ParamDirection.In
        assert fn.param_directions[1] == ir.ParamDirection.In
        assert fn.param_directions[2] == ir.ParamDirection.Out

    adds = _calls_named(_collect(_function(program, "annotated_add_incore_0").body), "adds")
    assert len(adds) == 1, "the annotated body must add the runtime scalar to the tile"


# ---------------------------------------------------------------------------
# Tracing jit family surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["opaque", "extern"])
def test_unverified_family_spellings_raise_at_decoration(form):
    """``host`` / ``graph`` are verified now; these two reject at decoration.

    The remaining ``@pl.jit`` family spellings raise at the attribute
    access, so neither the bare (``@pl.jit_trace.opaque``) nor the
    parenthesized (``@pl.jit_trace.opaque(...)``) decoration can start
    tracing a lane this wave does not verify — an accepted-but-unverified
    spelling would advertise a surface without acceptance. The rejection
    is per-lane: the parsing jit keeps the whole family.
    """
    with pytest.raises(TypeError, match=rf"@pl\.jit_trace\.{form} is not part of the verified"):
        getattr(pl.jit_trace, form)

    # The boundary is per-lane: the parser jit still carries the spelling.
    assert callable(getattr(pl.jit, form))


@pytest.mark.parametrize(
    ("form", "func_type", "level", "role"),
    [
        ("incore", "InCore", None, None),
        ("inline", "Inline", None, None),
        ("host", "Opaque", "HOST", "Orchestrator"),
        ("graph", "Graph", None, None),
    ],
)
def test_family_aliases_keep_the_parsing_lane_kind(form, func_type, level, role):
    """A family alias is the parser lane's declaration, traced.

    The function kind is what every pass keys on (an InCore body is outlined,
    a Graph body is replayed, a HOST entry owns the per-rank dispatch loop), so
    an alias that changed it would emit a different program than the spelling
    it claims to mirror.
    """

    @getattr(pl.jit_trace, form)
    def kernel(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
        pl.store(pl.add(pl.load(x, [0, 0], [8, 8]), 1.0), [0, 0], out)
        return out

    assert kernel._function_type == getattr(ir.FunctionType, func_type)
    assert kernel._level == (getattr(ir.Level, level) if level else None)
    assert kernel._role == (getattr(ir.Role, role) if role else None)


@pytest.mark.parametrize(
    ("form", "kwarg", "accepted"),
    [
        ("incore", "level", True),
        ("incore", "auto_scope", False),
        ("inline", "level", False),
        ("inline", "auto_scope", True),
        ("host", "level", False),
        ("host", "auto_scope", True),
        ("graph", "level", False),
        ("graph", "auto_scope", False),
    ],
)
def test_family_aliases_keep_the_parsing_lane_kwarg_surface(form, kwarg, accepted):
    """A kwarg that cannot take effect is refused, as the parser lane refuses it.

    ``level`` is meaningful only where the declaration picks a tier (``incore``),
    and ``auto_scope`` only where the body's scopes land in the caller — the
    entry, the HOST orchestrator, and an inline helper.
    """
    value = ir.Level.CORE_GROUP if kwarg == "level" else False
    alias = getattr(pl.jit_trace, form)
    if accepted:
        assert callable(alias(**{kwarg: value}))
        return
    with pytest.raises(TypeError, match=kwarg):
        alias(**{kwarg: value})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
