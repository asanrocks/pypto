# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The direct-call arm of the return-lineage walk (``return_lineage_utils.cpp``).

A writeback or user call used *directly* as the returned expression binds no SSA
var, so ``TraceVar`` never sees it; ``TraceIndexedExprsToParamIndices`` resolves
it through the value its result aliases. These tests pin the three properties
that arm must share with the var-bound arm:

* a callee position reached through a ``Submit`` must be translated with the
  ``Submit::args_`` contract (prefix / omitted-Out gap / CommCtx suffix), never
  by raw ``args_`` index — a bounded-coverage Submit's ``args_[param_idx]`` can
  name a *different* parameter (here: the trailing CommCtx arg standing in for
  the runtime-allocated Out);
* a metadata-changing writeback returned directly records *no* lineage, exactly
  like the var-bound spelling — not a wrong one ``NormalizeReturnOrder`` would
  trust;
* several direct calls forwarding through one shared alias chain all resolve
  (the chain is traced through the batch's shared memo);
* a call used directly as *another call's argument* resolves like its
  SSA-bound spelling (``return outer(inner(x))`` vs ``tmp = inner(x)``);
* a forwarded tuple whose callee declares ONE structural ``TupleType`` slot
  keeps its per-element delivered lineage (the callee's one-entry position
  map cannot name the elements).

The IR is built directly (like ``test_legalize_graph_boundary``'s submit-gap
tests) because the shapes under test — a materialized CommCtx suffix, a
type-narrowed writeback return — sit past what the tracing entry guard would
accept before the map could be inspected.
"""

from pypto import ir


def _tensor_type() -> ir.TensorType:
    return ir.TensorType([4], ir.DataType.FP32)


def _narrowed_type(span: ir.Span) -> ir.TensorType:
    """Same buffer as ``_tensor_type()``, but with valid-shape metadata set.

    This is the type relation ``tensor.set_validshape`` produces: the result
    aliases the input's storage while carrying different view metadata, so the
    two are NOT structurally equal.
    """
    return ir.TensorType(
        [4, 1],
        ir.DataType.FP32,
        tensor_view=ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[4, 1]),
    )


# ---------------------------------------------------------------------------
# Finding 1 — Submit arg translation on the direct path.
# ---------------------------------------------------------------------------


def _submit_gap_program(return_output: bool):
    """A Graph that returns a Submit launch directly, omitting the callee's Out.

    The callee has params ``[source, output, ctx]``; the Submit passes only
    ``[source, ctx]`` (the Out is runtime-allocated) — the canonical
    post-``MaterializeDistTensorCtx`` shape. When the callee returns its
    ``output`` param, the direct path must find *no* caller-side provenance: by
    the ``Submit::args_`` contract, callee param 1 sits in the runtime gap, and
    ``args_[1]`` actually binds the trailing CommCtx param.
    """
    span = ir.Span.unknown()
    tensor_type = _tensor_type()
    ctx_type = ir.CommCtxType.get()
    source = ir.Var("source", tensor_type, span)
    output = ir.Var("output", tensor_type, span)
    ctx = ir.Var("ctx", ctx_type, span)
    stage = ir.Function(
        "stage",
        [(source, ir.ParamDirection.In), (output, ir.ParamDirection.Out), ctx],
        [tensor_type],
        ir.ReturnStmt([output if return_output else source], span),
        span,
        ir.FunctionType.InCore,
    )
    graph_source = ir.Var("source", tensor_type, span)
    graph_ctx = ir.Var("ctx", ctx_type, span)
    pair_type = ir.TupleType([tensor_type, ir.ScalarType(ir.DataType.TASK_ID)])
    launch = ir.Submit(ir.GlobalVar("stage"), [graph_source, graph_ctx], [], pair_type, span)
    graph = ir.Function(
        "layer",
        [graph_source, graph_ctx],
        [pair_type],
        ir.ReturnStmt([launch], span),
        span,
        ir.FunctionType.Graph,
    )
    return stage, ir.Program([stage, graph], "submit_direct_return", span), graph


def test_a_direct_submit_return_never_binds_the_ctx_arg():
    """The omitted Out has no caller-side provenance — not the CommCtx param.

    Indexing ``args_`` with the callee param index reads ``args_[1]`` = the ctx
    arg, which IS a param of the launching function, so the mis-binding reports
    a real (wrong) index rather than failing safe.
    """
    _, program, graph = _submit_gap_program(return_output=True)
    assert ir.returned_param_indices(graph, program) == [None]
    assert ir.delivered_param_indices(graph, program) == [None]


def test_a_direct_submit_return_keeps_caller_supplied_provenance():
    """Control: the callee's caller-supplied param still maps through the prefix."""
    _, program, graph = _submit_gap_program(return_output=False)
    assert ir.returned_param_indices(graph, program) == [0]
    assert ir.delivered_param_indices(graph, program) == [0]


# ---------------------------------------------------------------------------
# Finding 2 — returned-type compatibility on the direct path.
# ---------------------------------------------------------------------------


def _writeback_program(bound: bool, matching_type: bool):
    """An entry whose return is a ``tensor.set_validshape`` over its Out param.

    ``matching_type`` picks the param's declared type: equal to the writeback
    result (lineage recorded) or without the view metadata (mismatch — no
    lineage). ``bound`` spells the writeback through an SSA var instead of
    returning the call directly; both spellings must classify identically.
    """
    span = ir.Span.unknown()
    result_type = _narrowed_type(span)
    param_type = result_type if matching_type else _tensor_type()
    out = ir.Var("out", param_type, span)
    rows = ir.ConstInt(4, ir.DataType.INDEX, span)
    cols = ir.ConstInt(1, ir.DataType.INDEX, span)
    call = ir.Call(ir.Op("tensor.set_validshape"), [out, rows, cols], result_type, span)
    statements: list[ir.Stmt] = []
    returned = call
    if bound:
        result = ir.Var("result", result_type, span)
        statements.append(ir.AssignStmt(result, call, span))
        returned = result
    statements.append(ir.ReturnStmt([returned], span))
    entry = ir.Function(
        "entry",
        [(out, ir.ParamDirection.Out)],
        [result_type],
        ir.SeqStmts(statements, span),
        span,
        ir.FunctionType.InCore,
    )
    return ir.Program([entry], "writeback_return", span), entry


def test_a_metadata_changing_writeback_return_records_no_lineage():
    """Direct and bound spellings agree: a type mismatch means no lineage.

    The writeback result aliases the Out param's buffer, but its view metadata
    differs from the param's declared type, so the map must not name the param —
    ``NormalizeReturnOrder`` trusts a recorded index as a complete writeback.
    """
    for bound in (False, True):
        program, entry = _writeback_program(bound=bound, matching_type=False)
        assert ir.returned_param_indices(entry, program) == [None], f"bound={bound}"
        assert ir.delivered_param_indices(entry, program) == [None], f"bound={bound}"


def test_a_type_matching_writeback_return_keeps_its_lineage():
    """Control: same op, result type equal to the param's — lineage recorded."""
    for bound in (False, True):
        program, entry = _writeback_program(bound=bound, matching_type=True)
        assert ir.returned_param_indices(entry, program) == [0], f"bound={bound}"
        assert ir.delivered_param_indices(entry, program) == [0], f"bound={bound}"


# ---------------------------------------------------------------------------
# Finding 3 — direct calls over a shared alias chain.
# ---------------------------------------------------------------------------


def test_many_direct_calls_through_a_shared_alias_chain():
    """Three direct calls over one alias chain all resolve, via the shared memo.

    The chain ``a2 -> a1 -> src`` is forwarded by every element of the returned
    tuple; each direct call resolves it through the batch's memo, so the chain
    is walked once and every position still lands on the parameter.
    """
    span = ir.Span.unknown()
    tensor_type = _tensor_type()
    value = ir.Var("value", tensor_type, span)
    relay = ir.Function(
        "relay",
        [(value, ir.ParamDirection.In)],
        [tensor_type],
        ir.ReturnStmt([value], span),
        span,
        ir.FunctionType.InCore,
    )
    src = ir.Var("src", tensor_type, span)
    a1 = ir.Var("a1", tensor_type, span)
    a2 = ir.Var("a2", tensor_type, span)
    call = ir.Call(ir.GlobalVar("relay"), [a2], tensor_type, span)
    entry = ir.Function(
        "entry",
        [src],
        [tensor_type, tensor_type, tensor_type],
        ir.SeqStmts(
            [
                ir.AssignStmt(a1, src, span),
                ir.AssignStmt(a2, a1, span),
                ir.ReturnStmt([ir.MakeTuple([call, call, call], span)], span),
            ],
            span,
        ),
        span,
        ir.FunctionType.InCore,
    )
    program = ir.Program([relay, entry], "shared_alias_chain", span)
    assert ir.returned_param_indices(entry, program) == [0, 0, 0]


# ---------------------------------------------------------------------------
# Finding 4 — a nested call used directly as another call's argument.
# ---------------------------------------------------------------------------


def _nested_call_program(bound: bool):
    """An entry that forwards a call *through* another call in one expression.

    ``return relay(relay(src))`` nests the inner call as a bare argument, so it
    binds no SSA var; the SSA-equivalent ``tmp = relay(src)`` spelling is the
    control. Both must classify identically — binding a name cannot change a
    result's classification.
    """
    span = ir.Span.unknown()
    tensor_type = _tensor_type()
    value = ir.Var("value", tensor_type, span)
    relay = ir.Function(
        "relay",
        [(value, ir.ParamDirection.In)],
        [tensor_type],
        ir.ReturnStmt([value], span),
        span,
        ir.FunctionType.InCore,
    )
    src = ir.Var("src", tensor_type, span)
    inner = ir.Call(ir.GlobalVar("relay"), [src], tensor_type, span)
    forwarded = ir.Call(ir.GlobalVar("relay"), [inner], tensor_type, span)
    statements: list[ir.Stmt] = []
    if bound:
        tmp = ir.Var("tmp", tensor_type, span)
        statements.append(ir.AssignStmt(tmp, inner, span))
        forwarded = ir.Call(ir.GlobalVar("relay"), [tmp], tensor_type, span)
    statements.append(ir.ReturnStmt([forwarded], span))
    entry = ir.Function(
        "entry",
        [src],
        [tensor_type],
        ir.SeqStmts(statements, span),
        span,
        ir.FunctionType.InCore,
    )
    return ir.Program([relay, entry], "nested_call", span), entry


def test_a_nested_direct_call_keeps_its_lineage():
    """Direct and SSA-bound spellings agree: both land on the parameter.

    Before the fix, the direct nesting stopped at the inner ``Call`` — only
    ``TraceVar`` followed user calls — so ``return outer(inner(x))`` reported
    no lineage while its SSA-equivalent spelling of the same value did not.
    """
    for bound in (False, True):
        program, entry = _nested_call_program(bound=bound)
        assert ir.returned_param_indices(entry, program) == [0], f"bound={bound}"
        assert ir.delivered_param_indices(entry, program) == [0], f"bound={bound}"


# ---------------------------------------------------------------------------
# Finding 5 — forwarding a callee that declares ONE structural TupleType slot.
# ---------------------------------------------------------------------------


def _structural_tuple_program(bound: bool):
    """A callee that returns ``pl.Tuple(out0, out1)`` under ONE TupleType slot.

    The caller forwards that call's whole tuple as its own single declared
    tuple return. The callee's *position* map is one entry (``[None]``) — the
    elements live in the delivered values — so the caller's delivered map must
    ask the value-faithful question or both collapse to ``[None]``.
    """
    span = ir.Span.unknown()
    tensor_type = _tensor_type()
    pair_type = ir.TupleType([tensor_type, tensor_type])
    a = ir.Var("a", tensor_type, span)
    o0 = ir.Var("o0", tensor_type, span)
    o1 = ir.Var("o1", tensor_type, span)
    inner = ir.Function(
        "inner",
        [(a, ir.ParamDirection.In), (o0, ir.ParamDirection.Out), (o1, ir.ParamDirection.Out)],
        [pair_type],
        ir.ReturnStmt([ir.MakeTuple([o0, o1], span)], span),
        span,
        ir.FunctionType.InCore,
    )
    src = ir.Var("src", tensor_type, span)
    d0 = ir.Var("d0", tensor_type, span)
    d1 = ir.Var("d1", tensor_type, span)
    forwarded = ir.Call(ir.GlobalVar("inner"), [src, d0, d1], pair_type, span)
    statements: list[ir.Stmt] = []
    if bound:
        result = ir.Var("result", pair_type, span)
        statements.append(ir.AssignStmt(result, forwarded, span))
        forwarded = result
    statements.append(ir.ReturnStmt([forwarded], span))
    outer = ir.Function(
        "outer",
        [(src, ir.ParamDirection.In), (d0, ir.ParamDirection.Out), (d1, ir.ParamDirection.Out)],
        [pair_type],
        ir.SeqStmts(statements, span),
        span,
        ir.FunctionType.InCore,
    )
    return ir.Program([inner, outer], "structural_tuple_forward", span), inner, outer


def test_a_forwarded_structural_tuple_keeps_its_delivered_lineage():
    """Direct and bound spellings agree: one entry per delivered value.

    The callee's delivered map names both backing Out params, so the forwarded
    tuple must carry that map back to the caller's arguments — ``[1, 2]``, not
    the ``[None]`` the callee's one-entry position map describes.
    """
    for bound in (False, True):
        program, inner, outer = _structural_tuple_program(bound=bound)
        assert ir.delivered_param_indices(inner, program) == [1, 2], f"bound={bound}"
        assert ir.delivered_param_indices(outer, program) == [1, 2], f"bound={bound}"


def test_a_forwarded_structural_tuple_keeps_its_single_declared_slot():
    """Control: the position-faithful map stays one entry per declared slot.

    A single declared ``TupleType`` keeps its 1-arity position map — only the
    value-faithful variant expands it.
    """
    program, inner, outer = _structural_tuple_program(bound=False)
    assert ir.returned_param_indices(inner, program) == [None]
    assert ir.returned_param_indices(outer, program) == [None]


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-v"]))
