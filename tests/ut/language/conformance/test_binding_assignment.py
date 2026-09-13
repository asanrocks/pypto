# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural: binding and assignment — the R8 construct-parity sweep.

Both frontends parse the same source. The parser's ``parse_assignment`` /
``parse_annotated_assignment`` / ``_assign_or_let`` (``ast_parser.py:1530``,
``:1834``, ``:1774``) lower each bind to ``let``-on-first / ``assign``-on-rebind,
with a type-changing rebind starting a fresh ``let`` (the "#642 failure mode"
mirror); the tracing ``Sema._default_bind`` (``semantics.py:1166``) is the
deliberate mirror of that rule (both compare through the shared
``_types_match`` seam). Subscript writes desugar on both sides to the
functional ``tensor.assemble`` / ``tile.assemble`` / ``array.update_element``
rebind, with the numpy full-rank offset rule (the parser's
``_build_assemble_offsets_and_extents``, ``ast_parser.py:2442``; the tracing
``subscript_write_offsets``, ``typing/_subscript.py``). Tuple unpack shares
the ``_tuple_tmp`` name-once temp (the parser's
``parse_assignment`` tuple branch, ``:1847``; the tracing
``_emit._structural_projection``). ``pl.const`` yields a typed staged
``Scalar`` on both sides (the parser's ``_parse_typed_constant``, ``:9349``).

**Regime.** Every straight-line anchor below is asserted **raw**-equal:
both lanes emit the identical ``LetStmt``/``AssignStmt`` chain with no pass
state involved. The staged anchors (a rebind inside a staged loop, the
``pl.const`` accumulator, the subscript write inside a loop) are additionally
asserted post-``canonicalize()``: raw equality still holds (neither lane
emits ``iter_arg`` for a plain loop — the loop-carried value is the
self-referential ``AssignStmt`` update), and canonical is the staged-flow
regime proper, where the SSA pass derives the loop-carried value identically
from that update.

**No shared spelling / intended divergences** are pinned at the bottom:
`scalar-index rank-reducing writes` (parser accepts, tracing is loud — the
AST's rank-lifting surface is deliberately not ported, recorded as a
needs-decision acceptance gap); the two directions of the typed-staged-
initialization guard (tracing is loud, the parser accepts — intended by
``docs/design/01-language.md`` §4); the parameter worlds (``pl.Scalar`` is a
runtime parameter on both sides, a plain ``int`` is rejected by both); and
``pl.runtime``, which is a jit-entry-call marker with no kernel-body or parser
spelling at all (out of scope; pinned by ``tests/ut/jit/test_param_worlds.py``).
"""

import pypto.language as pl
import pytest
from pypto.language.parser import function_parser
from pypto.language.tracer.decorators.function import function_trace
from pypto.pypto_core import ir

from ._harness import canonicalize, conformance

build = function_trace


# —— A. subscript write: tensor / tile / array, all-slice keys ——


def k_tensor_write_partial(
    x: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
    out[2:8] = pl.slice(x, [6, 64], [0, 0])
    return out


def k_tensor_write_bare_slice(
    x: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
    out[:] = x
    return out


def k_tensor_write_full_axis(
    x: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
    out[0:32, :] = pl.slice(x, [32, 64], [0, 0])
    return out


def k_tile_write(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    src = pl.load(x, [0, 0], [32, 64])
    t[0:32, :] = src
    pl.store(t, [0, 0], x)
    return x


def k_array_write_const(x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.INT32]:
    arr = pl.array.create(4, pl.INT32)
    arr[0] = 7
    return pl.array.get_element(arr, 0)


def k_array_write_staged_index(x: pl.Tensor[[64], pl.FP32], i: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
    arr = pl.array.create(4, pl.INT32)
    arr[i] = 7
    return pl.array.get_element(arr, 0)


def k_tensor_write_in_loop(
    x: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
    for _i in pl.range(2):
        out[0:32, :] = pl.slice(x, [32, 64], [0, 0])
    return out


def test_tensor_write_partial_anchor():
    """``out[2:8] = src`` on a rank-2 tensor: axis 1 is an implicit full slice
    (offsets always full-rank) — the tracing ``subscript_write_offsets`` pads
    with the numpy rule; the parser's ``_build_assemble_offsets_and_extents``
    does the same."""
    conformance(k_tensor_write_partial, build)


def test_tensor_write_bare_slice_anchor():
    """A bare ``out[:]`` expands to every axis on both sides (the K7/F1 fix:
    a one-axis assemble was the defect; the parser always emitted ``[0, 0]``)."""
    conformance(k_tensor_write_bare_slice, build)


def test_tensor_write_full_axis_anchor():
    """``out[0:32, :]`` — an explicit upper bound on axis 0 and a full slice
    on axis 1, offsets ``[0, 0]``."""
    conformance(k_tensor_write_full_axis, build)


def test_tile_write_anchor():
    """The tile path shares the numpy rule; both lanes desugar to
    ``tile.assemble(t, src, [0, 0])`` and rebind ``t``."""
    conformance(k_tile_write, build)


def test_array_write_const_anchor():
    """``arr[0] = 7`` desugars to ``arr = array.update_element(arr, 0, 7)`` on
    both sides; both retag the bare int literal to the array's element
    dtype (parser ``_parse_array_subscript_assignment``, tracing
    ``array_ops.update_element``)."""
    conformance(k_array_write_const, build)


def test_array_write_staged_index_anchor():
    """A staged index ``arr[i] = 7`` — the index is a ``Scalar`` facade on
    both sides and lowers to the same ``array.update_element`` rebind."""
    conformance(k_array_write_staged_index, build)


def test_tensor_write_in_loop_anchor():
    """A subscript write inside a staged ``for``: the assemble rebind is the
    loop-carried value (the SSA pass derives it identically from the
    self-referential update). Raw-equal and canonical-equal."""
    conformance(k_tensor_write_in_loop, build)
    conformance(k_tensor_write_in_loop, build, canonical=True)


# —— B. staged rebinding ——


def k_rebind_in_for(x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for _i in pl.range(n):
        acc = pl.add(acc, acc)  # same-type rebind: assign-through
    pl.store(acc, [0, 0], x)
    return x


def k_rebind_in_while(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 3:
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_typechange_rebind(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[1, 4096], pl.FP32]:
    a = pl.load(x, [0, 0], [64, 64])
    a = pl.reshape(a, [1, 4096])  # type-changing rebind: fresh let on both sides
    return a


def k_typechange_rebind_in_loop(
    x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    a = pl.load(x, [0, 0], [64, 64])
    for _i in pl.range(n):
        a = pl.reshape(a, [1, 4096])
        a = pl.reshape(a, [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_param_name_rebind(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    x = pl.add(x, x)  # rebind of a parameter name
    return x


def test_rebind_in_for_anchor():
    """A same-type rebind inside a staged ``for``: one mutable ``ir.Var``,
    ``assign`` on the rebind; both lanes, raw and canonical."""
    conformance(k_rebind_in_for, build)
    conformance(k_rebind_in_for, build, canonical=True)


def test_rebind_in_while_anchor():
    """The ``while`` sibling of the same-name rebind."""
    conformance(k_rebind_in_while, build)
    conformance(k_rebind_in_while, build, canonical=True)


def test_typechange_rebind_anchor():
    """A straight-line type-changing rebind: both ``_assign_or_let`` and
    ``_default_bind`` start a fresh ``let`` (``Tensor[[64,64]]`` →
    ``Tensor[[1,4096]]``) instead of assigning through the old ``ir.Var`` —
    the mirrored relaxed-reassignment rule."""
    conformance(k_typechange_rebind, build)


def test_typechange_rebind_in_loop_anchor():
    """The type-changing rebind under a staged loop: the fresh ``let`` shadows
    the name identically on both sides, so the raw IR is equal.

    The post-canonical half of this staged-flow anchor is **red on both
    lanes** — a shared frontend bug (F14, see
    ``test_typechange_rebind_in_loop_shared_free_var``): the fresh ``let``
    lands *inside* the loop body, the name after the loop resolves to that
    body-local Var, and ``ConvertToSSA`` builds a loop ``init_values`` entry
    from a free variable. Both frontends produce the byte-identical invalid
    IR, so this is not a parity divergence — it is the same ``§7`` relaxed-
    reassignment hole on both sides; a red anchor per the tracker's rule."""
    conformance(k_typechange_rebind_in_loop, build)


def test_typechange_rebind_in_loop_shared_free_var():
    """**F14 — a type-changing rebind inside a staged loop yields a free
    variable in the SSA loop init, on BOTH frontends.**

    ``k_typechange_rebind_in_loop`` traced and parsed produce the identical
    body (``a`` → fresh ``let`` ``a_1`` inside the loop → fresh ``let``
    ``a_2``; ``store(a_2)`` after the loop). The post-loop read therefore
    names a loop-body-local Var; ``ConvertToSSA``'s post verification rejects
    it (``SSAVerify: 'a' used outside its defining scope`` and
    ``UseAfterDefCheck: 'a' used before definition``). The same source in a
    straight line canonicalizes fine, and a *same-type* rebind in the loop
    canonicalizes fine (``test_rebind_in_for_anchor``) — the hole is the
    fresh-``let`` path of the §7 relaxed-reassignment rule entering a
    loop. No frontend edit is made here; the anchor is recorded red.

    Red check: ``canonicalize(function_trace(k_typechange_rebind_in_loop))``
    (under the unit-test verification context) raises the ``ConvertToSSA``
    post-verification error; ``function_parser`` of the same kernel raises
    it identically."""
    conformance(k_typechange_rebind_in_loop, build)  # raw: both lanes identical
    for build_fn in (build, function_parser):
        with pytest.raises(Exception, match="used outside its defining scope|used before definition"):
            canonicalize(build_fn(k_typechange_rebind_in_loop))


def test_param_name_rebind_anchor():
    """A rebind of a parameter's name: both lanes emit the same shadowing
    bind (no ``AssignStmt`` through the parameter ``Var``)."""
    conformance(k_param_name_rebind, build)


# —— C. defs / store-target first-encounter order ——


def k_defs_store_order(
    x: pl.Tensor[[16, 16], pl.FP32], y: pl.Tensor[[16, 16], pl.FP32]
) -> pl.Tensor[[16, 16], pl.FP32]:
    t1 = pl.load(x, [0, 0], [16, 16])
    t2 = pl.load(y, [0, 0], [16, 16])
    t3 = pl.add(t1, t2)
    pl.store(t3, [0, 0], y)
    t4 = pl.add(t2, t2)
    pl.store(t4, [0, 0], x)
    return x


def test_defs_store_order_anchor():
    """Interleaved defs and store targets keep source (first-encounter) order
    on both lanes: ``t1, t2`` defs, the ``t3`` store, the ``t4`` def, the
    ``t4`` store.

    The cited ``136a0f457`` ("iterate defs and store targets in
    first-encounter order") is a C++ *pass* determinism fix
    (``inline_functions_pass`` / ``scope_outline_utils`` iterated
    pointer-keyed containers whose hash order follows ASLR); it is out of the
    frontend comparison's regime (the harness runs neither outlining nor a
    pointer-keyed walk). Its IR-level contract is pinned by
    ``tests/ut/ir/transforms/test_outline_incore_scopes.py`` (the store-target
    walk is encounter-ordered) and by the inline-regime anchors in
    ``test_inline.py``. The frontend residue — emission order — is what this
    anchor holds: both frontends emit the same statement order, raw-equal."""
    conformance(k_defs_store_order, build)


# —— D. pl.const staged init ——


def k_const_straightline(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    zero = pl.const(0.0, pl.FP32)
    _one = pl.const(1, pl.INT32)
    _flag = pl.const(True, pl.BOOL)
    return pl.add(x, zero)


def k_const_loop_init(
    x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.const(0.0, pl.FP32)
    for _i in pl.range(n):
        acc = pl.add(acc, pl.const(1.0, pl.FP32))
    out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
    out[0:1, 0:1] = pl.slice(x, [1, 1], [0, 0])
    return out


def test_const_straightline_anchor():
    """``pl.const`` yields the same typed ``Const*`` node on both sides
    (parser ``_parse_typed_constant``, tracing ``typing/const.py``) — the
    straight-line bind is a plain ``Let`` of the constant."""
    conformance(k_const_straightline, build)


def test_const_loop_init_anchor():
    """The documented typed staged initialization: ``acc = pl.const(0.0,
    pl.FP32)`` creates a staged local, the loop rebind reuses its mutable
    ``ir.Var``, and the SSA pass derives the loop-carried value — identical
    on both sides, raw and canonical."""
    conformance(k_const_loop_init, build)
    conformance(k_const_loop_init, build, canonical=True)


# —— E. tuple unpack ——


def k_tuple_unpack_gather(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 8], pl.INT32]:
    dst, _cdst = pl.gather(x, kvalue=pl.const(4.0, pl.FP32), cmp_mode="eq", out_cols=8)
    return dst


def test_tuple_unpack_anchor():
    """``dst, cdst = <multi-result op>`` names the shared tuple node once as a
    ``_tuple_tmp`` ``Let`` and binds each projection with
    ``TupleGetItemExpr`` — the parser's ``parse_assignment`` tuple branch and
    the tracing ``_emit._structural_projection`` share the temp (the temp
    *name* differs — ``_tuple_tmp`` vs ``_tuple_tmp_1`` — which auto-mapping
    ignores)."""
    conformance(k_tuple_unpack_gather, build)


# —— F. parameter worlds (the shared ``pl.runtime`` world) ——


def k_scalar_param(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.mul(x, n)


def test_scalar_param_world_anchor():
    """A ``pl.Scalar[pl.INT32]`` parameter is a runtime parameter on both
    lanes: same signature type, same ``Var``. ``pl.runtime`` itself is the
    jit *entry-call* spelling for the same world and has no kernel-body or
    parser spelling, so it cannot be an anchor (see the out-of-scope pin
    below; the world is pinned by ``tests/ut/jit/test_param_worlds.py``)."""
    conformance(k_scalar_param, build)


# —— divergences and no-shared-spelling pins ——


def test_typed_staged_init_bare_literal_rejected_by_tracing():
    """**Intended divergence — not an anchor.**

    ``acc = 0`` used as staged state is the loud typed-staged-initialization
    error on the tracing side (``docs/design/01-language.md`` §4: a host
    literal cannot initialize staged state that survives a join — the bind
    produces no staged local, ConvertToSSA derives no loop-carried value,
    and the kernel would silently compute the wrong result). The AST parser
    has no host world in a kernel body, so it accepts the same source. The
    two frontends therefore cannot agree and the harness has no work to do;
    this pin keeps the divergence intentional in both directions."""

    def kernel(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
        acc = 0
        for _i in pl.range(n):
            acc = pl.add(acc, pl.const(1, pl.INT32))
        return x

    with pytest.raises(TypeError, match="typed staged initialization"):
        function_trace(kernel)
    # parser: no host world — the same source parses (to a Let of ConstInt).
    assert isinstance(function_parser(kernel), ir.Function)


def test_typed_staged_init_demotion_rejected_by_tracing():
    """The symmetric direction of the same intended divergence: a staged
    local cannot be demoted to a host literal under open staged control flow
    (reads after the back-edge would inline the literal instead of re-reading
    the mutable ``ir.Var``; ``docs/design/01-language.md`` §4). The parser
    accepts; tracing is loud."""

    def kernel(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
        _acc = pl.const(0, pl.INT32)
        for _i in pl.range(n):
            _acc = 5
        return x

    with pytest.raises(TypeError, match="cannot demote it to a host literal"):
        function_trace(kernel)
    assert isinstance(function_parser(kernel), ir.Function)


def test_scalar_index_write_has_no_shared_spelling():
    """**No shared spelling — needs-decision, not a bug.**

    A scalar (rank-reducing) index write — ``out[0, :] = src`` — is accepted
    by the parser (``_build_assemble_offsets_and_extents`` adds the axis to
    ``drop_dims`` and ``_lift_subscript_write_source_rank`` reshapes the rhs)
    and rejected loudly by tracing (``subscript_write_offsets``: "scalar
    indices (rank-reducing writes) are not supported by the traced
    frontend"). The tracing docstring records the omission; no design doc or
    manifest line intends it, so it is a needs-decision acceptance gap (loud,
    never silently wrong), not an anchor. Both lanes reject a *full-rank*
    scalar element write, so the rejected-by-both shape has no accepted
    spelling either."""

    def kernel(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        out[0, :] = pl.slice(x, [64], [0, 0])
        return out

    with pytest.raises(TypeError, match="scalar indices"):
        function_trace(kernel)
    assert isinstance(function_parser(kernel), ir.Function)


def test_host_param_world_rejected_by_both_lanes():
    """A plain ``int`` parameter is not a ``@pl.function`` world on either
    lane: the tracing frontend needs a facade annotation, the parser rejects
    the annotation outright. (The jit envelope's host-knob world — which
    ``pl.runtime`` cannot reclassify — is out of this construct's scope.)"""

    def kernel(x: pl.Tensor[[64], pl.FP32], n: int) -> pl.Tensor[[64], pl.FP32]:
        return pl.mul(x, n)

    with pytest.raises(TypeError, match="annotation"):
        function_trace(kernel)
    with pytest.raises(Exception, match="annotation"):
        function_parser(kernel)


def test_runtime_marker_has_no_kernel_body_spelling():
    """``pl.runtime`` is the **jit entry-call** marker for a runtime scalar
    (``typing/_runtime_scalar.py``): it appears in host call arguments, never
    in kernel source. In a kernel body tracing cannot unwrap it (annotation
    only) and the parser does not know the operation — no shared accepted
    spelling, so out of scope for this construct. The entry-call world is
    pinned by ``tests/ut/jit/test_param_worlds.py``."""

    def kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        n = pl.runtime(5)
        return pl.add(x, n)

    with pytest.raises(Exception):
        function_trace(kernel)
    with pytest.raises(Exception, match="runtime"):
        function_parser(kernel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
