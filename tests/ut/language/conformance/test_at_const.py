# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""`pl.at` + the operand boundary deltas + the exhaustive family
anchors.

The collection path is type-complete: anchors across the op families (tile,
tensor, unified/matmul, scalar, const) prove ``is_staged`` +
``bind``/``bare_stmt`` handle every facade kind uniformly — the ops are
already shared, so per-op implementation is not the point.
"""

import pypto.language as pl
from pypto.language.tracer.decorators.function import function_trace
from pypto.language.tracer.semantics import PtoSema
from pypto.pypto_core import ir

from ._harness import conformance

build = function_trace


# —— the pl.at scope ——


def k_at_incore(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [64, 64])
        s = pl.add(t, t)
        pl.store(s, [0, 0], x)
    return x


def k_at_hierarchy(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_split(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def test_at_incore_anchor():
    conformance(k_at_incore, build)


def test_at_hierarchy_anchor():
    conformance(k_at_hierarchy, build)


def test_at_split_anchor():
    conformance(k_at_split, build)


# —— the typed-constant reification (pl.const) ——


def k_const_int(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    c = pl.const(0, pl.FP32)
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.add(t, c)
    pl.store(s, [0, 0], x)
    return x


def k_const_float(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    c = pl.const(1.5, pl.FP32)
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.add(t, c)
    pl.store(s, [0, 0], x)
    return x


def k_const_index(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    # INDEX is the offset dtype: an INDEX constant is legal as a load offset,
    # while a tile/tensor scalar instruction would reject it (upstream #2132 —
    # the operand boundary asks for an explicit cast otherwise).
    c = pl.const(0, pl.INDEX)
    t = pl.load(x, [c, 0], [64, 64])
    s = pl.add(t, t)
    pl.store(s, [0, 0], x)
    return x


def test_const_int_anchor():
    conformance(k_const_int, build)


def test_const_float_anchor():
    conformance(k_const_float, build)


def test_const_index_anchor():
    conformance(k_const_index, build)


def test_const_reified_to_scalar():
    """pl.const yields a typed Scalar wrapping the Const node (the executed
    path's reification of the parser's intercept) — a bind therefore
    emits a typed ``LetStmt``, satisfying the typed staged initialization
    requirement."""
    c = pl.const(0, pl.FP32)
    assert isinstance(c, pl.Scalar)
    expr = c.unwrap()
    assert isinstance(expr, ir.ConstInt) and expr.dtype == pl.FP32
    cf = pl.const(1.5, pl.FP16)
    assert isinstance(cf.unwrap(), ir.ConstFloat)
    assert isinstance(pl.const(0, pl.INDEX).unwrap(), ir.ConstInt)


# —— the exhaustive family anchors (the collection path is uniform) ——


def k_tile_chain(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.tile.muls(t, 2.0)
    r = pl.tile.adds(s, 1.0)
    pl.store(r, [0, 0], x)
    return x


def k_tensor_ops(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    y = pl.tensor.add(x, x)
    z = pl.tensor.muls(y, 1.0)
    return z


# noqa: E501
def k_matmul(
    x: pl.Tensor[[64, 128], pl.FP16], w: pl.Tensor[[128, 64], pl.FP16]
) -> pl.Tensor[[64, 64], pl.FP32]:
    a = pl.load(x, [0, 0], [64, 64])
    b = pl.load(w, [0, 0], [64, 64])
    acc = pl.matmul(a, b)
    pl.store(acc, [0, 0], x)
    return x


def k_gemm(
    x: pl.Tensor[[64, 128], pl.FP16], w: pl.Tensor[[128, 64], pl.FP16]
) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        a = pl.load(x, [0, 0], [64, 64])
        b = pl.load(w, [0, 0], [64, 64])
        acc = pl.matmul(a, b)
        pl.store(acc, [0, 0], x)
    return x


def test_tile_chain_anchor():
    conformance(k_tile_chain, build)


def test_tensor_ops_anchor():
    conformance(k_tensor_ops, build)


def test_matmul_anchor():
    conformance(k_matmul, build)


def test_gemm_anchor():
    """The gemm kernel: straight-line + the pl.at scope, no loop (the first real
    kernel shape)."""
    conformance(k_gemm, build)


# —— guardrails ——


def test_open_with_rejects_non_at():
    """A host context declines at ``create_with_construct`` (None — the
    interpreter wraps the host context manager); only the ``pl.at`` trace
    context manager is staged."""
    sema = PtoSema(name="probe")
    frame = sema.begin_frame()
    assert frame.create_with_construct(None, object()) is None
    frame.finish()


def test_at_deps_deferred():
    """``pl.at(deps=…)`` is collect-tier: the trace context manager
    accepts the full signature and enters the scope ignoring the deps — the
    marker, not an exception, is the contract. The deps graph is the Part 4+
    slice."""
    sema = PtoSema(name="probe")
    frame = sema.begin_frame()
    bag = pl.at(pl.Level.CORE_GROUP, deps=[None])
    opened = frame.create_with_construct(None, bag)
    assert opened is not None  # the staged ``with`` construct opened (deps ignored)
    opened.construct.close()
    frame.finish()
