# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The op-zoo's buildable first slice: one kernel per
full-tier op the slice named as gaps.

The **typed-facade gate** is the primary assertion: the tracing frontend
builds the kernel with typed IR — a wired-but-untyped facade surfaces
loudly at the consumer (an op rejects an ``UnknownType`` operand), so a
clean trace is the typed gate. The **structural-equality half** vs
``function_parser`` is added per-op where the AST parses the same source and
the round-trip survives (the "while ``function_parser`` lives"
proviso).

Covered: ``tile.and_``/``tile.or_``/``tile.not_`` (logical — typed gate
only: the printer emits the keyword-safe ``and_``/``or_``/``not_``
spellings for registry names that collide with Python keywords);
``tile.gather_mask``/``tile.scatter_mask`` (mask form —
full conformance with the ``mask_pattern=`` enum kwarg the AST parses);
``tile.gather_compare`` (tuple-return compare-form gather — typed gate
only: the AST binds a tuple temp whose post-canonical shape differs).

**Not yet zoobable**: ``tile.aiv_shard`` /
``tile.aic_gather`` need the split-mode inheritance from the ``split_aiv``
region (the orchestration slice — they still raise); the cross-core
``tile.tpop_*``/``tpush_*`` facades are registered under different names
(inventory ``wired`` lookup). ``pld.*`` distributed ops are NOT
zoo-blocked — the distributed envelope traces (the sub-kernel-call
``device=`` kwarg landed); the name channel is the remaining
Sema-side piece.
"""

import pypto.language as pl
from pypto.language.tracer.decorators.function import function_trace
from pypto.pypto_core import ir

from ._harness import conformance


def _assert_zoo_typed(kernel, *, expect_op: str) -> None:
    """The typed-facade gate: the tracing frontend builds the kernel with
    typed IR and the expected op lands in the body (a wired-but-untyped
    facade would be rejected loudly at the consumer — the trace's success
    is the typed gate)."""
    fn = function_trace(kernel, function_type=ir.FunctionType.InCore, level=pl.Level.CHIP_DIE)
    assert expect_op in str(fn)


# —— tile.and_ / or_ / not_ (logical; typed gate only — the printer emits
#    the keyword-safe spellings for the keyword-colliding registry names) ——


def k_and(x: pl.Tensor[[64, 64], pl.INT32]) -> pl.Tensor[[64, 64], pl.INT32]:
    a = pl.load(x, [0, 0], [64, 64])
    b = pl.load(x, [0, 0], [64, 64])
    c = pl.tile.and_(a, b)
    pl.store(c, [0, 0], x)
    return x


def k_or(x: pl.Tensor[[64, 64], pl.INT32]) -> pl.Tensor[[64, 64], pl.INT32]:
    a = pl.load(x, [0, 0], [64, 64])
    b = pl.load(x, [0, 0], [64, 64])
    c = pl.tile.or_(a, b)
    pl.store(c, [0, 0], x)
    return x


def k_not(x: pl.Tensor[[64, 64], pl.INT16]) -> pl.Tensor[[64, 64], pl.INT16]:
    a = pl.load(x, [0, 0], [64, 64])
    c = pl.tile.not_(a)
    pl.store(c, [0, 0], x)
    return x


def test_tile_and_typed():
    _assert_zoo_typed(k_and, expect_op="pl.tile.and_(")


def test_tile_or_typed():
    _assert_zoo_typed(k_or, expect_op="pl.tile.or_(")


def test_tile_not_typed():
    _assert_zoo_typed(k_not, expect_op="pl.tile.not_(")


# —— tile.gather_mask / scatter_mask (mask form; full conformance) ——


def k_gather_mask(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    a = pl.load(x, [0, 0], [64, 64])
    c = pl.tile.gather_mask(a, mask_pattern=pl.tile.MaskPattern.P0101)
    pl.store(c, [0, 0], x)
    return x


def k_scatter_mask(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
    dst = pl.load(x, [0, 0], [64, 128])
    src = pl.load(x, [0, 0], [64, 64])
    c = pl.tile.scatter_mask(dst, src, mask_pattern=pl.tile.MaskPattern.P0101)
    pl.store(c, [0, 0], x)
    return x


def test_tile_gather_mask_zoo():
    conformance(k_gather_mask, function_trace, canonical=True)


def test_tile_scatter_mask_zoo():
    conformance(k_scatter_mask, function_trace, canonical=True)


# —— tile.gather_compare (tuple-return compare-form gather; typed gate
#    only — the AST binds a tuple temp whose post-canonical shape differs) ——


def k_gather_compare(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    src = pl.load(x, [0, 0], [64, 64])
    tmp = pl.tile.create([64, 64], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
    dst, cdst = pl.tile.gather_compare(src, pl.const(0, pl.FP32), tmp, out_cols=64)
    pl.store(dst, [0, 0], x)
    return x


def test_tile_gather_compare_typed():
    _assert_zoo_typed(k_gather_compare, expect_op="pl.tile.gather_compare(")
