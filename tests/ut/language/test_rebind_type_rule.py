# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The relaxed reassignment rule — one name, fresh Var on type change.

A same-type rebind assigns through the name's mutable ``ir.Var`` (§7 of the
language design); a **type-changing** rebind starts a fresh ``let`` instead.
The split is load-bearing: assign-through pins the Var's declared type to the
first binding, so every later trace-time type query on the name reads the
stale type — the deepseek v4-pro corpus's ``g_row_max = pl.reshape(g_row_max,
[1, 8])`` over an ``[8, 1]`` local broadcast a ``[1,8]`` maximum against the
stale ``[8,1]`` and surfaced seventy lines later as ``reshape 64 != 8``.
Both frontends implement the rule through the same typing seam, so they
accept, let, and reject identically.
"""

import pypto.language as pl
from pypto.pypto_core import ir


def _structural_equal(left, right) -> bool:
    return ir.structural_equal(left, right, enable_auto_mapping=True)


class TestRelaxedReassignment:
    def test_rebound_name_carries_the_new_type(self):
        """The regression pin: after ``rm = pl.reshape(rm, [1, 8])`` the name
        is typed ``[1, 8]`` — the maximum stays ``[1,8]`` instead of
        broadcasting against the stale ``[8,1]`` (the corpus failure)."""

        @pl.function
        def k(y: pl.Out[pl.Tensor[[8, 1024], pl.FP32]]):
            src = pl.create_tensor([8, 1024], dtype=pl.FP32)
            rm = pl.row_max(pl.abs(src[0:8, 0:1024]))  # [8, 1]
            rm = pl.reshape(rm, [1, 8])  # type-changing rebind → fresh Var
            am = pl.maximum(pl.full([1, 8], dtype=pl.FP32, value=1e-5), rm)
            y[0:1, 0:8] = pl.div(pl.full([1, 8], dtype=pl.FP32, value=127.0), am)
            return y

        text = k.as_python()
        # The stale-type bug would have widened the maximum to [8, 8] and
        # failed the reshape; a clean trace with a [1, 8] maximum is the fix.
        assert "pl.Tensor[[1, 8], pl.FP32] = pl.tensor.maximum" in text.replace("\n", " ")

    def test_tile_shape_rebind_relaxes_too(self):
        """The corpus ``tile = pl.load(...)`` pipelining idiom: successive
        chunk widths on one name."""

        @pl.function
        def k(x: pl.Tensor[[8, 384], pl.FP32], y: pl.Out[pl.Tensor[[8, 384], pl.FP32]]):
            tile = pl.load(x, [0, 0], [8, 256])
            pl.store(tile, [0, 0], y)
            tile = pl.load(x, [0, 256], [8, 128])  # different width, same name
            pl.store(tile, [0, 256], y)
            return y

        text = k.as_python().replace("\n", " ")
        # The rebind mints a fresh, printer-disambiguated let (tile_1) with
        # the new width — the relaxed rule on the Tile lane.
        assert "tile_1:" in text
        assert "pl.tile.load(x, [0, 0], [8, 256]" in text
        assert "pl.tile.load(x, [0, 256], [8, 128]" in text

    def test_both_frontends_agree(self):
        """Frontend parity: the parser and the tracer build structurally equal
        IR for a type-changing rebind (both mint the fresh let)."""

        @pl.function
        def traced(y: pl.Out[pl.Tensor[[8, 1024], pl.FP32]]):
            src = pl.create_tensor([8, 1024], dtype=pl.FP32)
            rm = pl.row_max(pl.abs(src[0:8, 0:1024]))
            rm = pl.reshape(rm, [1, 8])
            am = pl.maximum(pl.full([1, 8], dtype=pl.FP32, value=1e-5), rm)
            y[0:1, 0:8] = pl.div(pl.full([1, 8], dtype=pl.FP32, value=127.0), am)
            return y

        @pl.function_parser
        def parsed(y: pl.Out[pl.Tensor[[8, 1024], pl.FP32]]):
            src = pl.create_tensor([8, 1024], dtype=pl.FP32)
            rm = pl.row_max(pl.abs(src[0:8, 0:1024]))
            rm = pl.reshape(rm, [1, 8])
            am = pl.maximum(pl.full([1, 8], dtype=pl.FP32, value=1e-5), rm)
            y[0:1, 0:8] = pl.div(pl.full([1, 8], dtype=pl.FP32, value=127.0), am)
            return y

        assert _structural_equal(traced, parsed)

    def test_type_changing_rebind_round_trips(self):
        """Two lets on one name print and reparse to structurally equal IR
        (the round-trip dialect re-lets on the second, type-different
        binding — the parser applies the same relaxed rule)."""

        @pl.function
        def k(y: pl.Out[pl.Tensor[[8, 1024], pl.FP32]]):
            src = pl.create_tensor([8, 1024], dtype=pl.FP32)
            rm = pl.row_max(pl.abs(src[0:8, 0:1024]))
            rm = pl.reshape(rm, [1, 8])
            y[0:1, 0:8] = rm
            return y

        round_tripped = pl.parse(k.as_python())
        assert _structural_equal(k, round_tripped)

    def test_same_type_rebind_still_assigns_through(self):
        """The relaxed rule does not touch ordinary rebinding: same-type
        rebinds (accumulators, refreshed loads) keep the one-Var mutable-local
        semantics."""

        @pl.function
        def k(y: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
            acc = pl.create_tensor([4, 4], dtype=pl.FP32)
            acc = pl.add(acc, 1.0)
            acc = pl.add(acc, 2.0)
            y[0:4, 0:4] = acc
            return y

        text = k.as_python()
        assert text.count("pl.tensor.add") == 2

    def test_scalar_rebind_between_host_and_staged_is_untouched(self):
        """The rule governs staged rebinding only; host locals (counters,
        flags) rebind freely as always."""

        @pl.function
        def k(y: pl.Out[pl.Tensor[[4], pl.FP32]]):
            n = 0
            for _ in pl.range(2):
                n = n + 1
            y[0:4] = pl.full([4], dtype=pl.FP32, value=float(n))
            return y

        assert "pl.range(2)" in k.as_python()
