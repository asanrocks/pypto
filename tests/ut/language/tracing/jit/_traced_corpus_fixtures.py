# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The two focused tracing-jit fixtures of the first wave.

Defined once and imported by both the CPU acceptance suite
(``tests/ut/language/tracing/jit/test_minimal_corpus.py``) and the device acceptance suite
(``tests/st/examples/02_intermediate/test_minimal_corpus.py``): the same kernel
must be the one whose IR is pinned and the one that runs on hardware, or the two
suites can drift into testing different programs.
"""

import pypto.language as pl

__all__ = ["annotated_add", "controlled_accumulate"]


@pl.jit_trace
def controlled_accumulate(
    x: pl.Tensor,
    n: pl.Scalar[pl.INDEX],
    scale: float,
    out: pl.Out[pl.Tensor],
):
    """F1: a runtime loop carried into a runtime branch, with a baked host knob."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [64, 64])
        acc = tile
        for _i in pl.range(n):
            acc = pl.add(acc, tile)
        if n > 1:
            result = pl.mul(acc, scale)
        else:
            result = pl.sub(acc, scale)
        pl.store(result, [0, 0], out)
    return out


@pl.jit_trace
def annotated_add(
    x: pl.Tensor[[64, 64], pl.FP32],
    n: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    """F2: fully annotated entry; the scalar is a runtime value."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [64, 64])
        pl.store(pl.add(tile, n), [0, 0], out)
    return out
