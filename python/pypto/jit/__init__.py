# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO JIT compilation module — the quarantined AST-era runtime.

This package is the pre-migration ``@pl.jit`` implementation: AST
specialization (source rewriting into ``@pl.program`` form), the compile
cache, and device execution. ``pl.jit`` itself now collects through
tracing (:mod:`pypto.language.tracer.decorators.jit`) — including the
torch-argument bridge, multi-function program assembly, and the compile /
dispatch shell. This runtime survives temporarily under
``pl.jit_specializer`` for its own unit tests (specializer-mechanics
coverage); its behavioral tiers — torch-argument specialization,
multi-function assembly, dispatch, and ``@pl.jit.extern`` — are covered by
the tracing jit. It is deleted once the remaining mechanics tests port —
do not build new features here.

Example (quarantined form)::

    import pypto.language as pl

    @pl.jit_specializer
    def tile_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            M, N = a.shape
            tile_a = pl.load(a, [0, 0], [M, N])
            tile_b = pl.load(b, [0, 0], [M, N])
            tile_c = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
        return c

    a = torch.randn(128, 128)
    b = torch.randn(128, 128)
    c = torch.empty(128, 128)

    # First call: specializes, compiles, and executes on NPU.
    tile_add(a, b, c)  # c is modified in-place on device

    # Subsequent calls with the same shape/dtype: served from cache.
    tile_add(a, b, c)  # no recompilation
"""

from pypto._cache_config import CacheConfig, CacheStats, cache_stats, configure_cache

from .decorator import JITFunction, jit

__all__ = ["CacheConfig", "CacheStats", "JITFunction", "cache_stats", "configure_cache", "jit"]
