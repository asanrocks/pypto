# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PEP 563 annotations: the defining namespace is part of the declaration.

With ``from __future__ import annotations``, ``pl.Tensor[[rows, 64], ...]`` is
the string ``"pl.Tensor[[rows, 64], pl.FP32]"`` until something evaluates it.
A kernel built by a factory closes over ``rows``, and only the defining
namespace can resolve it — ``typing.get_type_hints`` sees module globals alone
and raises ``NameError``.

The jit therefore captures the namespace once, at decoration, before it
classifies the signature: classification resolves the hints too, so a capture
taken later could not repair the first lookup. The same namespace must reach
every later resolution — the constructor's parameter worlds, the deferred
trace, and the recorded specializer.
"""

from __future__ import annotations

import pypto.language as pl
import pytest
import torch

F32 = torch.float32

ROWS = 64


def _eager_factory(rows: int):
    """An annotation-determined kernel (every parameter fully shaped)."""

    @pl.jit_trace
    def scaled_eager(x: pl.Tensor[[rows, 64], pl.FP32], out: pl.Out[pl.Tensor[[rows, 64], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(x, [0, 0], [64, 64])
            pl.store(pl.mul(tile, 2.0), [0, 0], out)
        return out

    return scaled_eager


def _lazy_factory(rows: int):
    """A call-dependent kernel (bare ``x``), whose *output* still names ``rows``."""

    @pl.jit_trace
    def scaled_lazy(
        x: pl.Tensor,
        out: pl.Out[pl.Tensor[[rows, 64], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(x, [0, 0], [64, 64])
            pl.store(pl.mul(tile, 2.0), [0, 0], out)
        return out

    return scaled_lazy


def _tensor(rows: int = ROWS) -> torch.Tensor:
    return torch.randn(rows, 64, dtype=F32)


def _out(rows: int = ROWS) -> torch.Tensor:
    return torch.empty(rows, 64, dtype=F32)


class TestPostponedAnnotations:
    """A factory-local name in an annotation resolves where it was written."""

    def test_eager_factory_decorates_and_compiles(self):
        kernel = _eager_factory(ROWS)
        program = kernel.compile_for_test(_tensor(), _out())
        assert "scaled_eager_incore_0" in {fn.name for fn in program.functions.values()}

    def test_eager_factory_keeps_the_annotation_determined_extent(self):
        """``rows`` is the declared extent, not a sample-derived specialization."""
        kernel = _eager_factory(ROWS)
        entry = kernel.fn
        assert [dim.value for dim in entry.params[0].type.shape] == [ROWS, 64]

    def test_lazy_factory_decorates_and_compiles(self):
        kernel = _lazy_factory(ROWS)
        program = kernel.compile_for_test(_tensor(), _out())
        assert "scaled_lazy_incore_0" in {fn.name for fn in program.functions.values()}

    def test_lazy_factory_still_reports_the_direction(self):
        """The direction comes from the same resolved hints as the world check."""
        kernel = _lazy_factory(ROWS)
        assert kernel.output_param_names == ("out",)

    def test_unknown_name_is_still_loud(self):
        """Capturing a namespace must not turn an unknown name into a silent pass."""

        def factory():
            @pl.jit_trace
            def broken(
                # The name exists nowhere, by design — see the assertion below.
                x: pl.Tensor[[undefined_extent, 64], pl.FP32],  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
                out: pl.Out[pl.Tensor[[ROWS, 64], pl.FP32]],
            ):
                return out

            return broken

        with pytest.raises(NameError, match="undefined_extent"):
            factory()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
