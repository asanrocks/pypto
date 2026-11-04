# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The parameterless facades are annotation forms on the traced lane too.

``buf: pl.Ptr`` / ``ev: pl.AsyncEvent`` spell the handle kinds the way the
parser lane's bare names do (``_MARKER_TYPE_GETTERS``): the class itself is
the annotation, because these kinds have no subscript form. The traced
funnel — ``is_staged_annotation`` + ``annotation_to_type`` — must admit and
convert them, or a traced kernel signature using one is rejected with a
message that names the very annotation it refused.
"""

import pypto.language as pl
import pytest
from pypto.language.distributed.typing import CommCtx
from pypto.language.facade import PARAMETERLESS_FACADES
from pypto.language.protocol import is_staged_annotation
from pypto.language.tracing.decorators.function import function_trace
from pypto.language.typing import Array, Ptr, Scalar, Tensor, Tile, Tuple
from pypto.language.typing.prefetch_handle import AsyncEvent, AsyncSession, PrefetchAsyncContext
from pypto.language.typing.type_utils import annotation_to_type
from pypto.pypto_core import ir


def test_parameterless_classes_are_staged_annotations():
    """Every kind in the parameterless set is admitted as its bare class."""
    assert PARAMETERLESS_FACADES == (Ptr, CommCtx, PrefetchAsyncContext, AsyncEvent, AsyncSession)
    for cls in PARAMETERLESS_FACADES:
        assert is_staged_annotation(cls), cls


def test_subscript_kind_bare_classes_are_not_annotations():
    """The subscript kinds are annotations only once subscripted — a bare
    ``pl.Tensor`` states no shape, so it stays refused (both frontends)."""
    for cls in (Tensor, Tile, Scalar, Array, Tuple):
        assert not is_staged_annotation(cls), cls
    assert not is_staged_annotation(int)


def test_annotation_to_type_converts_parameterless_classes():
    assert isinstance(annotation_to_type(Ptr), ir.PtrType)
    assert isinstance(annotation_to_type(CommCtx), ir.CommCtxType)
    assert isinstance(annotation_to_type(PrefetchAsyncContext), ir.PrefetchAsyncContextType)
    assert isinstance(annotation_to_type(AsyncEvent), ir.AsyncEventType)
    assert isinstance(annotation_to_type(AsyncSession), ir.AsyncSessionType)


def test_traced_kernel_binds_parameterless_handle_params():
    """The standalone traced lane derives the IR params and the return type
    from the bare handle classes."""

    def kernel(
        buf: Ptr,
        ctx: PrefetchAsyncContext,
        ev: AsyncEvent,
        session: AsyncSession,
    ) -> AsyncEvent:
        return ev

    fn = function_trace(kernel)
    assert [str(p.type) for p in fn.params] == [
        "pl.Ptr",
        "pl.PrefetchAsyncContext",
        "pl.AsyncEvent",
        "pl.AsyncSession",
    ]
    assert [str(t) for t in fn.return_types] == ["pl.AsyncEvent"]


def test_program_marker_accepts_parameterless_params():
    """The class-body marker lane accepts the bare handle classes and prints
    them in the parser's own spelling, so the printed program round-trips
    through the parser frontend unchanged."""

    @pl.program
    class HandleParams:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(buf: Ptr, ctx: PrefetchAsyncContext):
            return

    printed = HandleParams.as_python()
    assert "buf: pl.Ptr" in printed
    assert "ctx: pl.PrefetchAsyncContext" in printed
    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(HandleParams, reparsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
