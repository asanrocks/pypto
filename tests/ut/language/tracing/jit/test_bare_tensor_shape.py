# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""B7 — a ``Tensor`` parameter's ``.shape`` is readable inside a trace.

The tracing contract states that a kernel whose annotations leave a shape
unknown (a bare ``pl.Tensor``) re-traces per call, so a body that derives
extents from the parameter must see the call site's shapes. The facade used
to answer ``None`` in expr mode regardless of the annotation, so ``a.shape``
had no tracing spelling at all — the parser lane lowers the read
(``parse_attribute``: a static extent folds to its constant, a dynamic one
is a runtime ``pl.tensor.dim`` read that folds onto a param-declared extent
symbol in an orchestration body).

The fix mirrors that lowering lazily on the expr-wrapping facade: the read
emits only when a body performs it, and both folding regimes agree with the
parser lane's printed IR.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from pypto.pypto_core import ir

N = pl.dynamic("N")


@pl.jit_trace
def bare_shape(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    rows, cols = a.shape
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.load(a, [0, 0], [rows, cols]), [0, 0], out)
    return out


@pl.jit_trace
def concrete_shape(a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    rows, cols = a.shape
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.load(a, [0, 0], [rows, cols]), [0, 0], out)
    return out


@pl.jit_trace
def dim_spelled(a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    rows = pl.tensor.dim(a, 0)
    cols = pl.tensor.dim(a, 1)
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.load(a, [0, 0], [rows, cols]), [0, 0], out)
    return out


@pl.jit_trace
def dyn_orchestration(x: pl.Tensor[[N, 16], pl.FP32], out: pl.Out[pl.Tensor[[N, 16], pl.FP32]]):
    rows = x.shape[0]
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.load(x, [0, 0], [rows, 16]), [0, 0], out)
    return out


@pl.jit_trace
def traced_entry(x: pl.Tensor[[N, 16], pl.FP32], out: pl.Out[pl.Tensor[[N, 16], pl.FP32]]):
    traced_leaf(x, out)
    return out


@pl.jit_trace.incore
def traced_leaf(x: pl.Tensor[[N, 16], pl.FP32], out: pl.Out[pl.Tensor[[N, 16], pl.FP32]]):
    rows = x.shape[0]
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.load(x, [0, 0], [rows, 16]), [0, 0], out)
    return out


captured: list[Any] = []


@pl.jit_trace
def grab_facade(a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
    captured.append(a)  # host side effect: the expr-wrapping facade under test
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.load(a, [0, 0], [16, 16]), [0, 0], out)
    return out


def _entry_of(program: ir.Program) -> ir.Function:
    (entry,) = program.functions.values()
    return entry


def test_bare_annotation_reads_the_call_site_shape():
    program = bare_shape.specialize(torch.zeros(16, 16), torch.zeros(16, 16))
    entry = _entry_of(program)
    assert str(entry.params[0].type) == "pl.Tensor[[16, 16], pl.FP32]"
    # Static extents fold to constants — no runtime dim reads in the body.
    assert "tensor.dim" not in str(program)


def test_concrete_annotation_reads_its_shape():
    program = concrete_shape.specialize(torch.zeros(16, 16), torch.zeros(16, 16))
    assert "pl.tile.load(a, [0, 0], [16, 16], [16, 16])" in str(program)


def test_tensor_dim_spelling_still_specializes():
    dim_spelled.specialize(torch.zeros(16, 16), torch.zeros(16, 16))


def test_dynamic_extent_folds_onto_the_param_symbol():
    program = dyn_orchestration.specialize(torch.zeros(32, 16), torch.zeros(32, 16))
    text = str(program)
    # The orchestration fold: the read names N itself, like the parser lane.
    assert "pl.tile.load(x, [0, 0], [N, 16], [N, 16])" in text
    assert "tensor.dim" not in text


def test_incore_body_emits_the_runtime_dim_read():
    program = traced_entry.specialize(torch.zeros(32, 16), torch.zeros(32, 16))
    text = str(program)
    assert "rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)" in text


def test_an_expr_wrapping_facade_is_refused_at_a_host_call():
    grab_facade.specialize(torch.zeros(16, 16), torch.zeros(16, 16))
    (leaked,) = captured
    assert leaked.shape == (16, 16)  # the derived read, outside any trace
    with pytest.raises(TypeError, match="expr-wrapping facade"):
        concrete_shape(leaked, torch.zeros(16, 16))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
