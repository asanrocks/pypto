# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""B5c — a window-buffer allocation's folded Python-int size is ``INDEX`` on both lanes.

The allocation builder is shared by both frontends and types a size by how it
received it: the parsing frontend folds ``256 * 4`` from the AST into a
``ConstInt`` on the index lattice, while the tracing frontend evaluates the
same expression to a Python ``int`` and hands the value to the builder's
``isinstance(size, int)`` branch. That branch used to reify the value as
``INT64``, so one kernel produced two different nodes — a structural
comparison saw the constant's dtype differ and nothing else.

The shaped spelling folds inside the shared ``_compute_window_buffer_bytes``
helper, so both lanes reach the same branch with a literal there too; a
*dynamic* dim never reaches the branch (it stays a ``Mul`` chain) and keeps
its own typing, pinned by the parser-side dynamic test.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.pypto_core import DataType, ir

_SAMPLE = torch.zeros(64, 64, dtype=torch.float32)


@pl.jit.host
def parsed_entry(out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    data_buf = pld.alloc_window_buffer(256 * 4)
    shaped_buf = pld.alloc_window_buffer([64, 128], dtype=pl.FP32)
    pld.window(data_buf, [1, 256], dtype=pl.FP32)
    pld.window(shaped_buf, [64, 128], dtype=pl.FP32)
    return out


@pl.jit_trace.host
def traced_entry(out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    data_buf = pld.alloc_window_buffer(256 * 4)
    shaped_buf = pld.alloc_window_buffer([64, 128], dtype=pl.FP32)
    pld.window(data_buf, [1, 256], dtype=pl.FP32)
    pld.window(shaped_buf, [64, 128], dtype=pl.FP32)
    return out


def _alloc_sizes(program: ir.Program) -> dict[str, ir.Expr]:
    """The alloc op's byte-size argument per buffer name, pre-pass."""
    sizes: dict[str, ir.Expr] = {}

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            call = stmt.value
            if call.op.name == ir.get_op("pld.tensor.alloc_window_buffer").name:
                sizes[call.kwargs["name"]] = call.args[0]
        if isinstance(stmt, ir.SeqStmts):
            for inner in stmt.stmts:
                walk(inner)

    for func in program.functions.values():
        walk(func.body)
    return sizes


def test_a_folded_python_int_size_is_index_on_the_traced_lane():
    """The tracer evaluates the size to a Python ``int``; the shared builder's
    ``isinstance(size, int)`` branch must reify it on the index lattice, not as
    ``INT64`` — including the shaped spelling's folded literal."""
    sizes = _alloc_sizes(traced_entry.specialize(_SAMPLE))
    folded, shaped = sizes["data_buf"], sizes["shaped_buf"]
    assert isinstance(folded, ir.ConstInt)
    assert folded.value == 1024
    assert folded.dtype == DataType.INDEX
    assert isinstance(shaped, ir.ConstInt)
    assert shaped.value == 64 * 128 * 4  # FP32 = 4 bytes
    assert shaped.dtype == DataType.INDEX


def test_the_folded_size_matches_the_parser_lane():
    """Both lanes' folded sizes agree on value and dtype, so a kernel spelled
    once is structurally equal whichever frontend collected it."""
    traced = _alloc_sizes(traced_entry.specialize(_SAMPLE))
    parsed = _alloc_sizes(parsed_entry.specialize(_SAMPLE))
    for name, size in traced.items():
        assert isinstance(size, ir.ConstInt), f"{name}: {size}"
        reference = parsed[name]
        assert isinstance(reference, ir.ConstInt), f"{name}: {reference}"
        assert (size.value, size.dtype) == (reference.value, reference.dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
