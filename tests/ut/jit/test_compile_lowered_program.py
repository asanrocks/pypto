# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compiling already-lowered IR — the documented ``ir.compile(kernel.lower(...))`` idiom.

``ir.compile``'s input contract is "whatever ``Program`` you handed it"
(``docs/en/user/execution/01-run.md``): pre-pass on the usual path, but the
documented setup compiles ``kernel.lower(...)``, which is already lowered. The
entry-output preparation at the compile boundary must therefore never re-check
a source declaration against the values of a lowered program — the pipeline
has attached layout and storage metadata by then, so a strict comparison
refuses the documented idiom outright. Frontends check their own return
contracts before a program reaches ``ir.compile``; the boundary only recovers
a signature for an entry that declares none.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.compiled_program import CompiledProgram
from pypto.jit.decorator import jit

ROWS = 64


@jit
def add_returns_out(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        left = pl.load(a, [0, 0], [ROWS, ROWS])
        right = pl.load(b, [0, 0], [ROWS, ROWS])
        pl.store(pl.add(left, right), [0, 0], out)
    return out


class TestCompileLoweredProgram:
    """``ir.compile`` accepts a post-pass program."""

    def test_parsing_lane_lowered_program_compiles(self, tmp_path):
        torch = pytest.importorskip("torch")
        a = torch.zeros(ROWS, ROWS, dtype=torch.float32)
        b = torch.zeros(ROWS, ROWS, dtype=torch.float32)
        out = torch.empty(ROWS, ROWS, dtype=torch.float32)

        lowered = add_returns_out.lower(a, b, out)
        compiled = ir.compile(
            lowered, output_dir=str(tmp_path / "parsing"), skip_ptoas=True, dump_passes=False
        )
        assert isinstance(compiled, CompiledProgram)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
