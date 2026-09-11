# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Program-assembly name uniqueness across the frozen/variant lanes.

An eagerly traced callee can be reached twice in one program build — as a
Submit reference (which registers the frozen decoration-time function) and
as a call (which mints a per-call variant). Both are distinct functions;
``_ProgramAssembly`` must reserve the base name so the variant gets a
suffixed name instead of colliding on ``ir.Program``'s program-unique name
check.
"""

import pypto.language as pl
import torch


# Module-level kernels: ``self`` materialization (the jit registry proxy)
# requires the sibling to be a module global, not a test-method closure.
@pl.jit.incore
def _dual_lane_callee(
    x: pl.Tensor[[8], pl.FP32], y: pl.Out[pl.Tensor[[8], pl.FP32]]
) -> pl.Tensor[[8], pl.FP32]:
    y[:] = x
    return y


@pl.jit
def _dual_lane_caller(
    x: pl.Tensor, y: pl.Tensor
):  # bare facade classes: the lazy jit lane (the corpus shape)
    _out, _tid = pl.spmd_submit(self._dual_lane_callee, x, y, core_num=1)  # noqa: F821
    self._dual_lane_callee(x, y)  # noqa: F821 — the same callee via the call lane


def test_submit_and_call_share_one_variant():
    """Submit reference + call of one callee in a single trace: both lanes
    resolve through the same per-call instantiation, so the program carries
    exactly one callee — the shared variant (deduped on the variant key), and
    the frozen decoration-time function never joins the program. The former
    duplicate-name collision (frozen + variant both registered) is gone by
    construction."""
    a = torch.zeros(8, dtype=torch.float32)
    b = torch.empty(8, dtype=torch.float32)
    bridged, _ = _dual_lane_caller._bridge_args((a, b))
    program = _dual_lane_caller._build_program(bridged)
    names = sorted(fn.name for fn in program.functions.values())
    assert names == ["_dual_lane_callee", "_dual_lane_caller"], names


def test_unique_name_reserves_registered_base():
    from pypto.language.tracer.decorators.jit import (  # noqa: PLC0415 - import-cycle guard
        _ProgramAssembly,
        _renamed_function,
    )

    @pl.function
    def _traced(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        return x

    assembly = _ProgramAssembly()
    assembly.register(_renamed_function(_traced, "k"))
    assert assembly.unique_name("k") == "k_2"
    assert assembly.unique_name("k") == "k_3"
    assert assembly.unique_name("other") == "other"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
