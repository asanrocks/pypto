# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""While-carry probe for ``pl.intrin`` — verifies the
tracing-side while-assembly channel round-trips (trace → print
``pl.while_(init_values=)``/``pl.cond`` → parse → structural-equal)."""

import pypto.language as pl
from pypto.language.tracer.decorators.function import function_trace
from pypto.pypto_core import ir


def _while_acc_intrin(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.FP32]:
    acc0 = pl.const(0.0, pl.FP32)
    cnt0 = pl.const(0, pl.INT32)
    (acc, cnt) = pl.intrin.carry(acc0, cnt0)
    while cnt < n:
        acc = pl.add(acc, pl.const(1.0, pl.FP32))
        cnt = pl.add(cnt, pl.const(1, pl.INT32))
        pl.intrin.yield_(acc, cnt)
    (out, _cnt_out) = pl.intrin.leak()
    return out


class TestIntrinWhileCarry:
    def test_while_carry_traces_and_roundtrips(self) -> None:
        # function_trace collects; the gate's PYPTO_VERIFY_LEVEL=roundtrip
        # instruments the print→parse→structural-equal check.
        fn = function_trace(_while_acc_intrin)
        assert isinstance(fn, ir.Function)
