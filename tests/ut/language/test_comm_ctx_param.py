# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pld.CommCtx`` as a staged parameter kind — the signature inspection
contract for the parameterless annotation.

The subscript kinds annotate with facade *instances* (``pl.Tensor[[s], dt]``);
``pld.CommCtx`` has no parameters, so the parser's spelling is the bare name
and the class itself is the annotation. Both frontends must resolve it to one
``ir.CommCtxType`` parameter, and the value threads through calls as a facade
(the closed vocabulary already includes it — ``FACADE_TYPES``).
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.language.parser import program_parser
from pypto.pypto_core import ir


def test_function_param_binds_a_comm_ctx_type_param():
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
        _rank = pld.system.rank(data_ctx)
        pld.system.wait(data, offsets=[0], expected=1, cmp=pld.WaitCmp.Eq)

    assert [type(p.type) for p in kernel.params[-1:]] == [ir.CommCtxType]


def test_program_method_param_binds_a_comm_ctx_type_param():
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data_ctx: pld.CommCtx):
            _rank = pld.system.rank(data_ctx)

    traced = pl.program(Prog)
    main = traced.functions[traced.get_global_var("main")]
    assert [type(p.type) for p in main.params] == [ir.CommCtxType]


def test_both_frontends_agree_on_the_comm_ctx_param():
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            _rank = pld.system.rank(data_ctx)

    class ParserProg:
        @pl.function_parser(type=pl.FunctionType.Orchestration)
        def main(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            _rank = pld.system.rank(data_ctx)

    ir.assert_structural_equal(pl.program(Prog), program_parser(ParserProg))


def test_unannotated_parameter_still_rejects_loudly():
    with pytest.raises(TypeError, match="parameter 'data' needs a"):

        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, data, data_ctx: pld.CommCtx):
                _rank = pld.system.rank(data_ctx)
