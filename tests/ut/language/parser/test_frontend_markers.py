# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend ownership markers used during the tracing migration."""

import pypto.language as pl
from pypto.language import parser
from pypto.pypto_core import ir


def test_parser_markers_are_explicit_and_pending_markers_are_removed() -> None:
    assert pl.function_parser is parser.function_parser
    assert pl.program_parser is parser.program_parser
    assert pl.function is not pl.function_parser
    assert pl.program is not pl.program_parser
    assert not hasattr(pl, "function_pending")
    assert not hasattr(pl, "program_pending")
    assert not hasattr(parser, "function_pending")
    assert not hasattr(parser, "program_pending")


def test_suffixed_function_marker_builds_ir() -> None:
    @pl.function_parser
    def identity(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
        return x

    assert isinstance(identity, ir.Function)
    assert identity.name == "identity"


def test_suffixed_program_and_method_markers_build_ir() -> None:
    @pl.program_parser
    class Program:
        @pl.function_parser
        def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
            return x

    assert isinstance(Program, ir.Program)
    assert [function.name for function in Program.functions] == ["main"]
