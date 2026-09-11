# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The tracing-side contract of ``pl.func_attr`` — the body-prologue attribute
directive.

``pl.func_attr`` is an expression-statement marker under tracing: the trace
hook validates and coerces the dict (the value rules mirror the parser's
``_parse_func_attr_value``) and delegates the prologue-placement check and the
merge to the tracing context. The prologue rule is enforced by the trace
interpreter's first-statement query — the same rule the parser enforces at
the AST level, so both frontends accept and reject identical programs.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.diagnostics import ParserSyntaxError
from pypto.language.parser import function_parser
from pypto.language.tracer.decorators.function import function_trace

_INCORE = {"function_type": ir.FunctionType.InCore}


def _collect(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
    _ = pl.tile.add(qk, qk)


def test_collects_into_function_attrs():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        pl.func_attr({"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
        _ = pl.tile.add(qk, qk)

    attrs = dict(function_trace(kernel, **_INCORE).attrs)
    assert attrs["split"] == 1  # the lowered SplitMode integer
    assert attrs["split_aiv"] is True


def test_docstring_may_precede_the_prologue():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        """A docstring is not a statement; the prologue follows it."""
        pl.func_attr({"split_aiv": True})
        _ = pl.tile.add(qk, qk)

    assert dict(function_trace(kernel, **_INCORE).attrs)["split_aiv"] is True


def test_multiple_directives_merge():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        pl.func_attr({"split": pl.SplitMode.UP_DOWN})
        pl.func_attr({"split_aiv": True})
        _ = pl.tile.add(qk, qk)

    attrs = dict(function_trace(kernel, **_INCORE).attrs)
    assert attrs["split"] == 1
    assert attrs["split_aiv"] is True


def test_decorator_attrs_and_body_directive_merge():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        pl.func_attr({"split_aiv": True})
        _ = pl.tile.add(qk, qk)

    attrs = dict(function_trace(kernel, attrs={"dual_aiv_dispatch": False}, **_INCORE).attrs)
    assert attrs["dual_aiv_dispatch"] is False
    assert attrs["split_aiv"] is True


def test_duplicate_key_across_directives_is_loud():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        pl.func_attr({"split_aiv": True})
        pl.func_attr({"split_aiv": False})
        _ = pl.tile.add(qk, qk)

    with pytest.raises(ParserSyntaxError, match="split_aiv"):
        function_trace(kernel, **_INCORE)


def test_duplicate_key_against_decorator_attrs_is_loud():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        pl.func_attr({"split_aiv": False})
        _ = pl.tile.add(qk, qk)

    with pytest.raises(ParserSyntaxError, match="split_aiv"):
        function_trace(kernel, attrs={"split_aiv": True}, **_INCORE)


def test_directive_after_a_statement_is_rejected():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        _ = pl.tile.add(qk, qk)
        pl.func_attr({"split_aiv": True})

    with pytest.raises(ParserSyntaxError, match="must appear before every other statement"):
        function_trace(kernel, **_INCORE)


def test_directive_after_a_host_statement_is_rejected():
    """The rule is the host's own first-statement notion — a host-only
    binding is still a statement (the parser rejects the same source at the
    AST level)."""

    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        count = 4  # noqa: F841 - a host binding is a statement, used or not
        pl.func_attr({"split_aiv": True})

    with pytest.raises(ParserSyntaxError, match="must appear before every other statement"):
        function_trace(kernel, **_INCORE)


def test_directive_after_a_host_if_is_rejected():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        if True:
            pass
        pl.func_attr({"split_aiv": True})

    with pytest.raises(ParserSyntaxError, match="must appear before every other statement"):
        function_trace(kernel, **_INCORE)


def test_decorator_only_keys_are_rejected():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        pl.func_attr({"auto_scope": False})
        _ = pl.tile.add(qk, qk)

    with pytest.raises(ParserSyntaxError, match="cannot be set with pl.func_attr"):
        function_trace(kernel, **_INCORE)


def test_split_requires_the_enum_spelling():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        pl.func_attr({"split": 1})
        _ = pl.tile.add(qk, qk)

    with pytest.raises(ParserSyntaxError, match="expected pl.SplitMode.X"):
        function_trace(kernel, **_INCORE)


def test_split_mode_none_is_omitted():
    """``SplitMode.NONE`` is a non-canonical spelling of "no split" that the
    printer filters — storing it would break the print -> parse round trip."""

    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        pl.func_attr({"split": pl.SplitMode.NONE})
        _ = pl.tile.add(qk, qk)

    assert "split" not in dict(function_trace(kernel, **_INCORE).attrs)


def test_parameter_reference_value_binds_the_param_var():
    def kernel(
        qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec],
        w: pl.Tensor[[128, 128], pl.FP32],
    ):
        pl.func_attr({"stationary": w})
        _ = pl.tile.add(qk, qk)

    attrs = dict(function_trace(kernel, **_INCORE).attrs)
    assert isinstance(attrs["stationary"], ir.Var)
    assert attrs["stationary"].name_hint == "w"


def test_both_frontends_agree_on_the_collected_attrs():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]):
        pl.func_attr({"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
        _ = pl.tile.add(qk, qk)

    traced = function_trace(kernel, **_INCORE)
    parsed = function_parser(kernel, type=ir.FunctionType.InCore)
    ir.assert_structural_equal(traced, parsed, enable_auto_mapping=True)
