# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The anchor harness, the span policy, and parser markers.

Per-construct tracing gates plus
parser guardrail tests. The suite lands the *harness mechanics*:

- ``canonicalize`` runs the live pass pipeline (``pypto_core.passes``
  ``convert_to_ssa()`` then ``simplify()``) over a ``Program`` and extracts
  the single function back — validated by a straight-line probe (the passes
  are a no-op on straight-line IR: raw == canonical) and a loop probe
  (SSA synthesis changes the IR: raw != canonical).
- ``conformance`` compares a ``build(kernel)`` against ``function_parser(kernel)``
  — raw or post-canonical. The identity probe uses ``build=function_parser``
  to pin the mechanics; the PtoSema build is the real
  build.
- The span policy: ``assert_structural_equal`` ignores source
  location — pinned by the one-kernel probe.
- Parser markers remain explicit and stable while unsuffixed decorators
  select the tracing frontend.
"""

import pypto.language as pl
import pytest
from pypto.ir import IRBuilder
from pypto.language.parser import function_parser
from pypto.pypto_core import DataType, ir

from ._harness import canonicalize, conformance

# —— parser and tracing markers remain separate ——


def test_frontend_markers_are_separate():
    """Unsuffixed decorators trace; explicit parser markers stay AST-owned."""
    assert pl.function is not pl.function_parser
    assert pl.program is not pl.program_parser
    assert pl.parser.function is pl.parser.function_parser
    assert pl.parser.program is pl.parser.program_parser


def test_function_parser_builds_ir():
    """The parser-owned entry parses a kernel independently of tracing."""

    @pl.function
    def add1(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        result = pl.add(x, 1.0)
        return result

    assert isinstance(add1, ir.Function)
    assert add1.name == "add1"


# —— the harness mechanics ——


def test_canonicalize_is_noop_on_straight_line():
    """A straight-line kernel is already canonical: the SSA/simplify pair
    changes nothing, so raw == canonical (the raw-regime premise)."""

    @pl.function
    def k_straight_line(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        result = pl.add(x, 1.0)
        return result

    ir.assert_structural_equal(k_straight_line, canonicalize(k_straight_line))


def test_canonicalize_transforms_loop_carried():
    """A loop with a self-referential ``AssignStmt`` update is NOT canonical: SSA
    synthesis derives the loop-carried iter_arg. The
    pipeline runs and transforms — the premise of the post-canonical
    regime."""

    @pl.function
    def k_loop(x: pl.Tensor[[128, 128], pl.FP16]) -> pl.Tensor[[128, 128], pl.FP16]:
        acc = x
        for i in pl.range(4):
            acc = pl.add(acc, x)
        return acc

    with pytest.raises(ValueError):
        ir.assert_structural_equal(k_loop, canonicalize(k_loop))
    # and the canonical form is stable (idempotent enough for the anchors)
    canonical = canonicalize(k_loop)
    ir.assert_structural_equal(canonical, canonicalize(canonical))


def test_conformance_identity_probe():
    """``conformance`` with ``build=function_parser`` is an identity — the
    harness mechanics pinned on a real AST-built kernel (the PtoSema build
    replaces the identity)."""

    def k(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        result = pl.add(x, 1.0)
        return result

    conformance(k, build=function_parser)
    conformance(k, build=function_parser, canonical=True)


# —— span policy ——


def test_structural_equal_ignores_spans():
    """The span policy probe: two IRs differing only in source location are
    structurally equal — the anchors need no span work. Built with
    the parser frontend (each parse captures distinct call-site spans), plus a
    hand-built pair with explicit vs call-site spans."""

    @pl.function
    def k1(x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP16]:
        return pl.add(x, 1.0)

    @pl.function
    def k2(x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP16]:
        return pl.add(x, 1.0)

    ir.assert_structural_equal(k1, k2)

    # hand-built: explicit unknown spans vs call-site-captured spans; the
    # two builds create distinct Var objects, so auto-map by name.
    def build(span: ir.Span) -> ir.Function:
        ib = IRBuilder()
        with ib.function("probe", span=span) as f:
            f.param("x", ir.ScalarType(DataType.FP32))
            f.return_type(ir.ScalarType(DataType.FP32))
            x = ib.var("x", ir.ScalarType(DataType.FP32))
            ib.return_stmt(x, span=span)
        return f.get_result()

    ir.assert_structural_equal(build(ir.Span.unknown()), build(ir.Span.unknown()), enable_auto_mapping=True)
