# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``to_expr``'s A-axis contract, including the exact-type fast lane.

``to_expr`` is the operand-boundary funnel (called per list element by the
op wrappers), so it carries an exact-type guard for bare literals ahead of
the structural Protocol check — the Protocol ``isinstance`` costs ~3µs per
element and has no negative-result caching. The guard is strictly narrower
than the fallback (``bool`` and int-subclasses still take the Protocol
path), so these tests pin that the fast lane never changes semantics, only
cost.
"""

from pypto.language.op.tensor_ops import _normalize_intlike
from pypto.language.typing import Scalar, Tensor, to_expr
from pypto.language.typing.type_utils import _UnwrapsToExpr
from pypto.pypto_core import DataType, ir


def _span() -> ir.Span:
    return ir.Span.unknown()


def test_bare_literals_pass_through_unchanged():
    # identity (not ==) is the contract: the object comes back untouched
    for x in (7, -1, 2.5, -0.25, 10**20):
        assert to_expr(x) is x


def test_bool_and_int_subclasses_take_the_fallback_path():
    """``cls is int`` excludes bool and subclasses: they must still route
    through the Protocol check and return unchanged, exactly as before
    the fast lane existed."""

    class MyInt(int):
        pass

    assert to_expr(True) is True
    assert isinstance(True, _UnwrapsToExpr) is False  # the fallback resolves the same way
    x = MyInt(3)
    assert to_expr(x) is x


def test_scalar_facade_unwraps_to_its_expr():
    expr = ir.ConstInt(5, DataType.INDEX, _span())
    s = Scalar(expr=expr)
    assert to_expr(s) is expr


def test_tensor_facade_unwraps_to_its_expr():
    expr = ir.Var("x", ir.TensorType([64], DataType.FP32), _span())
    t = Tensor(expr=expr)
    assert to_expr(t) is expr


def test_raw_expr_passes_through():
    expr = ir.Var("v", ir.TensorType([4], DataType.BF16), _span())
    assert to_expr(expr) is expr


def test_fast_lane_matches_the_protocol_path_for_every_kind():
    """The observable contract is path-independent: for each input kind,
    the guard spelling and a Protocol-first spelling agree."""
    expr = ir.Var("s", ir.TensorType([2], DataType.FP32), _span())
    cases = [0, 1, -5, 3, 0.5, -0.25, True, False, Scalar(expr=expr), Tensor(expr=expr), expr]
    for x in cases:
        protocol_first = x.unwrap() if isinstance(x, _UnwrapsToExpr) else x
        assert to_expr(x) is protocol_first


def test_slice_operand_funnel_uses_the_fast_lane():
    """End-to-end: a traced slice with int lists (the corpus's dominant
    spelling) produces the same call expression the slow path would —
    the dsv4 parity A/B relies on this equivalence."""
    assert _normalize_intlike([128, 512, 1, 512]) == [128, 512, 1, 512]
    staged = Scalar(expr=ir.ConstInt(4, DataType.INDEX, _span()))
    out = _normalize_intlike([staged, 0, 0, 0])
    assert out[0] is staged.unwrap()
    assert out[1:] == [0, 0, 0]
