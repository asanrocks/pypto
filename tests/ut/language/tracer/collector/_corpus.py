# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The behavioral corpus: every shipped construct lands an
end-to-end kernel here, run through the trace driver against its native twin.

The kernels live in this module (not in the test files) so the relocated-
copy portability litmus can drive the *full* corpus from a copied
package under a different name — the module imports nothing from the
collector or pypto. ``CASES`` is the (name, kernel, arg-tuples) table both
the suite (``test_corpus.py``) and the litmus
(``test_portability.py``) iterate.
"""

from collections.abc import Callable
from typing import Any


class _Cm:
    """The with-construct corpus context manager: a plain host cm."""

    def __init__(self, log: list[Any]):
        self.log = log

    def __enter__(self) -> int:
        self.log.append("enter")
        return 10

    def __exit__(self, *exc: Any) -> bool:
        self.log.append("exit")
        return False


def k_straight_line(a: int, b: int) -> tuple[int, int]:
    x = a + b
    y = x * 2
    z = (x, y)
    return z


def k_control_flow(n: int) -> int:
    total = 0
    for i in range(n):
        if i % 2 == 0:
            total = total + i
        else:
            total = total - i
    j = 0
    while j < 3:
        total = total + 1
        j = j + 1
    return total


def k_short_circuit(a: int, b: int, c: int) -> str:
    if a and b or not c:
        return "yes"
    return "no"


def k_with_loop_exit(cm: _Cm, n: int) -> int:
    acc = 0
    with cm as base:
        for i in range(n):
            if i == 2:
                break
            acc = acc + i
        acc = acc + base
    return acc


def k_walrus_unpacking(a: int) -> tuple[int, int, int, int]:
    b, c = (a, a * 2)
    n = (m := b + c)
    return n, m, b, c


def k_nested_scope(data: list[int], scale: int) -> int:
    def helper(v: int) -> int:
        return v * scale

    f = lambda y: y + 1  # noqa: E731 - the corpus kernel is the subject
    xs = [helper(x) for x in data if x > 0]
    return sum(xs) + f(scale)


def k_match(tag: object, base: int) -> int:
    match tag:
        case 1:
            return base + 1
        case [a, *rest] if base > 0:
            return a + len(rest) + base
        case {"key": val}:
            return val + base
        case _:
            return -1


def k_annotated_class(v: int) -> tuple[int, dict[str, object]]:
    """class-body annotations referencing an instrumented local:
    eager evaluation must match native — the synthesized module may not
    inherit the collector's own future-annotations flag, and the annotation
    text must route through rt.vars."""

    T = int

    class C:
        x: T = v  # type: ignore[reportInvalidTypeForm] - runtime annotation, not a type
        y: T  # type: ignore[reportInvalidTypeForm]

    return C.x, C.__annotations__


def k_annotated_nested_def(v: int) -> tuple[int, dict[str, object]]:
    """nested-def param/return annotations referencing an instrumented local:
    they evaluate at def time in the enclosing scope."""

    T = int

    def h(x: T) -> T:  # type: ignore[reportInvalidTypeForm] - runtime annotation, not a type
        return x + 1

    return h(v), h.__annotations__


def k_iife_lambda(x: int) -> tuple[int, int]:
    """a directly-invoked lambda: the call parens must
    survive the rewrite — ``(lambda y: y + 1)(x)`` must not become
    ``lambda y: y + 1(x)``."""

    return (lambda y: y + 1)(x), (lambda y=x: y * 2)()  # noqa: PLC3002 - the IIFE is the subject


def _sum2(*nums: int, **kw: int) -> int:
    """a corpus helper for ``k_starred``: a plain ``*args``/``**kw``
    receiver (imports nothing from the collector)."""
    return sum(nums) + sum(kw.values())


def k_grouped(a: int, b: int) -> tuple[int, int, int]:
    """explicit ``( ... )`` grouping as operands: the rewriter must
    preserve the parens or precedence breaks — ``(a + b) * 2`` would
    re-parse as ``a + (b * 2)``."""
    return (a + b) * 2, -(a + b), (a if a > b else b) + 1


def k_starred(a: int, b: int) -> int:
    """starred call args: ``f(*args)`` / ``f(**kw)`` must keep the
    unpacking operator through the rewrite (a fresh Arg dropped it)."""
    args = (a, b)
    kw = {"y": b}
    return _sum2(*args) + _sum2(**kw)


# (name, kernel, sample arg-tuples) — the corpus cases both the suite and
# the relocated-copy litmus drive.
CASES: list[tuple[str, Callable[..., Any], list[tuple[Any, ...]]]] = [
    ("straight_line", k_straight_line, [(2, 3)]),
    ("control_flow", k_control_flow, [(0,), (1,), (4,), (7,)]),
    ("short_circuit", k_short_circuit, [(1, 1, 1), (1, 0, 1), (0, 1, 0), (0, 0, 0)]),
    ("with_loop_exit", k_with_loop_exit, [(_Cm([]), 5)]),
    ("walrus_unpacking", k_walrus_unpacking, [(3,)]),
    ("nested_scope", k_nested_scope, [([1, -2, 3], 2)]),
    ("match", k_match, [(1, 2), ([3, 4, 5], 2), ({"key": 7}, 1), ("other", 0)]),
    ("annotated_class", k_annotated_class, [(5,), (0,)]),
    ("annotated_nested_def", k_annotated_nested_def, [(5,), (0,)]),
    ("iife_lambda", k_iife_lambda, [(5,), (0,)]),
    ("grouped", k_grouped, [(1, 2), (5, 3), (3, 8)]),
    ("starred", k_starred, [(1, 2), (4, 5)]),
]
