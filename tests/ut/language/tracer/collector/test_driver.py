# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Behavioral equivalence through the trace driver: the
instrumented kernel returns the same value and the same side effects as
native execution."""

import sys
from collections.abc import Callable

import collector
import pytest
from collector.rewriter import emit_kernel


def test_straight_line_equivalence():
    """params, locals, a host call as an expression statement, and a return — the hello-world
    slice."""
    acc = []

    def kernel(a, b):
        x = a + b
        acc.append(x)
        return x

    assert collector.instrument(kernel)(2, 3) == kernel(2, 3)


def test_augmented_assign_equivalence():
    def kernel(acc):
        acc += 1
        acc *= 2
        acc -= 1
        return acc

    assert collector.instrument(kernel)(5) == kernel(5)


def test_tuple_unpacking_equivalence():
    def kernel(pair):
        a, b = pair
        (c, d), e = ((a, b), a)
        f, *rest = [1, 2, 3, 4]
        return (a, b, c, d, e, f, rest)

    assert collector.instrument(kernel)((7, 8)) == kernel((7, 8))


def test_walrus_equivalence():
    def kernel(a):
        n = (m := a + 1)
        if (k := m * 2) > 0:  # noqa: F841 - walrus side effect is the point
            pass
        return n, k

    assert collector.instrument(kernel)(5) == kernel(5)


def test_freevar_closure_equivalence():
    scale = 3

    def kernel(x):
        return x * scale

    assert collector.instrument(kernel)(5) == kernel(5)


def test_freevar_closure_live_cell():
    """closure cells stay live — mutations by the enclosing scope after
    instrumentation are visible (freevars pass through natively)."""
    state = {"n": 1}

    def kernel(x):
        return x + state["n"]

    k = collector.instrument(kernel)
    assert k(1) == 2
    state["n"] = 10
    assert k(1) == 11


def test_bare_host_call_side_effect_equivalence():
    def kernel(acc, out):
        out.append(acc)
        acc = acc + 1
        out.append(acc)
        return acc

    side_native, side_instr = [], []
    native = kernel(1, side_native)
    instr = collector.instrument(kernel)(1, side_instr)
    assert native == instr
    assert side_native == side_instr


def test_attribute_access_equivalence():
    class Box:
        def __init__(self, v):
            self.v = v

    def kernel(box, n):
        box.v = n
        return box.v

    b1, b2 = Box(0), Box(0)
    assert collector.instrument(kernel)(b1, 3) == kernel(b2, 3) == 3
    assert b1.v == b2.v == 3


def test_nested_def_lambda_comprehension_equivalence():
    def kernel(x, data):
        def helper(v):
            return v * 2

        f = lambda y: y + 1  # noqa: E731 - the test kernel is the subject
        xs = [f(i) for i in data]
        ys = {v: v * v for v in data}
        return helper(x) + f(x) + sum(xs) + ys[2]

    assert collector.instrument(kernel)(5, [1, 2, 3]) == kernel(5, [1, 2, 3])


def test_nested_def_reads_routed_local():
    """a nested def reading an instrumented (non-param) local resolves through
    rt.vars (the general read row is type-blind)."""

    def kernel(data):
        factor = 2

        def helper(v):
            return v * factor

        return helper(data)

    assert collector.instrument(kernel)(21) == kernel(21)


def test_docstring_preserved():
    def kernel(a):
        """kernel docs"""
        return a

    k = collector.instrument(kernel)
    assert k.__doc__ == kernel.__doc__ == "kernel docs"


def test_param_named_rt_equivalence():
    def kernel(rt):
        x = rt + 1
        return x

    assert collector.instrument(kernel)(1) == kernel(1)


def test_signature_shapes_equivalence():
    """decorator call shapes: positional, keyword, defaults, kwonly,
    *args/**kwargs."""

    def kernel(a, b=2, *args, c=3, **kw):
        return a + b + c + len(args) + len(kw)

    k = collector.instrument(kernel)
    assert k(1) == kernel(1)
    assert k(1, 5) == kernel(1, 5)
    assert k(1, 2, 9, 10, c=7, z=1) == kernel(1, 2, 9, 10, c=7, z=1)


def test_repeated_calls_get_fresh_interpreter():
    """each call runs a fresh Interpreter — no cross-call state leaks."""

    def kernel(a):
        x = 0
        x = x + a
        return x

    k = collector.instrument(kernel)
    assert k(1) == 1
    assert k(2) == 2
    assert k(1) == 1


def test_unbound_local_read_raises():
    """the host's UnboundLocalError discipline."""

    def kernel(a):
        return a + late  # noqa: F821 - intentionally unbound at read  # type: ignore[reportUndefinedVariable]
        late = 1  # noqa: F841 - assigned after use: instrumented, unbound at read

    with pytest.raises(UnboundLocalError):
        kernel(1)
    with pytest.raises(UnboundLocalError):
        collector.instrument(kernel)(1)


def test_exception_propagates():
    class Boom(Exception):
        pass

    def kernel(a):
        x = a
        raise Boom(f"boom {x}")

    with pytest.raises(Boom):
        collector.instrument(kernel)(1)


def test_decorator_form():
    @collector.instrument
    def kernel(a, b):
        return a * b

    assert kernel(3, 4) == 12


def test_one_line_body():
    def kernel(a):
        x = a + 1
        return x

    assert collector.instrument(kernel)(1) == kernel(1)


# ------------------------------------------------------------------------


def test_if_else_equivalence():
    def kernel(c, x):
        if c:
            x = x + 1
        else:
            x = x * 2
        return x

    assert collector.instrument(kernel)(True, 1) == kernel(True, 1)
    assert collector.instrument(kernel)(False, 1) == kernel(False, 1)


def test_if_elision_via_side_effects():
    """the non-taken branch must not fire — elision verified via host side
    effects."""

    def kernel(c, out):
        if c:
            out.append("then")
        else:
            out.append("else")
        return out

    then_out = []
    collector.instrument(kernel)(True, then_out)
    assert then_out == ["then"]
    else_out = []
    collector.instrument(kernel)(False, else_out)
    assert else_out == ["else"]


def test_one_branch_if():
    """one-branch ``if``: else-passthrough (the else-guard is not
    emitted)."""

    def kernel(c, x):
        if c:
            x = x + 1
        return x

    assert collector.instrument(kernel)(True, 1) == kernel(True, 1)
    assert collector.instrument(kernel)(False, 1) == kernel(False, 1)


def test_elif_chain_equivalence():
    def kernel(n):
        if n < 0:
            tag = "neg"
        elif n == 0:
            tag = "zero"
        elif n < 10:
            tag = "small"
        else:
            tag = "big"
        return tag

    for n in (-5, 0, 5, 50):
        assert collector.instrument(kernel)(n) == kernel(n)


def test_predicate_call_evaluated_once():
    calls = []

    def kernel(c, x):
        if probe(c):
            x = x + 1
        else:
            x = x * 2
        return x

    def probe(c):
        calls.append(c)
        return c

    assert collector.instrument(kernel)(True, 1) == 2
    assert calls == [True]  # exactly once


def test_nested_ifs_equivalence():
    def kernel(a, b):
        if a:
            if b:
                r = "a&b"
            else:
                r = "a!b"
        else:  # noqa: PLR5501 - nested ifs are the subject under test
            if b:
                r = "!a&b"
            else:
                r = "!a!b"
        return r

    for a in (True, False):
        for b in (True, False):
            assert collector.instrument(kernel)(a, b) == kernel(a, b)


def test_exception_inside_branch_propagates():
    """exceptions inside a branch unwind cleanly — the exception propagates
    like native; the per-call interpreter is discarded (no leaked frames)."""

    class Boom(Exception):
        pass

    def kernel(c, x):
        if c:
            x = x + 1
            raise Boom("in-then")
        else:
            x = x * 2
        return x

    with pytest.raises(Boom):
        kernel(True, 1)
    with pytest.raises(Boom):
        collector.instrument(kernel)(True, 1)
    assert collector.instrument(kernel)(False, 1) == kernel(False, 1)


def test_if_in_return_position():
    def kernel(a, b):
        if a > b:
            return a
        return b

    assert collector.instrument(kernel)(3, 5) == kernel(3, 5)
    assert collector.instrument(kernel)(9, 5) == kernel(9, 5)


# ------------------------------------------------------------------------


def test_for_unrolls_n_times():
    """``for`` over a host iterable unrolls N× — count verified via a
    side-effect counter."""
    counts_instr, counts_native = [], []

    def kernel(it, counts):
        for x in it:
            counts.append(x)
        return counts

    assert collector.instrument(kernel)([1, 2, 3], counts_instr) == kernel([1, 2, 3], counts_native)
    assert counts_instr == counts_native == [1, 2, 3]


def test_for_iterable_evaluated_once():
    evals = []

    def kernel(n):
        total = 0
        for x in probe(n):
            total = total + x
        return total

    def probe(n):
        evals.append(n)
        return range(n)

    assert collector.instrument(kernel)(4) == 6
    assert evals == [4]  # exactly once


def test_for_loop_var_after_loop():
    def kernel(it):
        for x in it:
            y = x * 2
        return x, y

    assert collector.instrument(kernel)([1, 2, 3]) == kernel([1, 2, 3])
    # empty iterable: the loop var is never bound — both raise UnboundLocalError
    with pytest.raises(UnboundLocalError):
        kernel([])
    with pytest.raises(UnboundLocalError):
        collector.instrument(kernel)([])


def test_while_terminates_predicate_per_iteration():
    """``while`` terminates; its predicate is evaluated exactly as host
    (once per iteration, no double-eval)."""
    evals = []

    def kernel(n):
        acc = 0
        while probe(n) > 0:
            n = n - 1
            acc = acc + 1
        return acc

    def probe(n):
        evals.append(n)
        return n

    assert collector.instrument(kernel)(3) == 3
    assert evals == [3, 2, 1, 0]  # one probe per head evaluation, incl. the final False


def test_while_long_loop_stays_balanced():
    """A long ``while`` matches native and keeps the active-construct stack O(1):
    the source rewriter hoists the thunk to a per-site temp
    bound once before the loop, so the interpreter's object-identity dedup
    pushes the sentinel on the first sight of the thunk only."""

    def kernel(n):
        acc = 0
        i = 0
        while i < n:
            acc = acc + i
            i = i + 1
        return acc

    n = 10_000
    assert collector.instrument(kernel)(n) == kernel(n)


def test_while_call_predicate():
    def kernel(n, out):
        while n > 0:
            out.append(n)
            n = n - 1
        return out

    assert collector.instrument(kernel)(3, []) == kernel(3, [])


def test_nested_loops_equivalence():
    def kernel(n, m):
        total = 0
        for i in range(n):
            if i % 2 == 0:
                total = total + i
            for j in range(m):
                total = total + 1
        return total

    assert collector.instrument(kernel)(4, 3) == kernel(4, 3)

    def kernel2(n):
        acc = 0
        i = 0
        while i < n:
            for j in range(3):
                acc = acc + j
            i = i + 1
        return acc

    assert collector.instrument(kernel2)(4) == kernel2(4)


def test_early_return_inside_loop():
    """early ``return`` inside a loop exits via the host keyword."""

    def kernel(it):
        for x in it:
            if x > 2:
                return x * 10
        return -1

    for it in ([1, 2, 3], [1], []):
        assert collector.instrument(kernel)(it) == kernel(it)


def test_break_continue_innermost_targeting():
    """break/continue target the innermost loop — host-driven by
    construction."""

    def kernel(n):
        acc = 0
        for i in range(n):
            for j in range(3):
                if j == 1:
                    continue
                if i == 1:
                    break
                acc = acc + 1
        return acc

    assert collector.instrument(kernel)(4) == kernel(4)


def test_for_else_equivalence():
    def kernel(it):
        for x in it:
            if x == 5:
                break
        else:
            return "no-5"
        return "found"

    assert collector.instrument(kernel)([1, 2]) == kernel([1, 2])
    assert collector.instrument(kernel)([1, 5]) == kernel([1, 5])
    assert collector.instrument(kernel)([]) == kernel([])


def test_while_else_equivalence():
    def kernel(n):
        while n > 0:
            n = n - 1
            if n == 2:
                break
        else:
            return "exhausted"
        return "broke"

    assert collector.instrument(kernel)(5) == kernel(5)
    assert collector.instrument(kernel)(1) == kernel(1)


# ------------------------------------------------------------------------


def test_and_short_circuit_rhs_not_run():
    """``0 and f()`` — the RHS must not run (side-effect probe)."""
    fired = []

    def kernel(a, probe):
        return a and probe()

    def probe():
        fired.append(1)
        return "rhs"

    assert collector.instrument(kernel)(0, probe) == 0
    assert fired == []
    assert collector.instrument(kernel)(1, probe) == "rhs"
    assert fired == [1]
    # native twin on a fresh probe: same behavior
    fired2 = []

    def probe2():
        fired2.append(1)
        return "rhs"

    assert kernel(0, probe2) == 0 and fired2 == []
    assert kernel(1, probe2) == "rhs" and fired2 == [1]


def test_or_short_circuit_rhs_not_run():
    fired = []

    def kernel(a, probe):
        return a or probe()

    def probe():
        fired.append(1)
        return "rhs"

    assert collector.instrument(kernel)(1, probe) == 1
    assert fired == []
    assert collector.instrument(kernel)(0, probe) == "rhs"
    assert fired == [1]
    fired2 = []

    def probe2():
        fired2.append(1)
        return "rhs"

    assert kernel(1, probe2) == 1 and fired2 == []
    assert kernel(0, probe2) == "rhs" and fired2 == [1]


def test_and_or_return_the_operand():
    """host semantics: ``a and b`` returns ``a`` when falsy else ``b``."""

    def kernel(a, b):
        return (a and b, a or b)

    for a, b in [(0, 5), (3, 0), (0, 0), ("", "x"), ("y", "")]:
        assert collector.instrument(kernel)(a, b) == kernel(a, b)


def test_not_of_call():
    def kernel(a):
        return not check(a)

    def check(a):
        return a > 0

    assert collector.instrument(kernel)(1) == kernel(1)
    assert collector.instrument(kernel)(0) == kernel(0)


def test_chained_and_or_equivalence():
    def kernel(a, b, c):
        return (a and b and c, a or b or c, a or (b and not c))

    cases = [(0, 0, 0), (1, 0, 0), (1, 2, 0), (0, 2, 3)]
    for case in cases:
        assert collector.instrument(kernel)(*case) == kernel(*case)


def test_break_continue_equivalence():
    def kernel(n):
        acc = 0
        for i in range(n):
            if i == 2:
                continue
            if i == 4:
                break
            acc = acc + i
        return acc

    for n in (0, 3, 5, 8):
        assert collector.instrument(kernel)(n) == kernel(n)


def test_break_continue_inside_host_if_in_loop():
    """break/continue inside a host if-in-loop."""

    def kernel(n):
        total = 0
        for i in range(n):
            if i % 2 == 0:
                continue
            if i > 5:
                break
            total = total + i
        return total

    for n in (1, 6, 10):
        assert collector.instrument(kernel)(n) == kernel(n)


def test_while_break_continue():
    def kernel(n):
        acc = 0
        i = 0
        while i < n:
            i = i + 1
            if i == 2:
                continue
            if i == 5:
                break
            acc = acc + i
        return acc, i

    for n in (1, 3, 6):
        assert collector.instrument(kernel)(n) == kernel(n)


# ------------------------------------------------------------------------


def _make_cm(log, name):
    class Cm:
        def __enter__(self):
            log.append(f"enter:{name}")
            return name + "-value"

        def __exit__(self, exc_type, exc, tb):
            log.append(f"exit:{name}:{exc_type is not None}")
            return False

    return Cm()


def test_with_enter_exit_order_equivalence():
    """cm ``__enter__``/``__exit__`` order identical to native."""
    log_instr, log_native = [], []

    def kernel(a, b):
        with a as x:
            y = x
            with b as q:
                z = q
        return y, z

    assert collector.instrument(kernel)(_make_cm(log_instr, "a"), _make_cm(log_instr, "b")) == kernel(
        _make_cm(log_native, "a"), _make_cm(log_native, "b")
    )
    assert log_instr == log_native == ["enter:a", "enter:b", "exit:b:False", "exit:a:False"]


def test_with_cleanup_fires_on_exception():
    """cleanup fires on normal exit **and** on exception — host cm
    exception safety preserved."""
    log_instr, log_native = [], []

    class Boom(Exception):
        pass

    def kernel(cm):
        with cm:
            raise Boom("in-body")

    with pytest.raises(Boom):
        kernel(_make_cm(log_native, "cm"))
    with pytest.raises(Boom):
        collector.instrument(kernel)(_make_cm(log_instr, "cm"))
    assert log_instr == log_native == ["enter:cm", "exit:cm:True"]


def test_with_as_target_routing():
    def kernel(cm, n):
        with cm as value:
            acc = len(value) + n
        return acc

    assert collector.instrument(kernel)(_make_cm([], "c"), 1) == kernel(_make_cm([], "c"), 1)


def test_early_return_inside_nested_blocks():
    """early ``return`` inside nested blocks (if-in-loop-in-with) works via
    the guard."""

    def kernel(cm, it):
        with cm as base:
            for x in it:
                if x > 2:
                    return len(base) + x
        return -1

    for it in ([1, 2, 3], [1], []):
        assert collector.instrument(kernel)(_make_cm([], "c"), it) == kernel(_make_cm([], "c"), it)


def test_variadic_return():
    def kernel(a, b):
        return a, b, a + b

    assert collector.instrument(kernel)(2, 3) == kernel(2, 3)


def test_bare_return_value():
    def kernel(c):
        if c:
            return
        return 42

    assert collector.instrument(kernel)(True) is None
    assert collector.instrument(kernel)(False) == 42


def test_enter_frame_prologue_emitted():
    """the trace driver splices ``rt.enter_frame(a=a, ...)`` after the docstring."""
    out, _ = emit_kernel('def k(a, b=2):\n    """doc"""\n    return a + b\n', "k")
    assert "rt.enter_frame(a=a, b=b)" in out
    assert out.index('"""doc"""') < out.index("rt.enter_frame")


def test_kernel_annotations_surface():
    """the instrumented kernel exposes the kernel's annotations:
    ``types.FunctionType`` drops ``co_annotations``, so they are
    copied explicitly onto the synthesized function."""

    def kernel(x: int, *, y: str = "a") -> float:
        return float(x)

    k = collector.instrument(kernel)
    assert k.__annotations__ == kernel.__annotations__
    assert k.__annotations__["x"] is int
    assert k.__annotations__["return"] is float


def test_annotation_closure_names_do_not_break_freevar_alignment():
    """a kernel whose annotation references an enclosing name (e.g.
    ``s: pl.Tile[[64], DT] = ...`` with ``DT`` a closure var) instruments
    correctly. The rewriter drops annotation expressions, so the
    instrumented body loses the annotation-only name; the surviving freevars align
    to the original closure cells **by name**, and the dropped name (never
    read by the body) needs no cell. Positional alignment is NOT guaranteed:
    the original's ordering can put annotation names first (compiled before
    the body reads), so the synth is not necessarily a positional prefix."""
    DT = int  # referenced only by annotations, never by the body
    scale = 3  # a real freevar read by the body

    def kernel(x: DT, n: int) -> int:  # pyright: ignore[reportInvalidTypeForm]  # intentional: a *variable* annotation, dropped by the rewriter
        s: DT = x  # pyright: ignore[reportInvalidTypeForm]  # body annotated-assignment — annotation dropped by the rewriter
        return s * n + scale

    k = collector.instrument(kernel)
    # the annotation-only name is dropped from the synthesized freevar set;
    # the original (with its annotation compiled in) may order it first
    assert k.fn.__code__.co_freevars == ("scale",)
    assert "DT" in kernel.__code__.co_freevars and "DT" not in k.fn.__code__.co_freevars
    # the surfaced annotations still come from the original kernel
    assert k.__annotations__ == kernel.__annotations__
    assert k.__annotations__["x"] is int
    # behavior is unaffected — the surviving freevar reads the right cell
    assert k(2, 4) == kernel(2, 4) == 11


def test_annotated_class_equivalence():
    """class-body annotations referencing an instrumented local evaluate eagerly,
    matching native: the synthesized module must not
    inherit CO_FUTURE_ANNOTATIONS from the collector's own future import,
    and the annotation text must route through ``rt.vars``."""

    def kernel(v):
        T = int

        class C:
            x: T = v  # type: ignore[reportInvalidTypeForm] - runtime annotation, not a type
            y: T  # type: ignore[reportInvalidTypeForm]

        return C.x, C.__annotations__

    assert collector.instrument(kernel)(5) == kernel(5)


def test_annotated_nested_def_equivalence():
    """nested-def param/return annotations referencing an instrumented local
    evaluate at def time in the enclosing scope."""

    def kernel(v):
        T = int

        def h(x: T) -> T:  # type: ignore[reportInvalidTypeForm] - runtime annotation, not a type
            return x + 1

        return h(v), h.__annotations__

    assert collector.instrument(kernel)(5) == kernel(5)


def test_future_annotations_module_keeps_strings(tmp_path):
    """a kernel from a ``from __future__ import annotations`` module keeps
    stringified annotations — the trace driver mirrors the kernel's own future
    flags instead of unconditionally evaluating them."""

    mod_file = tmp_path / "futmod.py"
    mod_file.write_text(
        "from __future__ import annotations\n"
        "\n"
        "def k(v):\n"
        "    T = int\n"
        "    class C:\n"
        "        x: T = v\n"
        "    return C.__annotations__\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import importlib  # noqa: PLC0415 - function-local: dynamic module name

        mod = importlib.import_module("futmod")
        assert collector.instrument(mod.k)(5) == mod.k(5) == {"x": "T"}
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("futmod", None)


def test_iife_lambda_equivalence():
    """a directly-invoked lambda keeps its call parens through the rewrite
    — ``(lambda y: y + 1)(x)`` must not become
    ``lambda y: y + 1(x)``."""

    def kernel(x):
        return (lambda y: y + 1)(x), (lambda y=x: y * 2)()  # noqa: PLC3002 - the IIFE is the subject

    assert collector.instrument(kernel)(5) == kernel(5)


def test_closure_escape_raises_clear_error():
    """a kernel returning a nested helper over instrumented locals cannot outlive
    the invocation: reading through the closed interpreter
    fails with a clear error, not an IndexError."""

    def kernel():
        x = 1

        def helper():
            return x

        return helper

    f = collector.instrument(kernel)()
    with pytest.raises(RuntimeError, match="no active frame"):
        f()


def test_repeated_calls_with_break_and_early_return():
    """sentinel leaks from a host ``break``/early ``return`` inside loops are
    benign by construction: the exit_construct-after-loop is
    skipped, but a fresh interpreter per call means no cross-call pollution and
    correct results every time."""

    def kernel(n):
        acc = 0
        for i in range(n):
            if i == 3:
                break
            while i > 0:
                acc = acc + i
                i = i - 1
            if i == 2:
                return -100
        return acc

    for _ in range(5):
        for n in (0, 2, 5, 10):
            assert collector.instrument(kernel)(n) == kernel(n)


# ------------------ defaults / decorators / match review fixes ------------------


def test_lambda_default_equivalence():
    """a lambda default referencing an instrumented-assigned local
    routes through rt.vars — pre-fix this raised
    NameError in the synthesized function."""

    def kernel(x):
        n = x * 2
        f = lambda y=n: y  # noqa: E731 - the test kernel is the subject
        return f()

    assert collector.instrument(kernel)(5) == kernel(5)


def test_nested_def_default_equivalence():
    def kernel(x):
        n = x * 2

        def helper(v=n):
            return v

        return helper()

    assert collector.instrument(kernel)(5) == kernel(5)


def test_nested_def_decorator_equivalence():
    """a decorator referencing an instrumented-assigned local routes
    through rt.vars."""

    def kernel(x):
        def make_deco() -> Callable[[Callable[[], object]], Callable[[], object]]:
            return lambda fn: fn

        deco = make_deco()

        @deco
        def helper():
            return x

        return helper()

    assert collector.instrument(kernel)(7) == kernel(7)


def test_class_decorator_and_base_equivalence():
    def kernel(x):
        def make_base(v):
            class B:
                pass

            return B

        base = make_base(x)

        def deco(cls):
            return cls

        @deco
        class C(base):
            def m(self):
                return x

        return C().m()

    assert collector.instrument(kernel)(9) == kernel(9)


def test_match_equivalence():
    """``match``: subject, pattern value-expressions, guards, and case
    bodies behave like native; pattern binders sync into the instrumented
    locals."""

    class Point:
        def __init__(self, x, y):
            self.x, self.y = x, y

        __match_args__ = ("x", "y")

    def kernel(obj, scale):
        base = scale * 2
        match obj:
            case Point(px, py) if px > 0:
                return px + py + base
            case Point(x=px, y=py):  # class kwd patterns bind too
                return px * py + base
            case [a, *rest]:
                return a + len(rest) + base
            case (u, v) | (v, u):
                return u + v + base
            case {"key": val, **more}:  # mapping rest binds
                return val + len(more) + base
            case [s] as whole:  # as-pattern: the sub-pattern and the alias both bind
                return s + len(whole) + base
            case _:
                return -1

    for args in [
        (Point(1, 2), 3),
        (Point(-1, 2), 3),
        ([4, 5, 6], 3),
        ((1, 2), 3),
        ({"key": 5, "j": 1}, 3),
        ([9], 3),
        ("z", 3),
    ]:
        assert collector.instrument(kernel)(*args) == kernel(*args)


def test_match_binder_leaks_past_statement():
    """pattern binders are kernel-scope names (PEP 634): they leak past the
    match statement, so reads after it route through the sync."""

    def kernel(tag):
        base = 10
        match tag:
            case [a] if a > 0:
                r = a + base
            case _:
                r = -1
        return r, a

    assert collector.instrument(kernel)([3]) == kernel([3])
    assert collector.instrument(kernel)([0]) == kernel([0])


def test_match_inside_nested_def():
    def kernel(tag, scale):
        def helper(t):
            base = scale * 2
            match t:
                case 1:
                    return base
                case [a]:
                    return a + base
                case _:
                    return -1

        return helper(tag)

    assert collector.instrument(kernel)(1, 4) == kernel(1, 4)
    assert collector.instrument(kernel)([7], 4) == kernel([7], 4)


def test_cached_globals_not_mutated():
    """each call runs a one-shot function over a copied globals — the
    cached fn's globals never receive the injected interpreter
    (no cross-call/thread race on the shared dict)."""

    def kernel(a):
        x = 0
        x = x + a
        return x

    k = collector.instrument(kernel)
    assert k(1) == 1
    assert k(2) == 2
    assert k.prefix not in k.fn.__globals__  # the interpreter never lands on the cache


# ----------------- parens / unpacking / globals review fixes -------------------


def test_grouped_expressions_equivalence():
    """explicit ``( ... )`` grouping survives the rewrite: the
    rewriter rebuilds expression nodes with ``with_changes``, preserving
    lpar/rpar — pre-fix ``(a + b) * 2`` re-parsed as ``a + (b * 2)``,
    ``-(a + b)`` as ``(-a) + b``, ``(tern) + 1`` glued ``+ 1`` onto the
    else branch, and ``(a + [9])[0]`` glued the subscript onto ``[9]``."""

    def kernel(a, b):
        return (a + b) * 2, -(a + b), (a if a > b else b) + 1

    assert collector.instrument(kernel)(1, 2) == kernel(1, 2)
    assert collector.instrument(kernel)(5, 3) == kernel(5, 3)

    def k_sub(data):
        return (data + [9])[0]

    assert collector.instrument(k_sub)([1, 2, 3]) == k_sub([1, 2, 3])

    def k_cmp(a, b, c):
        return (a < b) == c  # must not become the chained `a < b == c`

    for args in [(1, 2, 0), (3, 2, 0), (3, 2, 1)]:
        assert collector.instrument(k_cmp)(*args) == k_cmp(*args)

    def k_fstr(a, b):
        return f"{(a + b) * 2}"

    assert collector.instrument(k_fstr)(1, 2) == k_fstr(1, 2)


def test_starred_call_args_equivalence():
    """``f(*args)`` / ``f(**kw)`` keep their unpacking operator —
    pre-fix the fresh Arg dropped ``star`` so ``f(*a)`` became ``f(a)``."""

    def _add(*nums, **kw):
        return sum(nums) + sum(kw.values())

    def k_star(fn, a, b):
        args = (a, b)
        return fn(*args)

    assert collector.instrument(k_star)(_add, 1, 2) == k_star(_add, 1, 2)

    def k_kw(fn, a):
        kw = {"y": a}
        return fn(**kw)

    assert collector.instrument(k_kw)(_add, 5) == k_kw(_add, 5)

    def k_both(fn, a, b):
        args = (a,)
        kw = {"y": b}
        return fn(*args, **kw)

    assert collector.instrument(k_both)(_add, 1, 2) == k_both(_add, 1, 2)


def test_multi_item_with_cross_reference():
    """a later ``with`` item reading an earlier as-target sees the synced
    value — pre-fix all syncs landed in the body after every
    ``__enter__``, so ``b(x)`` read the pre-sync ``x`` and raised
    UnboundLocalError."""

    class A:
        def __enter__(self):
            return "a-val"

        def __exit__(self, *e):
            return False

    class B:
        def __init__(self, pre):
            self.pre = pre

        def __enter__(self):
            return "b-val:" + str(self.pre)

        def __exit__(self, *e):
            return False

    def kernel(a, b):
        with a as x, b(x) as y:
            return (x, y)

    assert collector.instrument(kernel)(A(), B) == kernel(A(), B)


def test_multi_item_with_enter_exit_order():
    """nesting the multi-item ``with`` preserves the host enter (l-to-r) /
    exit (r-to-l) order."""
    log = []

    class Cm:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            log.append(f"enter:{self.name}")
            return self.name

        def __exit__(self, *e):
            log.append(f"exit:{self.name}")
            return False

    def kernel(a, b):
        with a as x, b as y:
            return (x, y)

    assert collector.instrument(kernel)(Cm("a"), Cm("b")) == ("a", "b")
    assert log == ["enter:a", "enter:b", "exit:b", "exit:a"]


def test_del_attribute_and_subscript_equivalence():
    """``del a.b`` / ``del a[i]`` route through ``rt.vars`` —
    behavioral equivalence with native."""

    class Box:
        def __init__(self):
            self.flag = 1
            self.keep = 2

    def k_attr(o):
        del o.flag
        return o.keep, hasattr(o, "flag")

    assert collector.instrument(k_attr)(Box()) == k_attr(Box())

    def k_sub(a):
        del a[0]
        return list(a)

    assert collector.instrument(k_sub)([1, 2, 3]) == k_sub([1, 2, 3])


def test_attribute_and_subscript_for_targets_equivalence():
    """``for x.a in it:`` / ``for a[i] in it:`` route."""

    class Box:
        def __init__(self):
            self.n = 0

    def k_attr(obj, it):
        for obj.n in it:
            pass
        return obj.n

    assert collector.instrument(k_attr)(Box(), [7, 8, 9]) == k_attr(Box(), [7, 8, 9])

    def k_sub(arr, it):
        for arr[0] in it:
            pass
        return list(arr)

    assert collector.instrument(k_sub)([0, 0], [1, 2, 3]) == k_sub([0, 0], [1, 2, 3])


def test_instrument_injects_custom_sema():
    """The Sema-injection seam: a custom Frame reaches the Interpreter through
    ``instrument`` (no longer hardcoded NullSema). A staged
    Marker value reaches the Frame's ``bind`` (which records it and returns
    the representation to store); the default NullSema path stays a no-op."""

    class Marker:
        """A stand-in staged value, recognized by ``bind``."""

    class RecordingFrame(collector.NullFrame):
        def __init__(self):
            self.bound: list = []

        def bind(self, interpreter, name, value, span=None):
            if isinstance(value, Marker):
                self.bound.append((name, value))
                return collector.StagedBinding(value)
            return collector.HostBinding()

    class RecordingSema(collector.NullSema):
        def __init__(self):
            self.frames: list[RecordingFrame] = []

        def begin_frame(self):
            frame = RecordingFrame()
            self.frames.append(frame)
            return frame

    def kernel(m):
        x = m  # noqa: F841 - staged bind redirected to the Frame, not read
        return 0  # host return; no read of the staged x

    sema = RecordingSema()
    marker = Marker()
    assert collector.instrument(kernel, sema)(marker) == 0
    assert sema.frames[-1].bound == [("x", marker)]

    # the default (NullSema) path is unchanged — no sema arg:
    assert collector.instrument(kernel)(marker) == 0


def test_sema_factory_supplies_fresh_sema_per_call():
    """A factory yields a fresh Sema per invocation, so a
    mutable Sema is never reused across traces."""

    class Marker:
        pass

    class CountingFrame(collector.NullFrame):
        def bind(self, interpreter, name, value, span=None):
            if isinstance(value, Marker):
                return collector.StagedBinding(value)
            return collector.HostBinding()

    created = []

    class CountingSema(collector.NullSema):
        def __init__(self):
            created.append(self)

        def begin_frame(self):
            return CountingFrame()

    def kernel(m):
        x = m  # noqa: F841
        return 1

    wrapped = collector.instrument(kernel, CountingSema)
    assert wrapped(Marker()) == 1
    assert wrapped(Marker()) == 1
    assert len(created) == 2  # a fresh Sema per call


# ------------------------------------------------------------------ unwind ---
# Native transfers unwind through unwind scope guards: logical-scope cleanup
# runs at each construct's exact lexical boundary, so observable order
# (finally bodies, context-manager exits) matches native Python exactly.


class _EventCM:
    """A context manager recording its enter/exit into a shared list."""

    def __init__(self, events, tag):
        self._events = events
        self._tag = tag

    def __enter__(self):
        self._events.append(f"enter:{self._tag}")
        return self

    def __exit__(self, *exc):
        self._events.append(f"exit:{self._tag}:{exc[0].__name__ if exc[0] else 'ok'}")
        return False


def test_native_break_unwind_order_matches_python():
    """``finally`` bodies inside a ``with`` run before the context manager
    exits — the eager pre-close scheme closed the with's construct (and
    ran its ``__exit__``) before the keyword."""

    def make():
        events = []

        def kernel(flag):
            for i in range(3):
                with _EventCM(events, "cm"):
                    try:
                        if flag:
                            break
                    finally:
                        events.append("finally")
                events.append("after")
            events.append("end")
            return flag

        return events, kernel

    events, kernel = make()
    native = kernel(True)
    assert events == ["enter:cm", "finally", "exit:cm:ok", "end"]

    events, kernel = make()
    assert collector.instrument(kernel)(True) == native
    assert events == ["enter:cm", "finally", "exit:cm:ok", "end"]


def test_native_continue_unwind_order_matches_python():
    def make():
        events = []

        def kernel(n):
            for i in range(3):
                with _EventCM(events, "cm"):
                    try:
                        if i == 1:
                            continue
                    finally:
                        events.append(f"finally:{i}")
                events.append("after")
            events.append("end")
            return n

        return events, kernel

    events, kernel = make()
    kernel(0)
    native_events = list(events)

    events, kernel = make()
    collector.instrument(kernel)(0)
    assert events == native_events
    # the continue's unwind: the finally, then the cm exit — per iteration
    assert events == [
        "enter:cm",
        "finally:0",
        "exit:cm:ok",
        "after",
        "enter:cm",
        "finally:1",
        "exit:cm:ok",  # continue skips "after"
        "enter:cm",
        "finally:2",
        "exit:cm:ok",
        "after",
        "end",
    ]


def test_native_break_nested_unwind_order_matches_python():
    """Nested with/if/with: the break's unwind crosses every construct in
    reverse lexical order — inner cm, inner instrumented if (silent), outer cm,
    outer instrumented if, loop."""

    def make():
        events = []

        def kernel(flag):
            for i in range(3):
                with _EventCM(events, "outer"):
                    if i == 1:
                        with _EventCM(events, "inner"):
                            if flag:
                                break
                    events.append("mid")
                events.append("after")
            events.append("end")
            return flag

        return events, kernel

    events, kernel = make()
    kernel(True)
    native_events = list(events)

    events, kernel = make()
    collector.instrument(kernel)(True)
    assert (
        events
        == native_events
        == [
            "enter:outer",
            "mid",
            "exit:outer:ok",
            "after",
            "enter:outer",
            "enter:inner",
            "exit:inner:ok",
            "exit:outer:ok",
            "end",
        ]
    )


def test_finally_raising_replaces_break_unwind():
    """A ``finally`` that raises replaces the break: the exception
    propagates, the context manager receives the real exception triple
    exactly once, and no unwind scope guard finalizes during the unwind."""

    events = []

    class Boom(Exception):
        pass

    def kernel(flag):
        for i in range(3):
            with _EventCM(events, "cm"):
                try:
                    if flag:
                        break
                finally:
                    if flag:
                        raise Boom("replaced")
            events.append("after")
        events.append("end")
        return flag

    with pytest.raises(Boom, match="replaced"):
        collector.instrument(kernel)(True)
    assert events == ["enter:cm", "exit:cm:Boom"]
    # no unwind scope guard closed a construct during the unwind and no statement after
    # the with ran — the exception replaced the break exactly as natively.


def test_host_return_through_unwind_guard_matches_python():
    """A host-mode return unwinds guarded constructs: every context
    manager exits (in reverse order) before the function returns."""

    def make():
        events = []

        def kernel(flag):
            total = 0
            for i in range(5):
                if i == 2:
                    if flag:
                        with _EventCM(events, "cm"):
                            return total + i
                    break  # makes both ifs and the loop transfer-affected
                total += i
            events.append("fell-through")
            return total

        return events, kernel

    events, kernel = make()
    native = kernel(True)
    assert native == 3  # 0 + 1, then i == 2 returns total + i
    assert events == ["enter:cm", "exit:cm:ok"]

    events, kernel = make()
    assert collector.instrument(kernel)(True) == native
    assert events == ["enter:cm", "exit:cm:ok"]

    events, kernel = make()
    assert collector.instrument(kernel)(False) == 1
    assert events == ["fell-through"]


def test_transfer_in_inner_loop_else_binds_to_outer():
    """A transfer in a loop's `else` suite targets the next enclosing
    loop (the suite runs after that loop completes) — the enclosing
    constructs and the outer loop must receive their unwind scope guards
    (the walker previously counted the else suite inside the loop,
    leaving stale constructs that corrupted routing)."""

    def make(kind, keyword):
        events = []

        def kernel(values):
            total = 0
            for outer in values:
                if values:  # a host truth: the enclosing if is host
                    if kind == "for":
                        for inner in values:
                            total += 1
                        else:  # noqa: PLW0120 — the else-suite transfer is the subject
                            events.append("inner-else")
                            if keyword == "break" or outer == 2:
                                break
                            continue
                    else:
                        i = 0
                        while i < 2:
                            total += 1
                            i += 1
                        else:  # noqa: PLW0120 — the else-suite transfer is the subject
                            events.append("while-else")
                            if keyword == "break" or outer == 2:
                                break
                            continue
                    events.append("after-inner")
                events.append(f"after-outer:{outer}")
            events.append("end")
            return total

        return events, kernel

    for kind in ("for", "while"):
        for keyword in ("break", "continue"):
            events, kernel = make(kind, keyword)
            native = kernel([1, 2, 3])
            native_events = list(events)

            events, kernel = make(kind, keyword)
            assert collector.instrument(kernel)([1, 2, 3]) == native, (kind, keyword)
            assert events == native_events, (kind, keyword)
