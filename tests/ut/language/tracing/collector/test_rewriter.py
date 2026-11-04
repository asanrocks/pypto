# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Emission-table unit tests: the four rows plus the tricky cases."""

import re
import sys

import pytest
from pypto.language.tracing.collector.rewriter import instrument_function_source


def emit(source: str, name: str = "k") -> str:
    return instrument_function_source(source, name)[0]


def test_assign_row():
    """``x = e`` → ``rt.vars.x = e`` (instrumentation boundary)."""
    out = emit("def k():\n    x = 1\n")
    assert "rt.vars.x = 1" in out


def test_read_row():
    """read of local ``x`` → ``rt.vars.x`` (chain-walks)."""
    out = emit("def k(a):\n    y = a\n")
    assert "rt.vars.y = rt.vars.a" in out


def test_bare_stmt_row():
    """call expression statement → ``rt._emit(expr)``."""
    out = emit("def k():\n    consume()\n")
    assert "rt._emit(consume())" in out


def test_bare_stmt_non_call_passes_through():
    out = emit("def k(x):\n    x + 1\n")
    assert "rt._emit" not in out
    assert "rt.vars.x + 1" in out


def test_return_row():
    """``return e`` → temp + ``rt.return_(e)`` + the ``in_host_mode()`` exit
    guard (the exact guard shape is the host
    ``return`` carries the value so behavioral equivalence holds)."""
    out = emit("def k(a):\n    return a\n")
    assert re.search(r"__pto_ret\d+ = rt\.vars\.a", out)
    assert re.search(r"rt\.return_\(__pto_ret\d+\)", out)
    assert "if rt.in_host_mode():" in out
    assert re.search(r"return __pto_ret\d+", out)


def test_return_row_does_not_infer_staged_tuple():
    """Python tuple syntax is not staged tuple construction; users must
    write ``return pl.Tuple(a, b)`` for a staged tuple value."""
    out = emit("def k(a, b):\n    return a, b\n")
    assert re.search(r"__pto_ret\d+ = \(?rt\.vars\.a, rt\.vars\.b\)?", out)
    assert re.search(r"rt\.return_\(__pto_ret\d+\)", out)


def test_return_row_variadic_and_bare():
    out = emit("def k(a, b):\n    return a, b\n")
    assert re.search(r"__pto_ret\d+ = \(?rt\.vars\.a, rt\.vars\.b\)?", out)
    out = emit("def k():\n    return\n")
    assert "rt.return_()" in out
    assert "return" in out


def test_augmented_assign_row():
    """``x += e`` → ``rt.vars.x += e`` — getattr once, setattr once."""
    out = emit("def k(a):\n    a += 1\n")
    assert "rt.vars.a += 1" in out


def test_annassign_row():
    """``x: T = e`` → ``rt.vars.x = e`` (annotations dropped);
    a bare annotation binds nothing."""
    out = emit("def k():\n    x: int = 1\n")
    assert "rt.vars.x = 1" in out
    assert "int" not in out
    out = emit("def k():\n    x: int\n    return 1\n")
    assert "x: int" not in out


def test_tuple_unpacking_row():
    """tuple targets route element-wise, preserving single-eval. The rewrite
    is lossless: ``with_changes`` carries the source's exact
    parenthesization, so a bare ``b, c = a`` stays bare (it is *not*
    rewritten with added parens)."""
    out = emit("def k(a):\n    b, c = a\n")
    assert "rt.vars.b, rt.vars.c = rt.vars.a" in out
    out = emit("def k(a):\n    (b, c), d = a\n")
    assert "(rt.vars.b, rt.vars.c), rt.vars.d = rt.vars.a" in out
    out = emit("def k(a):\n    b, *rest = a\n")
    assert "rt.vars.b, *rt.vars.rest = rt.vars.a" in out


def test_walrus_row():
    """``(x := e)`` → ``rt.bind("x", e)`` — the expression-level bind
    (must-rewrite); evaluates ``e`` once, returns the value."""
    out = emit("def k(a):\n    return (n := a) + 1\n")
    assert re.search(r"rt\.bind\(['\"]n['\"], rt\.vars\.a\)", out)


def test_attribute_access_passes_through():
    """attribute access on literal names passes through; the
    value part routes when local."""
    out = emit("def k(a):\n    return a.b\n")
    assert "rt.vars.a.b" in out
    out = emit("def k(a):\n    a.b = 1\n")
    assert "rt.vars.a.b = 1" in out


def test_nested_def_passes_through():
    """nested def: the construct stays host-language;
    reads inside route under the def's own scope."""
    out = emit("def k(x):\n    def helper(v):\n        return v * 2\n    return helper(x)\n")
    assert "def helper(v):" in out
    assert "return v * 2" in out  # v is helper-local — stays native
    assert "rt.vars.helper(rt.vars.x)" in out


def test_nested_def_mixed_binding_syncs():
    """a def name that is also instrumented-assigned gets the native-binding
    sync so instrumented reads see it."""
    out = emit("def k():\n    def helper():\n        return 1\n    helper = 2\n    return helper\n")
    assert "rt.vars.helper = helper" in out


def test_lambda_and_comprehension_route_outer_reads():
    out = emit("def k(x, data):\n    f = lambda y: x + y\n    xs = [f(i) for i in data]\n    return xs\n")
    assert "lambda y: rt.vars.x + y" in out
    assert "[rt.vars.f(i) for i in rt.vars.data]" in out


def test_docstring_stays_first():
    out = emit('def k():\n    """doc"""\n    x = 1\n    return x\n')
    assert out.strip().startswith('def k():\n    """doc"""')


def test_comment_event_preserves_simple_statement_comments():
    out = emit("def k(x):\n    # annotate\n    y = x  # assign\n    return y\n")
    event = "rt._leading_comments(['annotate', 'assign'])"
    assert event in out
    assert out.index(event) < out.index("rt.vars.y = rt.vars.x")


def test_comment_events_preserve_compound_statement_comments():
    out = emit(
        "def k(c, x):\n"
        "    # condition\n"
        "    if c:  # positive\n"
        "        y = x\n"
        "    # fallback\n"
        "    else:  # negative\n"
        "        y = -x\n"
    )
    assert "rt._leading_comments(['condition', 'positive'])" in out
    fallback = "rt._leading_comments(['fallback', 'negative'])"
    assert fallback in out
    assert out.index("rt.eval_else_active()") < out.index(fallback)
    assert out.index(fallback) < out.index("rt.vars.y = -rt.vars.x")


def test_docstring_emits_comment_event_after_docstring():
    out = emit('def k():\n    """preserved docstring"""\n    x = 1\n    return x\n')
    docstring = '    """preserved docstring"""'
    event = "rt._leading_comments(['preserved docstring'])"
    assert out.index(docstring) < out.index(event)
    assert out.index(event) < out.index("rt.vars.x = 1")


def test_comment_and_fstring_literal_rt_untouched():
    """AST-based rewriting keeps comment text out of executable code and
    does not confuse it with an f-string expression."""
    out = emit('def k(a):\n    # rt.vars.a = 999\n    s = f"rt.vars.a = {a}"\n    return s\n')
    assert "rt._leading_comments(['rt.vars.a = 999'])" in out
    assert "rt.vars.a = 999\n" not in out
    assert re.search(r'f["\']rt\.vars\.a = \{rt\.vars\.a\}["\']', out)


def test_param_named_rt_prefix_collision():
    """a parameter named ``rt`` must not be clobbered — the event prefix
    moves."""
    out, prefix, _ = instrument_function_source("def k(rt):\n    return rt + 1\n", "k")
    assert prefix != "rt"
    assert f"{prefix}.vars.rt" in out


def test_one_line_suite_splits():
    out = emit("def k(a):\n    x = a; y = x + 1\n    return y\n")
    assert "rt.vars.x = rt.vars.a" in out
    assert "rt.vars.y = rt.vars.x + 1" in out


def test_freevar_read_passes_through():
    """a freevar read is not redirected — it resolves through the synthesized
    function's closure."""
    out = emit("def k():\n    return scale\n")
    assert "rt.vars.scale" not in out
    assert "= scale" in out


def test_if_row():
    """``if`` row: eval_if + then-guard + exit_construct; reads
    inside route."""
    out = emit("def k(c, a):\n    if c:\n        a = a + 1\n    return a\n")
    assert "rt.eval_if(rt.vars.c)" in out
    assert "if rt.eval_then():" in out
    assert "rt.exit_construct()" in out
    assert "rt.vars.a = rt.vars.a + 1" in out


def test_if_else_row():
    out = emit("def k(c):\n    if c:\n        x = 1\n    else:\n        x = 2\n    return x\n")
    assert "rt.eval_else()" in out
    assert "if rt.eval_else_active():" in out
    assert "rt.vars.x = 1" in out
    assert "rt.vars.x = 2" in out


def test_elif_pre_normalization():
    """``elif`` pre-normalizes to an else-guard containing the nested if
    row."""
    out = emit(
        "def k(c):\n"
        "    if c:\n"
        "        x = 1\n"
        "    elif c2:\n"
        "        x = 2\n"
        "    else:\n"
        "        x = 3\n"
        "    return x\n"
    )
    assert out.count("rt.eval_if(") == 2
    assert out.count("rt.exit_construct()") == 2
    assert out.count("rt.eval_else()") == 2


def test_predicate_evaluated_exactly_once():
    """the predicate is a function call evaluated exactly once
    tricky case) — it appears once in the emission."""
    out = emit("def k(a):\n    if f(a):\n        x = 1\n    return x\n")
    assert out.count("rt.eval_if(f(rt.vars.a))") == 1


def test_nested_ifs_balanced():
    out = emit(
        "def k(a, b):\n"
        "    if a:\n"
        "        if b:\n"
        "            x = 1\n"
        "        else:\n"
        "            x = 2\n"
        "    return x\n"
    )
    assert out.count("rt.eval_if(") == 2
    assert out.count("rt.exit_construct()") == 2


# ------------------------------------------------------------------------


def test_for_row():
    """``for`` row: target redirection + begin_for + host for +
    exit_construct after the loop."""
    out = emit("def k(it):\n    for x in it:\n        y = x\n    return y\n")
    assert re.search(r"for __pto_it\d+ in rt\.begin_for\(rt\.vars\.it\):", out)
    assert re.search(r"rt\.vars\.x = __pto_it\d+", out)
    assert "rt.vars.y = rt.vars.x" in out
    assert "rt.exit_construct()" in out


def test_for_iterable_evaluated_once():
    out = emit("def k(it):\n    for x in make(it):\n        pass\n")
    assert out.count("rt.begin_for(make(rt.vars.it))") == 1


def test_while_row():
    """``while`` row: the mode-aware resolver + host while +
    exit_construct after the loop. The thunk is hoisted to a per-site temp
    bound once before the loop (stable identity across iterations — the
    interpreter dedups the Construct push on first sight of it)."""
    out = emit("def k(n):\n    while n > 0:\n        n = n - 1\n")
    assert "__pto_w0 = lambda: rt.vars.n > 0" in out  # hoisted once before the loop
    assert "while rt.eval_bool(__pto_w0):" in out  # the host while re-drives it each iteration
    assert "rt.vars.n = rt.vars.n - 1" in out
    assert "rt.exit_construct()" in out


def test_loop_else_keeps_host_else():
    """``for``/``while``-else keep the host else (runs on normal
    completion); exit_construct lives inside it and carries the ``has_else``
    static fact so the interpreter can reject a staged loop."""
    out = emit("def k(it):\n    for x in it:\n        pass\n    else:\n        y = 1\n")
    assert out.count("rt.exit_construct(else_follows=True)") == 1
    out = emit("def k(n):\n    while n > 0:\n        n = n - 1\n    else:\n        y = 1\n")
    assert out.count("rt.exit_construct(else_follows=True)") == 1


# ------------------------------------------------------------------------


def test_and_row():
    """``a and b`` → ``rt.eval_and(a, λ:b)``: the RHS is a
    single-expression thunk."""
    out = emit("def k(a, b):\n    return a and b\n")
    assert "rt.eval_and(rt.vars.a, lambda: rt.vars.b)" in out


def test_or_row():
    out = emit("def k(a, b):\n    return a or b\n")
    assert "rt.eval_or(rt.vars.a, lambda: rt.vars.b)" in out


def test_not_row():
    out = emit("def k(a):\n    return not a\n")
    assert "rt.eval_not(rt.vars.a)" in out


def test_chained_and_or():
    # chained ``a and b and c`` is left-nested in the tree — the
    # recursion preserves short-circuit either way.
    out = emit("def k(a, b, c):\n    return a and b and c\n")
    assert "rt.eval_and(rt.eval_and(rt.vars.a, lambda: rt.vars.b), lambda: rt.vars.c)" in out
    out = emit("def k(a, b, c):\n    return a or (b and not c)\n")
    assert "rt.eval_or(rt.vars.a, lambda: rt.eval_and(rt.vars.b, lambda: rt.eval_not(rt.vars.c)))" in out


def test_break_continue_rows():
    out = emit(
        "def k(n):\n"
        "    while n > 0:\n"
        "        if n == 2:\n"
        "            break\n"
        "        n = n - 1\n"
        "        continue\n"
    )
    assert "if rt.break_():" in out
    assert "if rt.continue_():" in out
    assert "break" in out
    assert "continue" in out


def test_native_transfer_unwind_guards():
    """A native break/continue skips a plain trailing ``exit_construct()``
    call, so every construct whose body holds a transfer keyword bound to
    an enclosing loop is wrapped in the interpreter's unwind scope guard
    instead —
    and only those constructs (shape churn stays out of transfer-free
    code)."""
    # break inside an if inside a for-else: the if and the loop are both
    # wrapped (the break crosses the if's close and the loop's else-suite
    # close); the guard body itself is the bare keyword.
    out = emit("def k(n):\n    for j in n:\n        if j == 2:\n            break\n    else:\n        pass\n")
    assert re.search(r"if rt\.break_\(\):\s*\n\s*break", out)
    assert out.count("with rt.unwind_guard():") == 2  # the if + the loop
    # the loop-else pre-close stays inside the unwind guard; no trailing close
    assert out.count("rt.exit_construct(else_follows=True)") == 1
    assert out.count("rt.exit_construct()") == 0

    # break inside an if in a loop WITHOUT else: only the if is wrapped —
    # continue never skips the loop's own close, and this loop has no
    # break, so the loop keeps its trailing close.
    out = emit("def k(n):\n    for j in n:\n        if j == 2:\n            continue\n")
    assert re.search(r"if rt\.continue_\(\):\s*\n\s*continue", out)
    assert out.count("with rt.unwind_guard():") == 1  # the if only
    assert out.count("rt.exit_construct()") == 1  # the loop's trailing close

    # a bare break (no enclosing construct) in a loop without else: no
    # unwind guard around the if (there is none) — the loop alone is wrapped.
    out = emit("def k(n):\n    for j in n:\n        break\n")
    assert out.count("with rt.unwind_guard():") == 1
    assert out.count("rt.exit_construct()") == 0

    # a break inside a nested inner loop is bound to it: the outer loop
    # keeps its trailing close; only the inner loop and its if wrap.
    out = emit(
        "def k(n):\n    for j in n:\n        for i in n:\n            if i == 2:\n                break\n"
    )
    assert out.count("with rt.unwind_guard():") == 2  # the inner loop + its if
    assert out.count("rt.exit_construct()") == 1  # the outer loop's trailing close

    # transfer-free code keeps the baseline shape: no unwind guards at all.
    out = emit("def k(n):\n    for j in n:\n        if j == 2:\n            x = j\n")
    assert "with rt.unwind_guard():" not in out
    assert out.count("rt.exit_construct()") == 2  # the if + the loop

    # a transfer in an inner loop's else suite binds to the OUTER loop:
    # the enclosing if and the outer loop wrap, the inner loop (whose
    # pre-else close ran before the keyword) does not.
    out = emit(
        "def k(n, m):\n"
        "    for outer in n:\n"
        "        if outer == 1:\n"
        "            for inner in m:\n"
        "                pass\n"
        "            else:\n"
        "                break\n"
    )
    assert out.count("with rt.unwind_guard():") == 2  # the if + the outer loop
    assert out.count("rt.exit_construct(else_follows=True)") == 1  # the inner loop's pre-else close
    assert out.count("rt.exit_construct()") == 0


# ------------------------------------------------------------------------


def test_with_row():
    """``with`` row: keep the host with, wrap in the adapter,
    route the as-target through a fresh temp."""
    out = emit("def k(cm):\n    with cm() as y:\n        z = y\n    return z\n")
    assert re.search(
        r"with rt\.on_begin_with\(rt\.vars\.cm\(\), name\s*=\s*'y', "
        r"single_call_body=False\) as __pto_with\d+:",
        out,
    )
    assert re.search(r"rt\.vars\.y = __pto_with\d+", out)
    assert "rt.vars.z = rt.vars.y" in out


def test_with_no_as_target():
    out = emit("def k(cm):\n    with cm():\n        z = 1\n")
    assert "with rt.on_begin_with(rt.vars.cm(), single_call_body=False):" in out
    assert "rt.vars.z = 1" in out


def test_with_multi_item():
    """multi-item ``with a, b:`` wraps each context (the mixed-mode
    rejection is an open item)."""
    out = emit("def k(a, b):\n    with a as x, b as y:\n        z = x + y\n")
    assert out.count("rt.on_begin_with(") == 2
    assert "rt.vars.x" in out
    assert "rt.vars.y" in out


def test_unknown_function_raises():
    from pypto.language.tracing.collector.rewriter import RewriteError  # noqa: PLC0415

    with pytest.raises(RewriteError):
        instrument_function_source("def other():\n    pass\n", "k")


def test_wildcard_import_raises():
    from pypto.language.tracing.collector.rewriter import RewriteError  # noqa: PLC0415

    with pytest.raises(RewriteError, match="wildcard imports"):
        instrument_function_source("def k():\n    from module import *\n", "k")


# ------------------ defaults / decorators / match review fixes ------------------


def test_lambda_default_routes_kernel_local():
    """a lambda default evaluates in the enclosing scope — reads of
    instrumented locals redirect through rt.vars."""
    out = emit("def k(x):\n    n = x * 2\n    f = lambda y=n: y\n    return f()\n")
    assert "lambda y=rt.vars.n: y" in out


def test_nested_def_default_routes_kernel_local():
    out = emit("def k(x):\n    n = x * 2\n    def helper(v=n):\n        return v\n    return helper()\n")
    assert "def helper(v=rt.vars.n):" in out


def test_decorator_routes_kernel_local():
    """def/class decorators evaluate in the enclosing scope — instrumented
    reads redirect."""
    out = emit(
        "def k():\n    deco = make()\n    @deco\n    def helper():\n        return 1\n    return helper()\n"
    )
    assert "@rt.vars.deco" in out


def test_class_decorator_and_base_route_kernel_locals():
    out = emit(
        "def k(x):\n"
        "    base = make_base(x)\n"
        "    deco = make_deco()\n"
        "    @deco\n"
        "    class C(base):\n"
        "        pass\n"
        "    return C\n"
    )
    assert "@rt.vars.deco" in out
    assert "class C(rt.vars.base):" in out


def test_match_row():
    """``match`` row: the subject, pattern
    value-expressions and guards read-redirect; case bodies get the full
    statement rows prefixed by the native→instrumented binder sync."""
    out = emit(
        "def k(tag, base):\n"
        "    match tag:\n"
        "        case 1:\n"
        "            r = base\n"
        "        case [a, *rest] if base > 0:\n"
        "            r = a + len(rest)\n"
        "        case Point(px, py):\n"
        "            r = px + py\n"
        "        case colors.RED:\n"
        "            r = 0\n"
        '        case {"k": val}:\n'
        "            r = val\n"
        "        case _:\n"
        "            r = -1\n"
        "    return r\n"
    )
    assert "match rt.vars.tag:" in out
    assert "rt.vars.r = rt.vars.base" in out
    assert "rt.vars.a = a" in out
    assert "rt.vars.rest = rest" in out
    assert "if rt.vars.base > 0" in out
    assert "rt.vars.r = rt.vars.a + len(rt.vars.rest)" in out
    assert "case Point(px, py):" in out  # a global — stays native
    assert "case colors.RED:" in out  # a global — stays native
    assert "rt.vars.r = rt.vars.px + rt.vars.py" in out
    assert "rt.vars.val = val" in out


def test_match_guard_reads_binder_natively():
    """the guard evaluates *before* the binder sync (host ``match`` binds
    first) — binder reads stay native, other instrumented reads redirect."""
    out = emit(
        "def k(tag, base):\n"
        "    match tag:\n"
        "        case [a] if a > 0:\n"
        "            r = a + base\n"
        "        case _:\n"
        "            r = -1\n"
        "    return r\n"
    )
    assert "case [a] if a > 0:" in out  # `a` is a real local at guard time
    assert "rt.vars.r = rt.vars.a + rt.vars.base" in out


def test_lambda_call_keeps_parens():
    """a lambda invoked immediately keeps its parens through the rewrite:
    rendering ``Call(func=Lambda)`` from the tree drops them, silently attaching
    the call to the body's last operand."""
    out = emit("def k(x):\n    return (lambda y: y + 1)(x)\n")
    assert "(lambda y: y + 1)(rt.vars.x)" in out
    assert "y + 1(rt.vars.x)" not in out


def test_class_body_annotation_reads_route():
    """class-body annotations referencing an instrumented local are
    read-redirected — they evaluate eagerly in a no-future module."""
    out = emit("def k(v):\n    T = int\n    class C:\n        x: T = v\n    return C\n")
    assert "x: rt.vars.T = rt.vars.v" in out


def test_nested_def_annotations_route():
    """nested-def param/return annotations route reads —
    they evaluate at def time in the enclosing scope."""
    out = emit("def k(v):\n    T = int\n    def h(a: T) -> T:\n        return a\n    return h(v)\n")
    assert "def h(a: rt.vars.T) -> rt.vars.T:" in out


def test_annotations_not_routed_under_future_flag():
    """under ``from __future__ import annotations`` the annotation is a
    string — redirecting it would change the stored text."""
    out = instrument_function_source(
        "def k(v):\n    T = int\n    class C:\n        x: T = v\n    return C\n",
        "k",
        future_annotations=True,
    )[0]
    assert "x: T = rt.vars.v" in out
    assert "x: rt.vars.T" not in out


# ----------------- parens / unpacking / multi-with review fixes ----------------
# The expression rewriter rebuilds each node with ``with_changes``
# (not fresh ``cst.X(...)`` construction) so lpar/rpar and Arg.star survive
# — the lambda-paren fix was one instance of this systemic bug.


def test_grouped_binop_keeps_parens():
    """``(a + b) * 2`` keeps its grouping — a fresh BinaryOperation dropped
    the parens, re-parsing as ``a + (b * 2)``."""
    out = emit("def k(a, b):\n    return (a + b) * 2\n")
    assert "(rt.vars.a + rt.vars.b) * 2" in out
    assert "rt.vars.a + rt.vars.b * 2" not in out


def test_grouped_unary_keeps_parens():
    out = emit("def k(a, b):\n    return -(a + b)\n")
    assert "-(rt.vars.a + rt.vars.b)" in out


def test_grouped_ternary_keeps_parens():
    out = emit("def k(a, b):\n    return (a if a > b else b) + 1\n")
    assert "(rt.vars.a if rt.vars.a > rt.vars.b else rt.vars.b) + 1" in out


def test_grouped_subscript_keeps_parens():
    """``(a + b)[0]`` must not become ``a + b[0]`` — the subscript gluing
    onto ``b`` (the catastrophic case)."""
    out = emit("def k(a):\n    return (a + [9])[0]\n")
    assert "(rt.vars.a + [9])[0]" in out
    assert "rt.vars.a + [9][0]" not in out


def test_grouped_comparison_keeps_parens():
    """``(a < b) == c`` must not re-parse as the chained ``a < b == c``."""
    out = emit("def k(a, b, c):\n    return (a < b) == c\n")
    assert "(rt.vars.a < rt.vars.b) == rt.vars.c" in out


def test_grouped_fstring_keeps_parens():
    out = emit('def k(a, b):\n    return f"{(a + b) * 2}"\n')
    assert "{(rt.vars.a + rt.vars.b) * 2}" in out


def test_starred_call_arg_keeps_star():
    """``cst.Arg.star`` survives via ``with_changes``: ``f(*args)`` / ``f(**kw)``
    keep their unpacking operator — a fresh Arg dropped ``star`` so ``f(*a)``
    became ``f(a)``."""
    out = emit("def k(a, b):\n    args = (a, b)\n    return f(*args)\n")
    assert "f(*rt.vars.args)" in out
    out = emit("def k(a):\n    kw = {'x': a}\n    return f(**kw)\n")
    assert "f(**rt.vars.kw)" in out


def test_multi_item_with_is_nested():
    """multi-item ``with a as x, b(x) as y:`` nests into single-item withs
    so each as-target sync lands after its own ``__enter__`` — the
    second item reads the synced ``rt.vars.x``, not the pre-sync value."""
    out = emit("def k(a, b):\n    with a as x, b(x) as y:\n        z = x + y\n")
    assert out.count("rt.on_begin_with(") == 2
    # The innermost item carries the suite's single-call fact; the outer
    # item's effective body is the nested with, so its fact is False.
    assert re.search(
        r"rt\.on_begin_with\(rt\.vars\.b\(rt\.vars\.x\), name\s*=\s*'y', single_call_body=(True|False)\)",
        out,
    )
    assert "rt.vars.x = " in out  # the first item's sync, before the second


def test_del_attribute_and_subscript_route():
    """``del a.b`` / ``del a[i]`` are rewrite-safe and route through
    ``rt.vars``; only ``del <name>`` is rejected."""
    out = emit("def k(a, i):\n    del a[i]\n")
    assert "del rt.vars.a[rt.vars.i]" in out
    out = emit("def k(a):\n    del a.b\n")
    assert "del rt.vars.a.b" in out


def test_attribute_and_subscript_for_targets_route():
    """``for x.a in it:`` / ``for a[i] in it:`` are rewrite-safe and route
    tuple/List for-targets stay rejected."""
    out = emit("def k(a, it):\n    for a.x in it:\n        pass\n")
    assert re.search(r"for __pto_it\d+ in rt\.begin_for\(rt\.vars\.it\):", out)
    assert re.search(r"rt\.vars\.a\.x = __pto_it\d+", out)
    out = emit("def k(a, it):\n    for a[0] in it:\n        pass\n")
    assert re.search(r"rt\.vars\.a\[0\] = __pto_it\d+", out)


# ------------------------- nested-scope and expression review fixes ----------


def test_walrus_in_lambda_routes_value_reads():
    """A lambda-local walrus keeps its native binder, but its value still
    reads outer kernel locals — the earlier fallthrough left the value
    un-routed, so a rebound param was read natively and stale."""
    out = emit("def k(a):\n    f = lambda: (y := a + 1)\n    return f()\n")
    assert "(y := (rt.vars.a + 1))" in out


def test_walrus_in_nested_def_routes_value_reads():
    """A nested-scope walrus binds natively in that scope; its value routes."""
    out = emit(
        "def k(a):\n    def helper():\n        y = (x := a + 1)\n        return x\n    return helper()\n"
    )
    assert "(x := (rt.vars.a + 1))" in out


def test_fstring_format_spec_routes_reads():
    """``f"{a:{w}.2f}"`` — the spec is a JoinedStr of expressions evaluated
    at f-string time; an un-routed spec read the param ``w`` natively and
    stale after a rebind."""
    out = emit('def k(a, w):\n    return f"{a:{w}.2f}"\n')
    assert "f'{rt.vars.a:{rt.vars.w}.2f}'" in out


def test_fstring_nested_format_spec_routes_reads():
    """Routing the spec recurses: ``f"{a:{w:{p}}}"`` routes the innermost
    spec expression too."""
    out = emit('def k(a, w, p):\n    return f"{a:{w:{p}}}"\n')
    assert "f'{rt.vars.a:{rt.vars.w:{rt.vars.p}}}'" in out


def test_nested_raise_routes_outer_reads():
    out = emit("def k(a):\n    def helper():\n        raise ValueError(a)\n    return helper()\n")
    assert "raise ValueError(rt.vars.a)" in out


def test_nested_assert_routes_outer_reads():
    out = emit("def k(a):\n    def helper():\n        assert a, a\n    helper()\n    return a\n")
    assert "assert rt.vars.a, rt.vars.a" in out


def test_nested_class_keyword_values_route_outer_reads():
    """A class keyword value (``metaclass=``) evaluates in the enclosing
    scope like a base and must route its reads."""
    out = emit(
        "def k(m):\n"
        "    def make():\n"
        "        class C(metaclass=m):\n"
        "            pass\n"
        "        return C\n"
        "    return make()\n"
    )
    assert "class C(metaclass=rt.vars.m):" in out


@pytest.mark.skipif(sys.version_info < (3, 11), reason="except* syntax requires Python 3.11")
def test_nested_try_star_routes_outer_reads():
    """``except*`` groups are host plumbing inside nested scopes: the body
    routes reads while the group structure stays."""
    out = emit(
        "def k(a):\n"
        "    def helper():\n"
        "        try:\n"
        "            raise ExceptionGroup('g', [ValueError(a)])\n"
        "        except* ValueError:\n"
        "            out = 1\n"
        "        return out\n"
        "    return helper()\n"
    )
    assert "except* ValueError:" in out
    assert "ValueError(rt.vars.a)" in out


# ------------------- round-2 review fixes: genexp escape, nested scopes ----


def test_returned_generator_alias_is_rejected():
    """``g = (v + a for v in values); return g`` leaks the same lazy reads the
    direct-return refusal covers — the refusal must follow the bound name,
    not just the syntactic return position."""
    from pypto.language.tracing.collector.rewriter import RewriteError  # noqa: PLC0415

    with pytest.raises(RewriteError, match="returned generator escapes"):
        emit("def k(values, a):\n    g = (v + a for v in values)\n    return g\n")


def test_returned_generator_container_is_rejected():
    """``return (g,)`` carries the live generator out in a container — the
    same post-close escape, previously missed by the direct-return check."""
    from pypto.language.tracing.collector.rewriter import RewriteError  # noqa: PLC0415

    with pytest.raises(RewriteError, match="returned generator escapes"):
        emit("def k(values, a):\n    g = (v + a for v in values)\n    return (g,)\n")


def test_returned_generator_chain_in_container_is_rejected():
    """The taint flows through aliasing containers: ``t = (g, 1); return t``
    escapes exactly like the direct container return."""
    from pypto.language.tracing.collector.rewriter import RewriteError  # noqa: PLC0415

    with pytest.raises(RewriteError, match="returned generator escapes"):
        emit("def k(values, a):\n    g = (v + a for v in values)\n    t = (g, 1)\n    return t\n")


def test_returned_walrus_generator_is_rejected():
    """``return (g := (v + a for v in values))`` returns the bound generator
    through ``rt.bind`` — the routed call carries the value out."""
    from pypto.language.tracing.collector.rewriter import RewriteError  # noqa: PLC0415

    with pytest.raises(RewriteError, match="returned generator escapes"):
        emit("def k(values, a):\n    return (g := (v + a for v in values))\n")


def test_eagerly_consumed_generator_alias_passes():
    """``list(g)`` consumes the generator inside the invocation — the
    alias/container refusal must not over-reject working shapes (calls are
    the trust boundary, exactly as for the direct ``sum(...)`` genexp)."""
    out = emit("def k(values, a):\n    g = (v + a for v in values)\n    return list(g)\n")
    assert "list(rt.vars.g)" in out


def test_returned_read_free_generator_alias_passes():
    """A genexp whose lazy parts touch only its own binders survives the
    alias check like it survives the direct one."""
    out = emit("def k(values):\n    g = (v for v in values)\n    return g\n")
    assert re.search(r"__pto_ret\d+ = rt\.vars\.g", out)


def test_nested_except_type_routes_outer_reads():
    """``except outer_error:`` evaluates the type at catch time — an un-routed
    copy read the native closure's stale value after a rebind instead of the
    current ``rt.vars`` value."""
    out = emit(
        "def k(a):\n"
        "    def helper():\n"
        "        try:\n"
        "            raise ValueError(a)\n"
        "        except a:\n"
        "            return 1\n"
        "        return 0\n"
        "    return helper()\n"
    )
    assert "except rt.vars.a:" in out


def test_nested_del_attribute_and_subscript_route():
    """Nested ``del obj.attr`` / ``del obj[i]`` must mutate the current
    ``rt.vars`` object — the reads-only fallthrough deleted from the stale
    native closure value instead."""
    out = emit("def k(a):\n    def helper():\n        del a.attr\n    helper()\n    return a\n")
    assert "del rt.vars.a.attr" in out
    out = emit("def k(a):\n    def helper():\n        del a[0]\n    helper()\n    return a\n")
    assert "del rt.vars.a[0]" in out


def test_nested_del_local_name_stays_native():
    """A nested-scope ``del <name>`` deletes that scope's own binder —
    natively correct, so it stays native (the kernel-body ``del <name>``
    policy does not govern nested scopes)."""
    out = emit(
        "def k(a):\n    def helper():\n        x = 1\n        del x\n        return a\n    return helper()\n"
    )
    assert "del x" in out


def test_nested_global_colliding_with_kernel_local_passes_through():
    """``global a`` binds the module scope, not the kernel's instrumented
    local — the combined nonlocal/global refusal over-rejected the colliding
    name even though the write cannot touch the kernel cell."""
    out = emit(
        "def k(a):\n    def helper():\n        global a\n        a = 100\n    helper()\n    return a\n"
    )
    assert "global a" in out
    assert "\n        a = 100\n" in out  # the write stays native (module scope)


def test_nested_global_read_stays_native():
    """A ``global``-declared read resolves to the module scope — the scope
    collector shadows the name so the read does not route to ``rt.vars``."""
    out = emit("def k(a):\n    def helper():\n        global a\n        return a\n    return helper()\n")
    assert "\n        return a\n" in out
    assert "return rt.vars.a" not in out


# ------------------- collector-rewriter review fixes -----------------------


def test_comprehension_later_iterable_reads_enclosing_scope():
    """``[x for x in xs for y in y]``: a later generator's iterable resolves
    BEFORE its own target binds — it reads the ENCLOSING kernel local ``y``;
    the pre-collected full shadow wrongly treated it as comprehension-local
    (a stale native read after a rebind)."""
    out = emit("def k(xs, y):\n    return [x for x in xs for y in y]\n")
    assert "for y in rt.vars.y" in out


def test_nested_comprehension_in_later_iterable_routes_outer_read():
    """A nested comprehension as a later generator's iterable evaluates in
    the scope of the earlier generators only — the outer generator's
    not-yet-bound target must not shadow its first iterable's read."""
    out = emit("def k(xs, y):\n    return [x for x in xs for y in [w for w in y]]\n")
    assert "[w for w in rt.vars.y]" in out


def test_comprehension_condition_sees_own_target_only_after_it_binds():
    """A generator's conditions see its own target plus the earlier ones —
    incrementally, like the iterables; the element still sees every target."""
    out = emit("def k(xs, y):\n    return [x for x in xs if y for v in y]\n")
    assert "if rt.vars.y" in out


def test_walrus_in_assert_is_collected():
    """``assert (x := a)`` binds ``x`` in the kernel scope — the missing
    Assert collection branch left it un-collected, so a later read hit an
    unset ``rt.vars.x``."""
    out = emit("def k(a):\n    assert (x := a), (m := a)\n    return x + m\n")
    assert "rt.bind('x', rt.vars.a)" in out
    assert "rt.bind('m', rt.vars.a)" in out


def test_walrus_in_raise_is_collected():
    """``raise (x := Error(a))`` collects ``x`` like any kernel-scope walrus
    (the missing Raise collection branch lost it the same way)."""
    out = emit("def k(a):\n    raise (x := ValueError(a))\n")
    assert "rt.bind('x', ValueError(rt.vars.a))" in out


def test_nested_async_def_await_routes_reads():
    """A nested async helper keeps its ``await`` (host plumbing — the policy
    gates the kernel body only); its operand still routes outer reads."""
    out = emit("def k(a):\n    async def helper():\n        return await a\n    return helper\n")
    assert "await rt.vars.a" in out


def test_nested_async_def_keeps_async_constructs():
    """A nested async helper keeps its ``async for``/``async with`` too —
    they await through the async-iterator/context-manager protocols, so a
    de-asynced rebuild would break the helper at runtime (the kernel body
    alone is de-asynced with its clone)."""
    out = emit(
        "def k(ait, m):\n"
        "    async def helper():\n"
        "        async for v in ait:\n"
        "            pass\n"
        "        async with m as w:\n"
        "            return w\n"
        "    return helper\n"
    )
    assert "async for v in rt.vars.ait" in out
    assert "async with rt.vars.m as w" in out


def test_nested_target_reads_route():
    """Nested attribute/subscript assignment, ``for``, and ``with``-``as``
    targets route their outer base reads — the clone no longer binds the
    kernel local as a plain name, so a native base would resolve to a
    global."""
    out = emit(
        "def k(a):\n"
        "    x = [1]\n"
        "    def helper():\n"
        "        x[0] = a\n"
        "        for x[1] in a:\n"
        "            pass\n"
        "        with a as x[2]:\n"
        "            pass\n"
        "    helper()\n"
        "    return x\n"
    )
    assert "rt.vars.x[0] = rt.vars.a" in out
    assert "for rt.vars.x[1] in rt.vars.a" in out
    assert "as rt.vars.x[2]" in out


def test_nested_except_binder_shadows_kernel_local():
    """An ``except E as e`` binder is local to its handler: a read of the
    name stays native even when it collides with an instrumented kernel
    local (it used to route through ``rt.vars`` and read the stale outer
    value)."""
    out = emit(
        "def k(e):\n"
        "    def helper():\n"
        "        try:\n"
        "            n = int(e)\n"
        "        except ValueError as e:\n"
        "            return str(e)\n"
        "        return n\n"
        "    return helper\n"
    )
    assert "return str(e)" in out
    assert "str(rt.vars.e)" not in out


def test_match_mapping_rest_in_prefix_scan():
    """A mapping pattern's ``**rest`` binder is a plain string the Name walk
    never sees; the prefix must avoid it — the binder makes the name a local
    of the rewritten function, shadowing the interpreter handle everywhere."""
    out, prefix, _params = instrument_function_source(
        "def k(subject):\n"
        "    match subject:\n"
        "        case {'key': value, **rt}:\n"
        "            return value\n"
        "    return 0\n",
        "k",
    )
    assert prefix == "rt_0"
    assert "rt_0.vars" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
