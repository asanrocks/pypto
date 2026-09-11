# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The instrumentor's construct corpus and rejection contract.

The snippet corpus below is the construct contract: every Python construct
the rewrite subset accepts, each emitting parseable instrumented source with
a collision-free event prefix and an unchanged parameter list. It began as
the differential oracle that held the stdlib-ast rewriter to parity with the
retired libcst implementation; the corpus itself is the durable artifact.

The behavioral side (every corpus kernel runs end-to-end through the trace
driver against its native twin) lives in ``test_corpus.py``; policy
rejections are asserted here at the ``RewriteError`` level and in
``test_subset_policy.py`` at the message level.
"""

import ast
import inspect
import textwrap

import pytest
from _corpus import CASES
from collector.rewriter import RewriteError, emit_kernel
from collector.rewriter import instrument_function_source as emit


def _kernel_name(source: str) -> str:
    tree = ast.parse(source)
    return next(
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


SNIPPETS = [
    ("assign", "def k(a):\n    x = a\n"),
    ("assign_multi_target", "def k(a):\n    x = y = a\n    return y\n"),
    ("assign_unpack", "def k(a, d):\n    b, c = d\n    return b\n"),
    ("assign_star", "def k(d):\n    b, *rest = d\n    return rest\n"),
    ("assign_list_target", "def k(d):\n    [b, c] = d\n    return b\n"),
    ("augassign", "def k(a):\n    a += 1\n"),
    ("annassign", "def k(a):\n    x: int = a\n    return x\n"),
    ("annassign_bare", "def k(a):\n    x: int\n    return a\n"),
    ("bare_call", "def k():\n    consume()\n"),
    ("bare_noncall", "def k(x):\n    x + 1\n"),
    ("return_value", "def k(a):\n    return a\n"),
    ("return_bare", "def k(a):\n    return\n"),
    ("return_tuple", "def k(a, b):\n    return a, b\n"),
    ("if_else", "def k(a):\n    if a:\n        x = 1\n    else:\n        x = 2\n    return x\n"),
    (
        "if_elif_else",
        "def k(a):\n    if a == 1:\n        x = 1\n    elif a == 2:\n        x = 2\n    else:\n        x = 3\n    return x\n",
    ),
    ("if_noelse", "def k(a):\n    if a:\n        x = 1\n    return a\n"),
    ("if_one_line", "def k(a):\n    if a:\n        x = 1\n    return x\n"),
    (
        "else_wrapped_single_if",
        "def k(a):\n    if a:\n        x = 1\n    else:\n        if a > 2:\n            x = 2\n        else:\n            x = 3\n    return x\n",
    ),
    (
        "for_plain",
        "def k(values):\n    total = 0\n    for v in values:\n        total = total + v\n    return total\n",
    ),
    (
        "for_else",
        "def k(values):\n    for v in values:\n        break\n    else:\n        total = 1\n    return total\n",
    ),
    (
        "for_unpack",
        "def k(pairs):\n    total = 0\n    for i, j in pairs:\n        total = total + i\n    return total\n",
    ),
    ("while_plain", "def k(a):\n    while a:\n        a = a - 1\n    return a\n"),
    ("while_else", "def k(a):\n    while a:\n        break\n    else:\n        a = 0\n    return a\n"),
    ("break_in_for", "def k(values):\n    for v in values:\n        break\n    return v\n"),
    ("continue_in_for", "def k(values):\n    for v in values:\n        continue\n    return v\n"),
    (
        "break_in_if_in_for",
        "def k(values):\n    for v in values:\n        if v:\n            break\n    return v\n",
    ),
    (
        "continue_in_if_in_for",
        "def k(values):\n    for v in values:\n        if v:\n            continue\n    return v\n",
    ),
    (
        "break_in_while_in_for",
        "def k(values):\n    for v in values:\n        while v:\n            break\n    return v\n",
    ),
    (
        "try_finally",
        "def k(a):\n    try:\n        a = a + 1\n    finally:\n        a = a + 2\n    return a\n",
    ),
    ("with_single", "def k(m):\n    with m as v:\n        x = v\n    return x\n"),
    ("with_no_as", "def k(m):\n    with m:\n        x = 1\n    return x\n"),
    ("with_multi", "def k(m, n):\n    with m as a, n as b:\n        x = a + b\n    return x\n"),
    ("with_tuple_as", "def k(m):\n    with m as (a, b):\n        x = a\n    return x\n"),
    ("with_single_call_body", "def k(m):\n    with m:\n        f()\n"),
    ("with_single_assign_call_body", "def k(m):\n    with m:\n        x = f()\n    return x\n"),
    ("nested_def", "def k(a):\n    def helper():\n        return a\n    return helper()\n"),
    ("nested_def_local_shadow", "def k(a):\n    def helper(a):\n        return a\n    return helper(a)\n"),
    (
        "nested_def_default_reads_outer",
        "def k(a):\n    def helper(q=a):\n        return q\n    return helper()\n",
    ),
    (
        "nested_def_annotated",
        "def k(a):\n    def helper(q: 'T') -> 'T':\n        return q\n    return helper(a)\n",
    ),
    (
        "nested_def_control_flow",
        "def k(a):\n    def helper(q):\n        for i in range(q):\n            if i:\n                return i\n        return 0\n    return helper(a)\n",
    ),
    ("nested_class", "def k(a):\n    class C:\n        value = a\n    return C\n"),
    ("import_plain", "def k():\n    import os\n    return os\n"),
    ("import_from", "def k():\n    from os import path\n    return path\n"),
    ("import_as", "def k():\n    import os.path as p\n    return p\n"),
    ("import_bind_sync", "def k():\n    import os\n    return os.sep\n"),
    ("del_attr", "def k(obj):\n    del obj.attr\n"),
    ("del_subscript", "def k(obj):\n    del obj[0]\n"),
    ("del_tuple_targets", "def k(obj, other):\n    del obj.attr, other.attr\n"),
    ("raise_exc", "def k(a):\n    raise ValueError(a)\n"),
    ("raise_bare", "def k(a):\n    raise\n"),
    ("raise_from", "def k(a):\n    raise ValueError(a) from OSError(a)\n"),
    ("assert_plain", "def k(a):\n    assert a\n"),
    ("assert_msg", "def k(a):\n    assert a, a\n"),
    ("pass_stmt", "def k(a):\n    pass\n"),
    ("walrus_expr", "def k(a):\n    y = (x := a + 1)\n    return x + y\n"),
    ("walrus_in_comp", "def k(values):\n    y = [t := v for v in values]\n    return t\n"),
    ("lambda_expr", "def k(a):\n    f = lambda q: q + a\n    return f(1)\n"),
    ("lambda_shadow", "def k(a):\n    f = lambda a: a\n    return f(a)\n"),
    ("lambda_default", "def k(a):\n    f = lambda q=a: q\n    return f()\n"),
    ("lambda_in_call", "def k(a):\n    return map(lambda q: q + a, [1])\n"),
    ("listcomp", "def k(values, a):\n    return [v + a for v in values]\n"),
    ("listcomp_if", "def k(values, a):\n    return [v for v in values if v > a]\n"),
    ("nested_comp", "def k(grid, a):\n    return [v + a for row in grid for v in row]\n"),
    ("dictcomp", "def k(values, a):\n    return {v: a for v in values}\n"),
    ("setcomp", "def k(values, a):\n    return {v for v in values}\n"),
    ("genexp", "def k(values, a):\n    return (v + a for v in values)\n"),
    ("boolop_chain", "def k(a, b, c):\n    if a and b and c:\n        return 1\n    return 0\n"),
    ("boolop_mixed", "def k(a, b, c):\n    if a and b or not c:\n        return 1\n    return 0\n"),
    ("compare_chain", "def k(a, b, c):\n    return a < b <= c\n"),
    ("fstring", 'def k(a):\n    return f"{a}!"\n'),
    ("fstring_literal", 'def k(a):\n    return f"lit {a} tail"\n'),
    ("slice", "def k(a, b, c):\n    return a[b:c:d] if False else a[b:c:d]\n"),
    ("slice_store", "def k(a, i, v):\n    a[i : i + 1] = v\n"),
    ("subscript_tuple", "def k(a, i, j):\n    return a[i, j]\n"),
    ("subscript_multi_slice", "def k(a, i, j, n):\n    return a[i : i + n, j : j + n]\n"),
    ("subscript_slice_mixed", "def k(a, i, n):\n    return a[i : i + n, 0]\n"),
    ("subscript_slice_store", "def k(a, i, n, v):\n    a[i : i + n, i : i + n] = v\n"),
    ("subscript_store", "def k(a, i, v):\n    a[i] = v\n"),
    ("attr_store", "def k(a, v):\n    a.field = v\n"),
    ("starred_call", "def k(a, b):\n    return f(a, *b)\n"),
    ("kwd_call", "def k(a):\n    return f(a, x=a)\n"),
    ("dict_literal", "def k(a, m):\n    return {a: 1, **m}\n"),
    ("set_literal", "def k(a, b):\n    return {a, b}\n"),
    ("tuple_expr", "def k(a):\n    return (a, *a)\n"),
    (
        "match_as",
        "def k(subject):\n    match subject:\n        case value as name:\n            return name\n    return 0\n",
    ),
    (
        "match_class",
        "def k(subject, cls):\n    match subject:\n        case cls(x, y=z):\n            return x\n    return 0\n",
    ),
    (
        "match_mapping",
        "def k(subject, a):\n    match subject:\n        case {a.kind: value}:\n            return value\n    return 0\n",
    ),
    (
        "match_sequence",
        "def k(subject):\n    match subject:\n        case [first, *rest]:\n            return first\n    return 0\n",
    ),
    (
        "match_or",
        "def k(subject, a, b):\n    match subject:\n        case a.x | b.y:\n            return 1\n    return 0\n",
    ),
    (
        "match_guard",
        "def k(subject):\n    match subject:\n        case value if value > 0:\n            return value\n    return 0\n",
    ),
    (
        "match_value_attr",
        "def k(subject, a):\n    match subject:\n        case a.kind:\n            return 1\n    return 0\n",
    ),
    (
        "match_singleton",
        "def k(subject):\n    match subject:\n        case None | True:\n            return 1\n    return 0\n",
    ),
    (
        "match_binder_survives",
        "def k(subject):\n    match subject:\n        case 0:\n            r = 1\n        case _:\n            r = 2\n    return r\n",
    ),
    ("async_for", "async def k(it):\n    async for v in it:\n        pass\n    return v\n"),
    ("async_with", "async def k(m):\n    async with m as v:\n        x = v\n    return x\n"),
    ("async_nested_def", "def k(a):\n    async def helper():\n        return a\n    return helper\n"),
    (
        "comments_full",
        (
            "def k(a):\n"
            "    # leading\n"
            "    x = a  # trailing\n"
            "    if a:\n"
            "        # then lead\n"
            "        y = x\n"
            "    else:\n"
            "        # else lead\n"
            "        y = 0\n"
            "    return y\n"
        ),
    ),
    ("comment_between_stmts", ("def k(a):\n    x = a\n    # inter statement\n    y = x\n    return y\n")),
    ("comment_inside_call", ("def k(a):\n    return f(\n        a,  # inline argument comment\n    )\n")),
    ("docstring", ('def k(a):\n    """One line."""\n    return a\n')),
    ("docstring_multiline", ('def k(a):\n    """First.\n\n    Second."""\n    return a\n')),
    ("docstring_with_leading_comment", ('def k(a):\n    # before doc\n    """Doc."""\n    return a\n')),
    ("string_stmt_mid_body", ('def k(a):\n    x = a\n    "interlude string"\n    return x\n')),
    ("prefix_collision_rt", "def k(rt, a):\n    x = rt + a\n    return x\n"),
    ("prefix_collision_attribute", "def k(obj, a):\n    return obj.rt + a\n"),
    (
        "nested_def_with_decorator",
        (
            "def k(a):\n"
            "    def helper(q):\n"
            "        return q\n"
            "    helper = staticmethod(helper)\n"
            "    return helper(a)\n"
        ),
    ),
    ("decorated_kernel", "def deco(fn):\n    return fn\n\n\n@deco\ndef k(a):\n    return a\n"),
    ("future_annotations", ("def k(a) -> 'T':\n    x: 'T' = a\n    return x\n")),
]


def _snippets_with_corpus():
    """The hand-written corpus plus the behavioral corpus kernels' sources."""
    out = list(SNIPPETS)
    for name, kernel, _argsets in CASES:
        out.append((f"corpus_{name}", textwrap.dedent(inspect.getsource(kernel))))
    return out


@pytest.mark.parametrize("name,source", _snippets_with_corpus(), ids=[s[0] for s in _snippets_with_corpus()])
def test_snippet_emits_instrumented_source(name, source):
    """Every construct in the corpus emits parseable instrumented source with
    a collision-free event prefix and an unchanged parameter list."""
    kernel_name = _kernel_name(source)
    out, prefix, params = emit(source, kernel_name)
    tree = ast.parse(out)  # parseable: the driver compiles exactly this text
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == kernel_name
    )
    # The signature (name, parameters, annotations) passes through unchanged.
    orig = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == kernel_name
    )
    assert [a.arg for a in fn.args.args] == [a.arg for a in orig.args.args]
    assert params == [a.arg for a in orig.args.args]
    assert prefix.startswith("rt")


# ---------------------------------------------------------------------------
# Rejections: constructs outside the rewrite subset
# ---------------------------------------------------------------------------

ERROR_SNIPPETS = [
    ("yield", "def k(a):\n    yield a\n"),
    ("yield_from", "def k(a):\n    yield from a\n"),
    ("global", "def k(a):\n    global x\n    return a\n"),
    ("nonlocal", "def k(a):\n    nonlocal x\n    return a\n"),
    ("del_name", "def k(a):\n    del a\n"),
    ("del_name_in_tuple", "def k(a, obj):\n    del (a, obj.x)\n"),
    (
        "try_except",
        "def k(a):\n    try:\n        a = 1\n    except ValueError:\n        a = 2\n    return a\n",
    ),
    ("metaclass", "def k(a):\n    class C(metaclass=M):\n        pass\n    return a\n"),
    ("class_reassign", "def k(a):\n    __class__ = a\n    return a\n"),
    ("eval", "def k(a):\n    return eval(a)\n"),
    ("exec", "def k(a):\n    exec(a)\n"),
    ("compile", "def k(a):\n    return compile(a, a, a)\n"),
    ("dunder_import", "def k(a):\n    return __import__(a)\n"),
    ("getattr_nonliteral", "def k(a):\n    return getattr(a, a)\n"),
    ("getattr_starred", "def k(a, b):\n    return getattr(a, *b)\n"),
]


@pytest.mark.parametrize("name,source", ERROR_SNIPPETS, ids=[s[0] for s in ERROR_SNIPPETS])
def test_snippet_rejected_with_span(name, source):
    """Every out-of-subset construct raises a span-pointed RewriteError."""
    with pytest.raises(RewriteError) as err:
        emit(source, "k")
    assert "rewrite-unsafe construct" in err.value.message
    assert err.value.pos is not None, "policy rejections carry the construct's span"


def test_function_not_found():
    with pytest.raises(RewriteError, match="cannot find function 'missing'"):
        emit("def k(a):\n    return a\n", "missing")


def test_wildcard_import_collected_at_scope_pass():
    """Wildcard imports are rejected by the scope collector (the binder sync
    cannot enumerate star-imported names); the error is message-only, no span."""
    with pytest.raises(RewriteError, match="wildcard imports"):
        emit("def k():\n    from os import *\n    return os\n", "k")


# ---------------------------------------------------------------------------
# The free-variable environment wrapper
# ---------------------------------------------------------------------------


def test_freevars_env_wrapper_binds_closure_shape():
    """A kernel with free variables is wrapped in the synthetic environment
    that binds them before the rewritten definition, preserving the original
    closure shape for cell-by-name attachment."""
    source = "def k(a):\n    return a + limit\n"
    out, prefix = emit_kernel(source, "k", ("limit",))
    tree = ast.parse(out)
    env = tree.body[0]
    assert isinstance(env, ast.FunctionDef)
    assert env.name == "__collector_env_k"
    binds = [
        stmt.targets[0].id
        for stmt in env.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
    ]
    assert binds == ["limit"]
    inner = env.body[-1]
    assert isinstance(inner, ast.FunctionDef)
    assert inner.name == "k"
    assert prefix == "rt"
