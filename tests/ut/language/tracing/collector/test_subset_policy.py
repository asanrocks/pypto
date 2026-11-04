# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Subset policy: rewrite-unsafe constructs are rejected at
rewrite time with a span-pointed ``RewriteError``."""

import sys

import pytest
from pypto.language.tracing.collector.rewriter import RewriteError, instrument_function_source


def _errors(source: str) -> RewriteError:
    with pytest.raises(RewriteError) as excinfo:
        instrument_function_source(source, "k")
    return excinfo.value


def test_try_except_is_rejected():
    err = _errors("def k(a):\n    try:\n        x = a\n    except ValueError:\n        x = 0\n")
    assert "try/except" in str(err)


def test_try_finally_passes_through():
    """``try``/``finally`` (no except) is rewrite-safe."""
    out = instrument_function_source(
        "def k(a):\n    try:\n        x = a\n    finally:\n        y = 0\n    return x\n", "k"
    )[0]
    assert "try:" in out
    assert "finally:" in out


def test_del_is_rejected():
    err = _errors("def k(a):\n    x = a\n    del x\n")
    assert "del" in str(err)


def test_global_nonlocal_are_rejected():
    assert "global/nonlocal" in str(_errors("def k(a):\n    global g\n    return a\n"))
    assert "global/nonlocal" in str(_errors("def k(a):\n    nonlocal n\n    return a\n"))


def test_eval_exec_compile_are_rejected():
    assert "eval()" in str(_errors("def k():\n    x = eval('1 + 1')\n"))
    assert "exec()" in str(_errors("def k():\n    exec('x = 1')\n"))
    assert "compile()" in str(_errors("def k():\n    compile('x', '<s>', 'exec')\n"))


def test_dynamic_import_is_rejected():
    """``__import__`` is a dynamic import — rewrite-unsafe."""
    assert "__import__()" in str(_errors("def k():\n    __import__('os')\n"))


def test_dynamic_getattr_is_rejected():
    assert "getattr()" in str(_errors("def k(a, name):\n    return getattr(a, name)\n"))
    assert "setattr()" in str(_errors("def k(a, name):\n    setattr(a, name, 1)\n"))


def test_literal_getattr_passes_through():
    out = instrument_function_source("def k(a):\n    return getattr(a, 'x')\n", "k")[0]
    assert "getattr(rt.vars.a, 'x')" in out


def test_tuple_unpack_for_target_routes_through():
    """Tuple for-targets route through ``_route_target`` — the earlier
    rejection is dropped for the round-trip dialect's
    ``for i, (acc,) in pl.range(init_values=...)``."""
    out = instrument_function_source("def k(it):\n    for a, b in it:\n        pass\n", "k")[0]
    assert "rt.vars.a, rt.vars.b = __pto_it0" in out


def test_metaclass_is_rejected():
    err = _errors("def k():\n    class C(metaclass=type):\n        pass\n")
    assert "metaclass" in str(err)


def test_dunder_class_reassignment_is_rejected():
    err = _errors("def k():\n    __class__ = 1\n    return __class__\n")
    assert "__class__" in str(err)


def test_yield_is_rejected():
    err = _errors("def k():\n    yield 1\n")
    assert "yield" in str(err)


def test_error_is_span_pointed():
    err = _errors("def k():\n    x = 1\n    del x\n")
    assert err.lineno == 3  # the del statement's line, 1-based


def test_nested_def_is_host_plumbing_not_policed():
    """helpers are pure host plumbing: a helper may use
    eval/yield — the policy gates the kernel body only."""
    out = instrument_function_source(
        "def k(a):\n    def helper():\n        return eval('1 + 1')\n    return helper() + a\n",
        "k",
    )[0]
    assert "eval('1 + 1')" in out


# --------------------------- del / for-target review fixes --------------------


def test_del_name_still_rejected():
    """``del <name>`` desyncs the instrumented locals from the host locals
    (the host local is gone but ``rt.vars`` keeps the old value) — stays
    in the subset policy."""
    err = _errors("def k():\n    x = 1\n    del x\n")
    assert "del <name>" in str(err)


def test_del_attribute_and_subscript_pass_through():
    """``del a.b`` / ``del a[i]`` mutate an attribute/subscript (rewrite-safe)
    — not policed; they route through ``rt.vars``."""
    instrument_function_source("def k(a):\n    del a.b\n", "k")  # no raise
    instrument_function_source("def k(a, i):\n    del a[i]\n", "k")


def test_attribute_and_subscript_for_targets_pass_through():
    """``for x.a in it:`` / ``for a[i] in it:`` are rewrite-safe;
    tuple/List for-targets route through ``_route_target``."""
    instrument_function_source("def k(a, it):\n    for a.x in it:\n        pass\n", "k")
    instrument_function_source("def k(a, it):\n    for a[0] in it:\n        pass\n", "k")


def test_list_unpack_for_target_routes_through():
    """List for-targets route through ``_route_target`` too."""
    out = instrument_function_source("def k(it):\n    for [a, b] in it:\n        pass\n", "k")[0]
    assert "[rt.vars.a, rt.vars.b] = __pto_it0" in out


# --------------------------- review fixes: except*, nested nonlocal ----------


@pytest.mark.skipif(sys.version_info < (3, 11), reason="except* syntax requires Python 3.11")
def test_try_star_is_rejected():
    """``try/except*`` does not share ``ast.Try``'s rewrite and would fall
    through uninstrumented — outside the rewrite subset."""
    err = _errors("def k(a):\n    try:\n        x = a\n    except* ValueError:\n        x = 0\n")
    assert "except*" in str(err)


def test_nested_nonlocal_to_kernel_local_is_rejected():
    """A nested ``nonlocal`` naming an instrumented kernel local would write
    a cell nothing reads (or fail to compile for plain locals); refuse it
    loudly instead of silently dropping the write."""
    err = _errors(
        "def k(a):\n    def helper():\n        nonlocal a\n        a = 100\n    helper()\n    return a\n"
    )
    assert "nonlocal naming a kernel local" in str(err)


def test_nested_global_colliding_with_kernel_local_is_accepted():
    """A nested ``global`` binds the module scope, not the kernel's cell — a
    name merely colliding with an instrumented kernel local is host plumbing,
    not a mis-routed write, so it stays accepted (unlike ``nonlocal``)."""
    instrument_function_source(
        "def k(a):\n    def helper():\n        global a\n        a = 100\n    helper()\n    return a\n",
        "k",
    )  # no raise


def test_nested_global_to_module_name_passes_through():
    """A nested ``global`` that does not name a kernel local is host
    plumbing and stays unchanged."""
    out = instrument_function_source(
        "def k(a):\n    def helper():\n        global g\n        g = 1\n    helper()\n    return a\n",
        "k",
    )[0]
    assert "global g" in out


def test_returned_generator_alias_is_rejected():
    """A generator bound to a local and then returned escapes the instrumented
    frame exactly like a directly returned one — its lazy parts read the
    interpreter after the driver closed it."""
    err = _errors("def k(values, a):\n    g = (v + a for v in values)\n    return g\n")
    assert "returned generator escapes" in str(err)


def test_returned_generator_in_container_is_rejected():
    """A container returned with a leaked generator inside escapes the same
    way (``return (g,)``)."""
    err = _errors("def k(values, a):\n    g = (v + a for v in values)\n    return (g,)\n")
    assert "returned generator escapes" in str(err)


def test_await_in_kernel_body_is_rejected():
    """The kernel clone is de-asynced (rebuilt as a plain ``def``), so a
    surviving ``await`` would fail at Python compile time with a foreign
    location — rejected here with the construct's span instead."""
    err = _errors("async def k(a):\n    return await a\n")
    assert "await" in str(err)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the type statement requires Python 3.12")
def test_type_alias_is_rejected():
    """``type T = ...`` binds a kernel-scope name the instrumented locals
    never see (the fail-open default emitted it unchanged); unknown
    statement kinds are rejected loudly — only the known-safe ``pass``\n    stays allowed."""
    err = _errors("def k(a):\n    type T = int\n    return a\n")
    assert "TypeAlias" in str(err)


def test_pass_stays_allowed():
    instrument_function_source("def k(a):\n    pass\n", "k")  # no raise


def test_policy_error_carries_line_offset():
    """A policy error in dedented source reports the ORIGINAL file line —
    the same ``line_offset`` shift the generated ``_span`` events apply."""
    from pypto.language.tracing.collector.rewriter import emit_kernel  # noqa: PLC0415

    with pytest.raises(RewriteError) as excinfo:
        emit_kernel("def k(a):\n    x = a\n    del x\n", "k", line_offset=10)
    assert excinfo.value.lineno == 13  # del on dedented line 3, offset 10


def test_returned_generator_through_subscript_is_rejected():
    """Indexing a container that holds the generator picks it back out
    (``return [g][0]``) — the same escape as returning the container itself."""
    err = _errors("def k(values, a):\n    g = (v + a for v in values)\n    return [g][0]\n")
    assert "returned generator escapes" in str(err)


def test_eagerly_indexed_comprehension_still_passes():
    """An eagerly evaluated comprehension indexed on the spot consumes
    itself inside the invocation — not a leak."""
    out = instrument_function_source("def k(values, a):\n    return [v + a for v in values][0]\n", "k")[0]
    assert "[v + rt.vars.a for v in rt.vars.values][0]" in out


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the type statement requires Python 3.12")
def test_nested_type_alias_is_rejected():
    """The kernel body rejects unknown statement kinds; a nested helper
    must not fail open on them either — ``type T = ...`` binds a nested-local
    name the reads-only walker would leave un-routed."""
    err = _errors("def k(a):\n    def helper():\n        type T = a\n        return T\n    return helper\n")
    assert "TypeAlias" in str(err)


def test_nested_import_stays_native():
    """A nested import binds nothing instrumented and reads nothing — host
    plumbing, kept native (its binders are collected into the shadow)."""
    out = instrument_function_source(
        "def k(a):\n    def helper():\n        import os\n        return os.sep, a\n    return helper\n", "k"
    )[0]
    assert "import os" in out


def test_wildcard_import_error_carries_line_offset():
    """The wildcard-import rejection is raised by the scope collector before
    a ``_Ctx`` exists, so it must shift its span by ``line_offset`` itself —
    dedented source reports the ORIGINAL file line like every other policy
    error."""
    from pypto.language.tracing.collector.rewriter import emit_kernel  # noqa: PLC0415

    with pytest.raises(RewriteError) as excinfo:
        emit_kernel("def k(a):\n    from x import *\n    return a\n", "k", line_offset=100)
    assert excinfo.value.pos == (102, 4)  # import on dedented line 2, offset 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
