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

import pytest
from collector.rewriter import RewriteError, instrument_function_source


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
