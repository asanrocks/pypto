# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PtoSema + the straight-line value anchors.

Every anchor asserts **RAW** equality: the tracing frontend's ``bind`` must emit
the same ``let``-on-first / ``assign``-on-rebind as the parser frontend — any
divergence is a ``bind`` bug, not a pass artifact.
"""

import pypto.language as pl
from pypto.language.tracer.collector.interface import HostBinding, StagedBinding
from pypto.language.tracer.decorators.function import function_trace
from pypto.language.tracer.semantics import PtoSema
from pypto.pypto_core import DataType, ir

from ._harness import conformance

# The tracing-frontend build for the anchors.
build = function_trace


def k_load_add_store(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.add(t, t)
    pl.store(s, [0, 0], x)
    return x


def k_scalar_chain(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.add(t, t)
    r = pl.add(s, t)
    pl.store(r, [0, 0], x)
    return x


def k_rebind(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    t = pl.add(t, t)  # rebind reuses the same mutable ir.Var
    pl.store(t, [0, 0], x)
    return x


def k_mem_vec(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
    pl.store(t, [0, 0], x)
    return x


# —— the raw-regime anchors ——


def test_load_add_store_raw_anchor():
    conformance(k_load_add_store, build)


def test_scalar_chain_raw_anchor():
    conformance(k_scalar_chain, build)


def test_rebind_assigns_same_cell():
    """A rebind reuses the ``ir.Var`` (an ``AssignStmt``, not a second
    ``LetStmt``) — the C1 rebind shape, verified raw against the AST
    frontend."""
    conformance(k_rebind, build)


def test_target_memory_kwarg_passthrough():
    """Keyword args on ops (target_memory) flow through the executed path
    identically to the parser's invoke_dsl path."""
    conformance(k_mem_vec, build)


# —— the PtoSema surface ——


def test_function_trace_returns_function():
    fn = function_trace(k_load_add_store)
    assert isinstance(fn, ir.Function)
    assert fn.name == "k_load_add_store"


def test_is_staged_query():
    """Facades are staged values,
    host values (literals, raw Expr, lists) are not."""
    sema = PtoSema(name="probe")
    frame = sema.begin_frame()
    assert frame.is_staged(pl.Tensor(expr=ir.ConstInt(1, DataType.INDEX, ir.Span.unknown())))
    assert not frame.is_staged(0)
    assert not frame.is_staged(1.5)
    assert not frame.is_staged([1, 2])


def test_bind_refuses_bare_literal():
    """Typed staged initialization at a staged bind: a
    bare literal must never reach the staged-local path. The flat surface
    declines at ``try_bind`` (None — the interpreter stores the host value);
    the loud guardrail is the interpreter's typed staged initialization
    check on a staged rebind over a literal. Here the Sema-side
    decline is pinned."""
    sema = PtoSema(name="probe")
    frame = sema.begin_frame()
    assert isinstance(frame.bind(None, "acc", 0), HostBinding)
    assert isinstance(frame.bind(None, "acc", 1.5), HostBinding)


def test_bind_param_returns_wrapped_var():
    """bind_parameters (the decorator-adapter entrance) creates the param Var and
    returns the expr-wrapping facade wrapping it — the value the trace interpreter
    stores in its locals. The locals map holds the param Var, so a
    rebind assigns through it."""
    sema = PtoSema(name="probe")
    facade = pl.Tensor([64, 128], pl.FP16, _annotation_only=True)
    (bound_param,) = sema.bind_parameters({"x": facade})
    frame = sema.begin_frame()
    assert isinstance(bound_param, pl.Tensor)
    assert isinstance(bound_param.unwrap(), ir.Var)
    # the read-path handback: a facade to store (a fresh facade wrapping
    # the new Var), not the bool True — alias binds must not mutate the
    # source facade.
    bound = frame.bind(None, "x", bound_param)
    assert isinstance(bound, StagedBinding)
    assert isinstance(bound.value, pl.Tensor)
    assert isinstance(bound.value.unwrap(), ir.Var)
    frame.finish()
    fn = sema.result()
    assert fn.name == "probe"


def test_finish_idempotent():
    """finish is idempotent (the second call hands back the stored result)."""
    sema = PtoSema(name="probe")
    frame = sema.begin_frame()
    frame.finish()
    frame.finish()
    assert sema.result().name == "probe"
