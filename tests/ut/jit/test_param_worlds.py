# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Declaration-pinned parameter worlds and the ``pl.runtime`` entry marker.

A jit parameter's world is fixed by its declaration: a pl facade annotation
(bare class included — the deferred-shape spelling) is a **runtime
parameter**; a plain Python annotation (or none) is a **host knob** baked at
trace time. A contradicting call site is a type lie and is loud in both
directions — the migration contract's no-context-retyping rule applied to
call arguments. ``pl.runtime`` is the entry spelling for a runtime scalar:
the host value binds a runtime parameter and the dispatch cache keys on the
dtype, never the value.
"""

import pypto.language as pl
import pytest
from pypto.language.tracer.decorators.jit import _staged_declared
from pypto.pypto_core import ir

# —— the hint classifier ——————————————————————————————————————————————


@pytest.mark.parametrize(
    ("hint", "staged"),
    [
        (pl.Tensor, True),
        (pl.Tensor[[8], pl.FP32], True),
        (pl.Scalar, True),
        (pl.Scalar[pl.INT32], True),
        (pl.Out[pl.Tensor[[8], pl.FP32]], True),
        (pl.InOut[pl.Tensor[[8], pl.FP32]], True),
        (int, False),
        (float, False),
        (bool, False),
        (None, False),
    ],
)
def test_staged_hint_classification(hint, staged):
    assert _staged_declared(hint) is staged


# —— pl.runtime: the entry runtime-scalar spelling ————————————————————————


@pl.jit
def _entry(
    x: pl.Tensor[[8], pl.FP32],
    n: pl.Scalar[pl.INT32],
    scale: float = 1.0,
):
    t = pl.load(x, [0, 0], [8, 1])
    return pl.store(pl.mul(t, scale), [0, 0], x)


def _torch_x():
    torch = pytest.importorskip("torch")
    return torch.zeros(8, dtype=torch.float32)


def test_runtime_scalar_binds_a_runtime_param():
    x = _torch_x()
    bridged, has_torch = _entry._bridge_args((x, pl.runtime(5), 2.0))
    assert has_torch
    program = _entry._build_program(bridged)
    entry = next(fn for fn in program.functions.values() if fn.name == "_entry")
    assert [p.name_hint for p in entry.params] == ["x", "n"], "n survives, scale (host knob) bakes away"


def test_runtime_scalar_cache_key_is_type_only():
    """One artifact serves every value — the property the spelling exists
    for. The dtype still splits (a different ABI)."""
    x = _torch_x()
    b5, _ = _entry._bridge_args((x, pl.runtime(5), 2.0))
    b9, _ = _entry._bridge_args((x, pl.runtime(9), 2.0))
    b64, _ = _entry._bridge_args((x, pl.runtime(5, pl.INT64), 2.0))
    assert _entry._cache_key(b5) == _entry._cache_key(b9)
    assert _entry._cache_key(b5) != _entry._cache_key(b64)


def test_host_knob_still_bakes():
    x = _torch_x()
    p2 = _entry._build_program(_entry._bridge_args((x, pl.runtime(5), 2.0))[0])
    p3 = _entry._build_program(_entry._bridge_args((x, pl.runtime(5), 3.0))[0])
    assert str(p2.functions) != str(p3.functions), "scale is a host knob: values bake differently"


def test_runtime_scalar_dtype_inference_and_validation():
    assert pl.runtime(5).dtype == pl.INT32
    assert pl.runtime(2.5).dtype == pl.FP32
    assert pl.runtime(True).dtype == pl.BOOL
    assert pl.runtime(5, pl.INT64).dtype == pl.INT64
    with pytest.raises(TypeError, match="pl.runtime accepts a host bool/int/float"):
        pl.runtime("nope")


def test_dispatch_unwraps_the_marker(monkeypatch):
    """The dispatch lane hands CompiledProgram the raw host value."""
    x = _torch_x()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        _entry, "_dispatch", lambda args, bridged, config=None: captured.setdefault("args", args)
    )
    _entry(x, pl.runtime(7), 2.0)
    assert captured["args"][1].value == 7


# —— the mismatch errors ———————————————————————————————————————————————


def test_entry_host_scalar_into_runtime_param_is_loud():
    x = _torch_x()
    with pytest.raises(TypeError, match="pass pl.runtime"):
        _entry(x, 5, 2.0)


def test_entry_runtime_marker_into_host_param_is_loud():
    x = _torch_x()
    with pytest.raises(TypeError, match="cannot become host"):
        _entry(x, pl.runtime(5), pl.runtime(2.0))


def test_entry_tensor_into_unannotated_param_is_loud():
    """An unannotated param is a host knob; a tensor cannot become host —
    annotate the param (any pl facade spelling) to receive tensors."""

    @pl.jit
    def k(x):
        return x

    with pytest.raises(TypeError, match="cannot become host"):
        k(_torch_x())


@pl.jit.incore
def _callee(x: pl.Tensor[[8], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[8], pl.FP32]:
    return pl.add(x, 0.0)


@pl.jit.incore
def _host_knob_callee(x: pl.Tensor[[8], pl.FP32], scale: float) -> pl.Tensor[[8], pl.FP32]:
    return pl.add(x, scale)


def test_jit_callee_lane_host_into_runtime_param_is_loud():
    """Inside a jit trace, a host int into the eager callee's Scalar param
    is the declaration lie the variant lane used to bake silently."""
    torch = pytest.importorskip("torch")

    @pl.jit
    def caller(x: pl.Tensor, y: pl.Tensor):
        return _callee(x, 3)

    with pytest.raises(TypeError, match="pass pl.runtime"):
        caller(torch.zeros(8), torch.empty(8))


def test_jit_callee_lane_staged_into_host_param_is_loud():
    torch = pytest.importorskip("torch")

    @pl.jit
    def caller(x: pl.Tensor, y: pl.Tensor):
        s = pl.const(2.0, pl.FP32)
        return _host_knob_callee(x, s)

    with pytest.raises(TypeError, match="cannot become host"):
        caller(torch.zeros(8), torch.empty(8))


def test_standalone_lane_stays_parser_aligned():
    """The standalone ``pl.function`` lane is the parser's world: a host
    literal reaching a staged param lifts to a typed ``Const`` *argument*
    (the param stays runtime) — the round-trip dialect prints ``f(x, 3)``
    and the parser must re-read it. The world checks are jit-envelope
    policy, not a frontend-wide rule."""

    @pl.function
    def caller(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        return _callee(x, 3)

    assert isinstance(caller, ir.Function)
    assert any(p.name_hint == "n" for p in _callee.params), "the param stays runtime"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
