# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call and submit share one instantiation semantics.

A Submit reference resolves through the same per-call lane a call takes
(``Marker.resolve_submit`` → ``_LazyGlobalFn._specialized_resolve``): the
specialized target, the surviving runtime args, the world checks, and the
variant dedup are identical — only the emitted instruction differs
(``ir.Submit`` vs ``ir.Call``). Specialization is likewise a property of the
callee, never of where it is bound: closure-captured callees route through
the same lane as module-global ones.
"""

import pypto.language as pl
import pytest
from pypto.pypto_core import ir

# Module-level kernels: ``self`` materialization (the jit registry proxy)
# requires the sibling to be a module global.


@pl.jit.incore
def _routed_callee(
    x: pl.Tensor[[8], pl.FP32],
    y: pl.Out[pl.Tensor[[8], pl.FP32]],
    n: pl.Scalar[pl.INT32],
) -> pl.Tensor[[8], pl.FP32]:
    y[:] = pl.add(x, pl.cast(n, pl.FP32))
    return y


@pl.jit.incore
def _knob_callee(
    x: pl.Tensor[[8], pl.FP32],
    y: pl.Out[pl.Tensor[[8], pl.FP32]],
    scale: float,
) -> pl.Tensor[[8], pl.FP32]:
    y[:] = pl.add(x, scale)
    return y


@pl.jit
def _submit_runtime(x: pl.Tensor, y: pl.Tensor, n: pl.Scalar[pl.INT32]):
    out, _tid = pl.spmd_submit(self._routed_callee, x, y, n, core_num=1)  # noqa: F821
    return out


@pl.jit
def _call_runtime(x: pl.Tensor, y: pl.Tensor, n: pl.Scalar[pl.INT32]):
    return self._routed_callee(x, y, n)  # noqa: F821


@pl.jit
def _submit_knob(x: pl.Tensor, y: pl.Tensor, scale: float):
    out, _tid = pl.spmd_submit(self._knob_callee, x, y, scale, core_num=1)  # noqa: F821
    return out


@pl.jit
def _call_knob(x: pl.Tensor, y: pl.Tensor, scale: float):
    return self._knob_callee(x, y, scale)  # noqa: F821


def _tensors():
    torch = pytest.importorskip("torch")
    return torch.zeros(8, dtype=torch.float32), torch.empty(8, dtype=torch.float32)


def _callee_params(program, name_part):
    return {
        fn.name: [p.name_hint for p in fn.params] for fn in program.functions.values() if name_part in fn.name
    }


def test_submit_and_call_instantiate_identically_runtime_param():
    """A staged runtime scalar reaches the callee identically through both
    instructions: the param survives, the program shapes match."""
    x, y = _tensors()
    via_submit = _submit_runtime._build_program(_submit_runtime._bridge_args((x, y, pl.runtime(5)))[0])
    via_call = _call_runtime._build_program(_call_runtime._bridge_args((x, y, pl.runtime(5)))[0])
    submit_callee = _callee_params(via_submit, "_routed_callee")
    call_callee = _callee_params(via_call, "_routed_callee")
    assert submit_callee == call_callee == {"_routed_callee": ["x", "y", "n"]}


def test_submit_and_call_instantiate_identically_host_knob():
    """A host knob bakes identically through both instructions: the param is
    dropped, the constant lands in the body."""
    x, y = _tensors()
    via_submit = _submit_knob._build_program(_submit_knob._bridge_args((x, y, 2.0))[0])
    via_call = _call_knob._build_program(_call_knob._bridge_args((x, y, 2.0))[0])
    submit_callee = _callee_params(via_submit, "_knob_callee")
    call_callee = _callee_params(via_call, "_knob_callee")
    assert submit_callee == call_callee == {"_knob_callee": ["x", "y"]}
    assert "adds(x, 2.0)" in "".join(str(fn) for fn in via_submit.functions.values())
    assert "adds(x, 2.0)" in "".join(str(fn) for fn in via_call.functions.values())


def test_submit_world_mismatch_is_loud():
    """The declaration-pinned world check covers submit references: a host
    scalar into a Scalar-declared param is the same type lie as at a call."""
    x, y = _tensors()
    with pytest.raises(TypeError, match="pass pl.runtime"):

        @pl.jit
        def caller(x: pl.Tensor, y: pl.Tensor, n: pl.Scalar[pl.INT32]):
            out, _tid = pl.spmd_submit(self._routed_callee, x, y, 3, core_num=1)  # noqa: F821
            return out

        caller._build_program(caller._bridge_args((x, y, pl.runtime(5)))[0])


def test_submit_and_call_of_one_callee_share_the_variant():
    """Both lanes in one trace dedupe to one callee: the program carries the
    shared variant, and the frozen decoration-time function never joins."""
    x, y = _tensors()

    @pl.jit
    def both(x: pl.Tensor, y: pl.Tensor, n: pl.Scalar[pl.INT32]):
        out, _tid = pl.spmd_submit(self._routed_callee, x, y, n, core_num=1)  # noqa: F821
        return self._routed_callee(x, y, n)  # noqa: F821

    program = both._build_program(both._bridge_args((x, y, pl.runtime(5)))[0])
    assert sorted(fn.name for fn in program.functions.values()) == ["_routed_callee", "both"]


# —— closure-captured callees: specialization is callee-owned ———————————————


@pl.jit.incore
def _closure_knob_callee(x: pl.Tensor[[8], pl.FP32], scale: float) -> pl.Tensor[[8], pl.FP32]:
    return pl.add(x, scale)


def _global_entry():
    @pl.jit
    def via_global(x: pl.Tensor, scale: float):
        return _closure_knob_callee(x, scale)

    return via_global


def _closure_entry():
    helper = _closure_knob_callee  # the closure cell

    @pl.jit
    def via_closure(x: pl.Tensor, scale: float):
        return helper(x, scale)

    return via_closure


def test_closure_captured_callee_bakes_like_the_global_lane():
    """A host knob bakes identically whether the callee was reached through
    a closure cell or a module global — the former verbatim asymmetry."""
    x, _ = _tensors()
    via_global = _global_entry()._build_program(_global_entry()._bridge_args((x, 2.0))[0])
    via_closure = _closure_entry()._build_program(_closure_entry()._bridge_args((x, 2.0))[0])
    g = _callee_params(via_global, "_closure_knob_callee")
    c = _callee_params(via_closure, "_closure_knob_callee")
    assert g == c == {"_closure_knob_callee": ["x"]}
    assert "adds(x, 2.0)" in "".join(str(fn) for fn in via_global.functions.values())
    assert "adds(x, 2.0)" in "".join(str(fn) for fn in via_closure.functions.values())


# —— the verbatim world: plain Markers decline ————————————————————————————


@pl.function(type=ir.FunctionType.InCore)
def _standalone_callee(
    x: pl.Tensor[[8], pl.FP32],
    y: pl.Out[pl.Tensor[[8], pl.FP32]],
) -> pl.Tensor[[8], pl.FP32]:
    y[:] = x
    return y


@pl.jit
def _submit_standalone(x: pl.Tensor, y: pl.Tensor):
    out, _tid = pl.spmd_submit(_standalone_callee, x, y, core_num=1)
    return out


def test_standalone_callee_stays_verbatim():
    """A parser-produced standalone function has no decoration record: the
    submit reference is verbatim — the frozen function joins the program
    unchanged (the conformance-oracle world)."""
    x, y = _tensors()
    program = _submit_standalone._build_program(_submit_standalone._bridge_args((x, y))[0])
    callees = _callee_params(program, "_standalone_callee")
    assert callees == {"_standalone_callee": ["x", "y"]}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
