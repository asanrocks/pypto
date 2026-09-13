# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Return-style jit dispatch must deliver the returned tensors to the host.

A ``@pl.jit`` kernel whose outputs are *return values* (no ``pl.Out``
parameters) used to compile and execute, then hand the host ``()``: the
compilation pipeline never gave the entry a host-visible slot for the result.
``ConvertTensorToTileOps`` allocates an internal ``tensor.create`` for the
callee's appended Out arg, orchestration codegen writes the task output into
that scope-local buffer, and nothing exports it — so the artifact's calling
convention has no output position at all.

These tests pin the fix at the three layers it spans (gap F8):

* **Pipeline**: the entry Orchestration function gains an appended Out param
  covering the returned tensor, the internal create disappears, and the return
  references the param.
* **Codegen**: the generated orchestration C++ expects one more tensor arg and
  binds the promoted buffer to ``orch_args.tensor(<n>)`` instead of a
  scope-local ``alloc_tensors``.
* **Artifact metadata**: the ``CompiledProgram`` built by ``ir.compile``
  reports the appended param as an auto-allocatable output, which is what the
  return-style dispatch packing in ``_invoke_compiled`` consumes.

The end-to-end dispatch check (the returned tensor equals ``a + b`` on the
simulator) lives in ``tests/st/runtime/framework_and_models/test_jit.py``.
"""

import pypto.language as pl
import pytest
from pypto import backend, codegen
from pypto.backend import BackendType
from pypto.ir.compiled_program import CompiledProgram
from pypto.pypto_core import DataType, ir


@pl.jit
def return_style_add(a: pl.Tensor, b: pl.Tensor):
    """Two inputs, one *returned* output — no ``pl.Out`` parameter."""
    with pl.at(level=pl.Level.CORE_GROUP):
        c = pl.add(a, b)
    return c


def _tensors():
    torch = pytest.importorskip("torch")
    return (
        torch.zeros(16, 16, dtype=torch.float32),
        torch.zeros(16, 16, dtype=torch.float32),
    )


def _post_pass_program():
    a, b = _tensors()
    return return_style_add.compile_for_test(a, b)


def _entry(program):
    entry = next(
        (f for f in program.functions.values() if f.func_type == ir.FunctionType.Orchestration),
        None,
    )
    assert entry is not None, "no Orchestration entry in the compiled program"
    return entry


def _find_return(stmt):
    """Depth-first search for the (function-terminal) ReturnStmt."""
    if isinstance(stmt, ir.ReturnStmt):
        return stmt
    for child in getattr(stmt, "stmts", None) or []:
        found = _find_return(child)
        if found is not None:
            return found
    body_children = []
    if hasattr(stmt, "body") and stmt.body is not None:
        body_children.append(stmt.body)
    if hasattr(stmt, "then_body") and stmt.then_body is not None:
        body_children.append(stmt.then_body)
    if getattr(stmt, "else_body", None) is not None:
        body_children.append(stmt.else_body)
    for child in body_children:
        found = _find_return(child)
        if found is not None:
            return found
    return None


class TestReturnStylePipeline:
    """The pass pipeline materializes the entry's return as an Out param."""

    def test_entry_gains_appended_out_param(self):
        program = _post_pass_program()
        entry = _entry(program)

        out_indices = [i for i, d in enumerate(entry.param_directions) if d == ir.ParamDirection.Out]
        assert out_indices == [2], (
            f"return-style entry must gain one appended Out param, got directions "
            f"{[str(d) for d in entry.param_directions]}"
        )
        out_param = entry.params[2]
        assert out_param.name_hint == "ret0__out"
        # The promoted buffer keeps the tensor's shape/dtype so the host can
        # allocate it (return-style output packing reads exactly this type).
        out_type = out_param.type
        assert isinstance(out_type, ir.TensorType)
        assert [d.value for d in out_type.shape] == [16, 16]
        assert out_type.dtype == DataType.FP32

    def test_return_references_the_promoted_param(self):
        program = _post_pass_program()
        entry = _entry(program)
        out_param = entry.params[2]

        ret = _find_return(entry.body)
        assert ret is not None, "entry has no ReturnStmt"
        assert len(ret.value) == 1
        (value,) = ret.value
        assert isinstance(value, ir.Var)
        assert value.unique_id == out_param.unique_id, (
            "the entry's return value must be the promoted Out param by identity"
        )

    def test_internal_create_is_gone(self):
        program = _post_pass_program()
        entry = _entry(program)

        source = entry.body.as_python()
        assert "tensor.create" not in source, (
            "the internal create that used to hold the return buffer must be removed"
        )


class TestReturnStyleCodegen:
    """Codegen binds the return buffer to an external orchestration arg."""

    def test_orchestration_exports_return_buffer(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        program = _post_pass_program()
        entry = _entry(program)

        code = codegen.generate_orchestration(program, entry).code

        # The host must pass one more tensor (the output slot).
        assert ".expected_arg_count = 3" in code, "the generated entry must expect the appended output tensor"
        assert "orch_args.tensor(2)" in code, "the return buffer must bind to an external orchestration arg"
        # ...and the task writes into that external tensor, not a scope-local
        # allocation that dies with the SIMPLER_SCOPE. (The emit name is the
        # param's auto-name base: ``ret0__out`` -> ``ret0``.)
        assert "params_t0.add_output(ext_ret0)" in code
        assert "alloc_tensors" not in code


class TestReturnStyleArtifactMeta:
    """``ir.compile`` reports the appended param as a dispatchable output."""

    def test_compile_reports_output_indices(self):
        a, b = _tensors()

        compiled = return_style_add.compile(a, b)
        assert isinstance(compiled, CompiledProgram)

        assert compiled.has_return
        assert compiled.output_indices == [2], (
            "the artifact's dispatch metadata must expose the return-style output slot"
        )
        param_infos, _, _ = compiled._get_metadata()
        assert len(param_infos) == 3
        out_info = param_infos[2]
        assert out_info.direction == ir.ParamDirection.Out
        assert out_info.shape == [16, 16]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
