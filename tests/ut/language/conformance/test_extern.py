# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``@pl.jit.extern`` conformance on the tracing frontend.

A single-core stub traces as a header-only AIC/AIV ``ir.Function``; a
``mixed`` stub traces an AIC member + an AIV member and assembles the Group
wrapper at the call site (the ``@pl.program`` group route). These assert the
user-observable contract: the assembled program's functions, the
extern-source and include-dir attrs, and the generated kernel config. The
specializer's generated-source shape tests stay quarantined
(``tests/ut/jit/test_extern_kernel.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import backend
from pypto._external_source import EXTERNAL_INCLUDE_DIRS_ATTR
from pypto.backend import BackendType
from pypto.backend.pto_backend import generate as _generate
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import ir

_KERNEL_SRC = """
#include "kernel.h"
extern "C" void relu_kernel(const __gm__ float* in, __gm__ float* out) {
    int n = 64 * 64;
    for (int i = 0; i < n; ++i) out[i] = in[i] > 0 ? in[i] : 0;
}
"""


def _write_kernel(tmp_path: Path, name: str = "ext.cpp") -> Path:
    src = tmp_path / name
    src.write_text(_KERNEL_SRC)
    return src


@pytest.fixture(autouse=True)
def _setup_backend():
    """The pass pipeline verifies AIC/AIV functions against the configured
    backend (the jit suite's conftest does the same)."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _post_pass(entry, *args) -> ir.Program:
    program = entry.compile_for_test(*args)
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def _kernel_config(program: ir.Program, tmp_path: Path) -> str:
    files = _generate(program, str(tmp_path / "codegen"), skip_ptoas=True)
    return files["kernel_config.py"]  # path -> content


class TestSingleCoreExtern:
    def test_aiv_member_in_program(self, tmp_path):
        cpp = _write_kernel(tmp_path)

        @pl.jit.extern(core_type="aiv", source=cpp)
        def relu(
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]: ...

        @pl.jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            out = relu(a, out)
            return out

        torch = pytest.importorskip("torch")
        post = _post_pass(entry, torch.zeros(64, 64), torch.zeros(64, 64))

        # one AIV declaration, no Group / member split — and nothing else
        # leaks into the program (lazy global registration)
        assert {f.name for f in post.functions.values()} == {"entry", "relu"}
        relu_fn = post.get_function("relu")
        assert relu_fn.func_type == ir.FunctionType.AIV
        assert relu_fn.attrs["external_source"] == str(cpp.resolve())
        assert "relu_aic" not in {f.name for f in post.functions.values()}

    def test_aic_member_in_program(self, tmp_path):
        cpp = _write_kernel(tmp_path)

        @pl.jit.extern(core_type="aic", source=cpp)
        def gemm(
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]: ...

        @pl.jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            out = gemm(a, out)
            return out

        torch = pytest.importorskip("torch")
        post = _post_pass(entry, torch.zeros(64, 64), torch.zeros(64, 64))
        assert post.get_function("gemm").func_type == ir.FunctionType.AIC

    def test_include_dirs_reach_member_and_config(self, tmp_path):
        cpp = _write_kernel(tmp_path)
        include_dir = tmp_path / "include"
        include_dir.mkdir()

        @pl.jit.extern(core_type="aiv", source=cpp, include_dirs=[include_dir])
        def relu(
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]: ...

        @pl.jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            out = relu(a, out)
            return out

        torch = pytest.importorskip("torch")
        post = _post_pass(entry, torch.zeros(64, 64), torch.zeros(64, 64))
        expected = str(include_dir.resolve())
        assert json.loads(post.get_function("relu").attrs[EXTERNAL_INCLUDE_DIRS_ATTR]) == [expected]
        config = _kernel_config(post, tmp_path)
        assert f'"extra_include_dirs": {[expected]!r}' in config

    def test_relative_source_and_include_resolve_from_stub_file(self, tmp_path):
        # a real stub module file, so relative resolution has a base
        stub_dir = tmp_path / "kernels"
        stub_dir.mkdir()
        _write_kernel(stub_dir, "k.cpp")
        (stub_dir / "include").mkdir()
        stub = stub_dir / "stub.py"
        stub.write_text(
            "import pypto.language as pl\n\n"
            "@pl.jit.extern(core_type='aiv', source='k.cpp', include_dirs=['include'])\n"
            "def rel(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]])"
            " -> pl.Tensor[[64, 64], pl.FP32]: ...\n"
        )
        sys.path.insert(0, str(stub_dir))
        try:
            # a direct import puts the extern ir.Function in the entry's
            # globals — the adaptation surface (module-attribute callees,
            # ``stub.rel(...)``, are a known gap)
            from stub import rel  # noqa: PLC0415
        finally:
            sys.path.remove(str(stub_dir))

        @pl.jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            out = rel(a, out)
            return out

        torch = pytest.importorskip("torch")
        post = _post_pass(entry, torch.zeros(64, 64), torch.zeros(64, 64))
        rel_fn = post.get_function("rel")
        assert rel_fn.attrs["external_source"] == str((stub_dir / "k.cpp").resolve())
        assert json.loads(rel_fn.attrs[EXTERNAL_INCLUDE_DIRS_ATTR]) == [str((stub_dir / "include").resolve())]


class TestMixedExtern:
    def test_group_expansion(self, tmp_path):
        cpp = _write_kernel(tmp_path)

        @pl.jit.extern(core_type="mixed", aic_source=cpp, aiv_source=cpp)
        def pa(
            a: pl.Tensor[[128, 128], pl.FP16],
            out: pl.Out[pl.Tensor[[128, 128], pl.FP16]],
        ) -> pl.Tensor[[128, 128], pl.FP16]: ...

        @pl.jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            out = pa(a, out)
            return out

        torch = pytest.importorskip("torch")
        post = _post_pass(
            entry, torch.zeros(128, 128, dtype=torch.float16), torch.zeros(128, 128, dtype=torch.float16)
        )

        # AIC member + AIV member + the Group wrapper the entry dispatches
        names = {f.name for f in post.functions.values()}
        assert {"entry", "pa", "pa_aic", "pa_aiv"} <= names
        assert post.get_function("pa").func_type == ir.FunctionType.Group
        assert post.get_function("pa_aic").func_type == ir.FunctionType.AIC
        assert post.get_function("pa_aiv").func_type == ir.FunctionType.AIV
        assert post.get_function("pa_aic").attrs["external_source"] == str(cpp.resolve())
        assert post.get_function("pa_aiv").attrs["external_source"] == str(cpp.resolve())

    def test_include_dirs_reach_both_members(self, tmp_path):
        cpp = _write_kernel(tmp_path)
        first = tmp_path / "include_first"
        second = tmp_path / "include_second"
        first.mkdir()
        second.mkdir()
        expected = [str(first.resolve()), str(second.resolve())]

        @pl.jit.extern(
            core_type="mixed",
            aic_source=cpp,
            aiv_source=cpp,
            include_dirs=[first, second],
        )
        def pa(
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]: ...

        @pl.jit
        def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]) -> pl.Tensor:
            out = pa(a, out)
            return out

        torch = pytest.importorskip("torch")
        post = _post_pass(entry, torch.zeros(64, 64), torch.zeros(64, 64))
        for name in ("pa_aic", "pa_aiv"):
            assert json.loads(post.get_function(name).attrs[EXTERNAL_INCLUDE_DIRS_ATTR]) == expected
        config = _kernel_config(post, tmp_path)
        assert config.count(f'"extra_include_dirs": {expected!r}') == 2

    def test_call_outside_a_trace_is_loud(self, tmp_path):
        cpp = _write_kernel(tmp_path)

        @pl.jit.extern(core_type="mixed", aic_source=cpp, aiv_source=cpp)
        def pa(
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]: ...

        with pytest.raises(TypeError, match="outside a trace"):
            pa(1, 2)


class TestExternGuardrails:
    def test_bad_core_type(self, tmp_path):
        cpp = _write_kernel(tmp_path)
        with pytest.raises(ValueError, match="core_type"):

            @pl.jit.extern(core_type="cube", source=cpp)
            def k(
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]: ...

    def test_mixed_requires_both_sources(self, tmp_path):
        cpp = _write_kernel(tmp_path)
        with pytest.raises(ValueError, match="both aic_source"):

            @pl.jit.extern(core_type="mixed", aic_source=cpp)
            def k(
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]: ...

    def test_single_requires_source(self, tmp_path):
        with pytest.raises(ValueError, match="requires source"):

            @pl.jit.extern(core_type="aiv")
            def k(
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]: ...

    def test_source_not_found(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):

            @pl.jit.extern(core_type="aiv", source=tmp_path / "missing.cpp")
            def k(
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]: ...

    def test_include_dirs_rejects_scalar_path(self, tmp_path):
        cpp = _write_kernel(tmp_path)
        with pytest.raises(TypeError, match="sequence of paths"):

            @pl.jit.extern(core_type="aiv", source=cpp, include_dirs=str(tmp_path))
            def k(
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]: ...

    def test_include_dirs_rejects_missing_directory(self, tmp_path):
        cpp = _write_kernel(tmp_path)
        with pytest.raises(ValueError, match="directory not found"):

            @pl.jit.extern(core_type="aiv", source=cpp, include_dirs=[tmp_path / "nope"])
            def k(
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]: ...

    def test_dual_aiv_dispatch_requires_mixed(self, tmp_path):
        cpp = _write_kernel(tmp_path)
        with pytest.raises(ValueError, match="dual_aiv_dispatch"):

            @pl.jit.extern(core_type="aiv", source=cpp, dual_aiv_dispatch=True)
            def k(
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]: ...


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
