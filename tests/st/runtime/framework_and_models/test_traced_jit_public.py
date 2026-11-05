# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device acceptance for the tracing jit's public execution path.

The corpus device suite (``tests/st/examples/02_intermediate/
test_minimal_corpus.py``) drives the st harness, which builds the traced IR
itself and compiles and dispatches outside the jit object — that proves the
traced IR, not the jit. This suite exercises the surface a user actually
calls: ``kernel(*tensors, config=...)`` and ``kernel.compile(...)``, through
the jit's own argument bridging, program assembly, compile-config
forwarding, artifact cache, and runtime-ABI dispatch, against the same
corpus entries and goldens.
"""

import pytest
import torch
from examples.beginner.elementwise import chunked_add
from examples.intermediate.fused_linear import fused_linear_relu, fused_matmul_bias
from examples.intermediate.softmax import tile_softmax
from pypto.language.tracing.decorators import jit_trace as jit_module

from examples.utils.cross_function_calls import main_kernel
from tests.ut.language.tracing.jit._traced_corpus_fixtures import annotated_add, controlled_accumulate


@pytest.fixture(autouse=True)
def _cache_live(monkeypatch, tmp_path):
    """Run with the artifact cache actually in play.

    The ST conftest redirects ir.compile() output through PYPTO_PROG_BUILD_DIR,
    which every jit lane treats as a request to run the compiler (never serve
    from cache). These tests assert reuse, so they clear the variable and
    redirect the default output base by changing directory instead — the same
    isolation the unit-tier jit conftest uses.
    """
    monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def compile_count(monkeypatch) -> dict[str, int]:
    """Count the jit's real compilations (the boundary the cache guards)."""
    counter = {"n": 0}
    real = jit_module._compile_program

    def counting(program, config=None):
        counter["n"] += 1
        return real(program, config)

    monkeypatch.setattr(jit_module, "_compile_program", counting)
    return counter


# ---------------------------------------------------------------------------
# Public dispatch: kernel(*tensors, config=...) — every corpus entry
# ---------------------------------------------------------------------------


class TestPublicDispatch:
    """Every corpus entry runs and matches its golden through the jit's own path."""

    def test_chunked_add(self, test_config):
        torch.manual_seed(0)
        a = torch.randn(512, 128, dtype=torch.float32)
        b = torch.randn(512, 128, dtype=torch.float32)
        c = torch.zeros(512, 128, dtype=torch.float32)

        chunked_add(a, b, c, config=test_config)

        assert torch.allclose(c, a + b, rtol=1e-5, atol=1e-5)

    def test_chunked_add_fp16(self, test_config):
        torch.manual_seed(0)
        a = torch.randn(512, 128, dtype=torch.float16)
        b = torch.randn(512, 128, dtype=torch.float16)
        c = torch.zeros(512, 128, dtype=torch.float16)

        chunked_add(a, b, c, config=test_config)

        assert torch.allclose(c, a + b, rtol=1e-3, atol=1e-3)

    def test_tile_softmax(self, test_config):
        torch.manual_seed(0)
        a = torch.randn(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        tile_softmax(a, out, config=test_config)

        assert torch.allclose(out, torch.softmax(a, dim=-1), rtol=1e-5, atol=1e-5)

    def test_fused_linear_relu(self, test_config):
        torch.manual_seed(0)
        x = torch.randn(64, 64, dtype=torch.float32) * 2.0  # both ReLU signs
        w = torch.randn(64, 64, dtype=torch.float32)
        bias = torch.randn(64, 64, dtype=torch.float32)
        y = torch.zeros(64, 64, dtype=torch.float32)

        fused_linear_relu(x, w, bias, y, config=test_config)

        assert torch.allclose(y, torch.relu(x @ w + bias), rtol=1e-3, atol=1e-3)

    def test_fused_matmul_bias(self, test_config):
        torch.manual_seed(0)
        a = torch.randn(64, 64, dtype=torch.float32)
        b = torch.randn(64, 64, dtype=torch.float32)
        bias = torch.randn(64, 64, dtype=torch.float32)
        c = torch.zeros(64, 64, dtype=torch.float32)

        fused_matmul_bias(a, b, bias, c, config=test_config)

        assert torch.allclose(c, a @ b + bias, rtol=1e-3, atol=1e-3)

    def test_main_kernel_inline_composition(self, test_config):
        torch.manual_seed(0)
        a = torch.randn(128, 128, dtype=torch.float32)
        c = torch.zeros(128, 128, dtype=torch.float32)

        main_kernel(a, c, config=test_config)

        assert torch.allclose(c, (a + 1.0) * 2.0, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("n", [0, 1, 3])
    @pytest.mark.parametrize("scale", [2.0, 3.0])
    def test_controlled_accumulate_matrix(self, test_config, n, scale):
        torch.manual_seed(0)
        x = torch.randn(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        controlled_accumulate(x, n, scale, out, config=test_config)

        acc = (n + 1) * x
        expected = acc * scale if n > 1 else acc - scale
        assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_controlled_accumulate_leading_tile_shape_variant(self, test_config):
        """A taller input reads the same leading [64, 64] tile — no dynamic-shape claim."""
        torch.manual_seed(0)
        x = torch.randn(128, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        controlled_accumulate(x, 3, 2.0, out, config=test_config)

        assert torch.allclose(out, 4 * x[:64, :64] * 2.0, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("value", [1.0, -2.0])
    def test_annotated_add(self, test_config, value):
        torch.manual_seed(0)
        x = torch.randn(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        annotated_add(x, value, out, config=test_config)

        assert torch.allclose(out, x + value, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Explicit compile: the returned handle speaks the runtime ABI
# ---------------------------------------------------------------------------


class TestExplicitCompileRuntimeAbi:
    """``kernel.compile(...)`` returns a handle whose signature is the runtime ABI."""

    def test_lazy_entry_host_knob_consumed_at_compile(self, test_config):
        torch.manual_seed(0)
        x = torch.randn(64, 64, dtype=torch.float32)
        out = torch.zeros(64, 64, dtype=torch.float32)

        impl = controlled_accumulate.compile(x, 3, 2.0, out, config=test_config)
        impl(x, 3, out)  # x, n, out — scale was consumed at compile
        assert torch.allclose(out, 4 * x * 2.0, rtol=1e-5, atol=1e-5)

        # One materialization, many dispatches: the handle reuses its artifact.
        out2 = torch.zeros(64, 64, dtype=torch.float32)
        impl(x, 0, out2)
        assert torch.allclose(out2, x - 2.0, rtol=1e-5, atol=1e-5)

    def test_annotated_entry_one_artifact_serves_every_scalar_value(self, test_config):
        torch.manual_seed(0)
        x = torch.randn(64, 64, dtype=torch.float32)

        impl = annotated_add.compile(x, 1.0, torch.zeros(64, 64), config=test_config)
        for value in (1.0, -2.0):
            out = torch.zeros(64, 64, dtype=torch.float32)
            impl(x, value, out)
            assert torch.allclose(out, x + value, rtol=1e-5, atol=1e-5)

    def test_composed_entry_compiles_and_dispatches(self, test_config):
        torch.manual_seed(0)
        a = torch.randn(128, 128, dtype=torch.float32)
        c = torch.zeros(128, 128, dtype=torch.float32)

        impl = main_kernel.compile(a, c, config=test_config)
        impl(a, c)
        assert torch.allclose(c, (a + 1.0) * 2.0, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Specialization on device: what reuses an artifact and what recompiles
# ---------------------------------------------------------------------------


class TestSpecializationOnDevice:
    """The cache contract, observed through real compilation counts."""

    def test_runtime_scalar_reuses_and_host_knob_specializes(self, test_config, compile_count):
        controlled_accumulate._cache.clear()

        torch.manual_seed(0)
        x = torch.randn(64, 64, dtype=torch.float32)

        def run(n, scale):
            out = torch.zeros(64, 64, dtype=torch.float32)
            controlled_accumulate(x, n, scale, out, config=test_config)
            acc = (n + 1) * x
            expected = acc * scale if n > 1 else acc - scale
            assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5)

        run(3, 2.0)
        assert compile_count["n"] == 1

        run(0, 2.0)
        run(1, 2.0)
        assert compile_count["n"] == 1, "the runtime scalar n must not recompile"

        run(3, 3.0)
        assert compile_count["n"] == 2, "the baked host knob scale must specialize"

    def test_annotated_scalar_value_does_not_split_the_artifact(self, test_config, compile_count):
        annotated_add._cache.clear()

        torch.manual_seed(0)
        x = torch.randn(64, 64, dtype=torch.float32)
        for value in (1.0, -2.0):
            out = torch.zeros(64, 64, dtype=torch.float32)
            annotated_add(x, value, out, config=test_config)
            assert torch.allclose(out, x + value, rtol=1e-5, atol=1e-5)

        assert compile_count["n"] == 1, "one artifact serves every scalar value"

    def test_dtype_splits_the_artifact(self, test_config, compile_count):
        chunked_add._cache.clear()

        for dtype, tol in ((torch.float32, 1e-5), (torch.float16, 1e-3)):
            torch.manual_seed(0)
            a = torch.randn(512, 128, dtype=dtype)
            b = torch.randn(512, 128, dtype=dtype)
            c = torch.zeros(512, 128, dtype=dtype)
            chunked_add(a, b, c, config=test_config)
            assert torch.allclose(c, a + b, rtol=tol, atol=tol)

        assert compile_count["n"] == 2, "a dtype change must compile a new artifact"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
