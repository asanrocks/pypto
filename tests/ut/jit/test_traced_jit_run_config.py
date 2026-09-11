# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The tracing ``pl.jit`` RunConfig surface.

``config=RunConfig(...)`` on a host ``compile``/``__call__`` is compatibility
behavior ported from the quarantined AST specializer: compile-side fields
(``strategy``, ``dump_passes``, diagnostics, ...) forward to ``ir.compile()``,
runtime fields drive ``CompiledProgram.__call__``, and the compile identity
(platform, strategy, distributed config, planner, dbC flag) participates in
the dispatch cache key so two compile configurations never share an artifact
(design doc 03-entrance.md §5.4/§5.5). In-trace calls stay parser-aligned:
``config=`` is a host-call concept and is rejected inside a trace.
"""

import pypto.language as pl
import pytest
from pypto.ir.compiled_program import CompiledProgram
from pypto.pypto_core import passes
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime.runner import RunConfig


@pl.jit.incore
def _add_incore(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    tile_a = pl.load(a, [0, 0], [16, 16])
    tile_b = pl.load(b, [0, 0], [16, 16])
    pl.store(pl.add(tile_a, tile_b), [0, 0], c)
    return c


@pl.jit
def add_kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Tensor):
    return _add_incore(a, b, c)


def _tensors():
    torch = pytest.importorskip("torch")
    return (
        torch.zeros(16, 16, dtype=torch.float32),
        torch.zeros(16, 16, dtype=torch.float32),
        torch.empty(16, 16, dtype=torch.float32),
    )


# —— compile: config consumption and forwarding ——


def test_compile_consumes_config_kwarg():
    """``config=`` is consumed by the jit, never forwarded to the kernel —
    the compile succeeds instead of raising "unexpected keyword 'config'"."""
    a, b, c = _tensors()
    compiled = add_kernel.compile(a, b, c, config=RunConfig(platform="a2a3sim"))
    assert isinstance(compiled, CompiledProgram)


def test_compile_without_config_unchanged():
    a, b, c = _tensors()
    assert isinstance(add_kernel.compile(a, b, c), CompiledProgram)


def test_compile_rejects_non_run_config():
    a, b, c = _tensors()
    with pytest.raises(TypeError, match="config= must be a pypto.runtime.RunConfig"):
        add_kernel.compile(a, b, c, config="oops")


def test_compile_each_call_is_fresh():
    """Explicit compile-then-call freezes one materialization: no compile
    cache, so two calls return distinct artifacts (03-entrance.md §5.5)."""
    a, b, c = _tensors()
    first = add_kernel.compile(a, b, c)
    second = add_kernel.compile(a, b, c)
    assert first is not second


# —— __call__: the top-level config carve-out ——


def test_call_forwards_config_to_dispatch(monkeypatch):
    """The host call extracts ``config=`` and hands it to the dispatch lane
    (compile-side knobs at compile, runtime fields at ``CompiledProgram.__call__``).
    Dispatch itself needs a device, so the lane boundary is what is under
    test here."""
    a, b, c = _tensors()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        add_kernel, "_dispatch", lambda args, bridged, config=None: captured.setdefault("config", config)
    )
    cfg = RunConfig(platform="a2a3sim")
    add_kernel(a, b, c, config=cfg)
    assert captured["config"] is cfg


def test_call_rejects_unknown_kwargs():
    a, b, c = _tensors()
    with pytest.raises(TypeError, match="keyword arguments are not accepted"):
        add_kernel(a, b, c, foo=1)


def test_call_rejects_config_mixed_with_other_kwargs():
    """``config=`` is the only legal top-level keyword; pairing it with
    anything else stays under the parser-aligned rule."""
    a, b, c = _tensors()
    with pytest.raises(TypeError, match="keyword arguments are not accepted"):
        add_kernel(a, b, c, config=RunConfig(), foo=1)


def test_call_rejects_config_without_torch_dispatch():
    """A facade-only call returns the traced function and compiles
    nothing, so ``config=`` (a compile/dispatch concept) is a loud error."""
    x = pl.Tensor([16], pl.FP32, _annotation_only=True)
    with pytest.raises(TypeError, match="config= applies to compiled dispatch"):
        add_kernel(x, x, x, config=RunConfig())


def test_call_rejects_config_in_trace():
    """In-trace jit calls are parser-aligned (positional only, plus the
    ``device=``/``attrs=`` call-site overrides); ``config=`` is a host-call
    concept and must not silently pass through to the detached call."""

    with pytest.raises(TypeError, match="keyword arguments are not accepted"):

        @pl.function
        def caller(
            a: pl.Tensor[[16, 16], pl.FP32], b: pl.Tensor[[16, 16], pl.FP32]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return add_kernel(a, b, a, config=RunConfig())


# —— the dispatch cache key: compile identity participates ——


def _bridged():
    a, b, c = _tensors()
    bridged, _ = add_kernel._bridge_args((a, b, c))
    return bridged


def test_cache_key_splits_on_platform():
    bridged = _bridged()
    sim = add_kernel._cache_key(bridged, RunConfig(platform="a2a3sim"))
    board = add_kernel._cache_key(bridged, RunConfig(platform="a2a3"))
    assert sim != board


def test_cache_key_splits_on_pass_context_planner():
    """The planner usually arrives via a PassContext that never touches
    RunConfig; an ambient context must split the cache exactly like an
    explicit one (ir.compile inherits it either way)."""
    bridged = _bridged()
    with passes.PassContext([], memory_planner=MemoryPlanner.PTOAS):
        ptoas = add_kernel._cache_key(bridged, None)
    assert ptoas != add_kernel._cache_key(bridged, None)


def test_cache_key_splits_on_distributed_config():
    from pypto.ir.distributed_compiled_program import (  # noqa: PLC0415 - heavy import, this test only
        DistributedConfig,
    )

    bridged = _bridged()
    single = add_kernel._cache_key(bridged, None)
    distributed = add_kernel._cache_key(bridged, RunConfig(distributed_config=DistributedConfig()))
    assert single != distributed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
