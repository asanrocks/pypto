# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lane-neutral compile-identity contract for both jit frontends.

Both the tracing ``pl.jit_trace`` and the quarantined AST specializer
(``pl.jit_specializer``) compile through ``ir.compile``. "Which call reuses
which artifact" is therefore one contract: every artifact-affecting dimension
must split both keys identically. ``pypto.runtime.jit_cache`` owns that
contract; this module pins it against the shared core, while
``tests/ut/jit/test_cache.py`` keeps the specializer-owned provenance.

Every dimension asserted here affects both lanes' artifacts, and both keys are
split by the shared core. Pinning an identity only in one frontend's suite
would leave the other frontend free to drift, so the sharable dimensions are
asserted against the shared core here; each frontend's own suite pins only the
provenance it adds on top.
"""

import pytest
from pypto.ir import DistributedConfig, OptimizationStrategy
from pypto.jit.cache import SCALAR_SEMANTICS, make_cache_key
from pypto.pypto_core import passes
from pypto.pypto_core.passes import MemoryPlanner, RuntimeKind
from pypto.runtime.jit_cache import (
    compile_option_identity,
    resolve_compile_request,
    resolve_enable_buffer_ir,
    resolve_enable_software_pipeline,
    resolve_runtime_kind,
    run_config_compile_identity,
)
from pypto.runtime.runner import (
    RunConfig,
    resolve_enable_pypto_l0c_double_buffer,
    resolve_memory_planner,
)


def _identity(
    *,
    analyze_auto_scopes_for_deps: bool = False,
    memory_planner: MemoryPlanner | None = None,
    enable_pypto_l0c_double_buffer: bool = False,
    runtime: RuntimeKind = RuntimeKind.TENSORMAP_AND_RINGBUFFER,
    enable_buffer_ir: bool = True,
    emit_source_loc: bool = True,
    enable_software_pipeline: bool = False,
) -> tuple[object, ...]:
    """Compile-option identity under defaults that mirror ``ir.compile``."""
    return compile_option_identity(
        analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
        memory_planner=memory_planner,
        enable_pypto_l0c_double_buffer=enable_pypto_l0c_double_buffer,
        runtime=runtime,
        enable_buffer_ir=enable_buffer_ir,
        emit_source_loc=emit_source_loc,
        enable_software_pipeline=enable_software_pipeline,
    )


class TestCompileOptionIdentity:
    """Each compile option that changes the artifact must split the identity."""

    def test_analyze_auto_scopes_for_deps_splits_identity(self):
        """AUTO-scope auto-deps changes generated code, so it must split."""
        assert _identity(analyze_auto_scopes_for_deps=False) != _identity(analyze_auto_scopes_for_deps=True)

    def test_memory_planner_splits_identity(self):
        """The planner changes placement and address ownership."""
        keys = [_identity(memory_planner=planner) for planner in (None, MemoryPlanner.PYPTO)]
        keys += [_identity(memory_planner=planner) for planner in (MemoryPlanner.DSA_RP, MemoryPlanner.PTOAS)]
        assert len(set(keys)) == len(keys), f"planner must split the identity, got {keys}"

    def test_dbc_double_buffer_flag_splits_legacy_pypto_key(self):
        """The legacy-PyPTO dbC=2 opt-in changes AutoTileMatmulL0/MemoryReuse."""
        off = _identity(memory_planner=MemoryPlanner.PYPTO, enable_pypto_l0c_double_buffer=False)
        on = _identity(memory_planner=MemoryPlanner.PYPTO, enable_pypto_l0c_double_buffer=True)
        assert off != on, "dbC=2 opt-in must split the identity"

    @pytest.mark.parametrize("planner", [MemoryPlanner.DSA_RP, MemoryPlanner.PTOAS])
    def test_dbc_double_buffer_flag_does_not_split_non_pypto_key(self, planner):
        """DSA_RP and PTOAS enable dbC=2 automatically, so the flag is inert."""
        off = _identity(memory_planner=planner, enable_pypto_l0c_double_buffer=False)
        on = _identity(memory_planner=planner, enable_pypto_l0c_double_buffer=True)
        assert off == on

    @pytest.mark.parametrize("planner", [MemoryPlanner.PYPTO, MemoryPlanner.DSA_RP, MemoryPlanner.PTOAS])
    def test_buffer_ir_splits_key_for_every_planner(self, planner):
        """Buffer IR changes storage legalization under every planner."""
        off = _identity(memory_planner=planner, enable_buffer_ir=False)
        on = _identity(memory_planner=planner, enable_buffer_ir=True)
        assert off != on
        assert ("buffer_ir_version", 10) in on
        assert ("buffer_ir_version", 0) in off

    def test_runtime_splits_key(self):
        """The runtime is baked into ``kernel_config.py`` and binds the worker."""
        tmrb = _identity(runtime=RuntimeKind.TENSORMAP_AND_RINGBUFFER)
        hbg = _identity(runtime=RuntimeKind.HOST_BUILD_GRAPH)
        assert tmrb != hbg, "runtime must split the identity"

    def test_emit_source_loc_splits_key(self):
        """Source locations change generated ``.pto`` text, hence the artifact."""
        assert _identity(emit_source_loc=True) != _identity(emit_source_loc=False)

    def test_equal_options_are_a_cache_hit(self):
        assert _identity() == _identity()
        assert len({_identity(), _identity()}) == 1

    def test_identity_is_hashable(self):
        assert {_identity(): "value"}[_identity()] == "value"


class TestRunConfigIdentity:
    """The full identity a ``RunConfig``-carrying call resolves to."""

    def test_different_platforms_cause_miss(self):
        """Same shapes compiled for different platforms must not collide."""
        sim = run_config_compile_identity(RunConfig(platform="a2a3sim"))
        device = run_config_compile_identity(RunConfig(platform="a2a3"))
        assert sim != device

    def test_same_platform_is_cache_hit(self):
        assert run_config_compile_identity(RunConfig(platform="a2a3sim")) == run_config_compile_identity(
            RunConfig(platform="a2a3sim")
        )

    def test_none_platform_differs_from_named_platform(self):
        """``run_config=None`` must not collide with an explicit platform."""
        assert run_config_compile_identity(None) != run_config_compile_identity(RunConfig(platform="a2a3sim"))

    def test_distributed_config_in_key(self):
        """``distributed_config`` is baked into the artifact and its dispatch."""

        def identity(config):
            return run_config_compile_identity(RunConfig(platform="a2a3", distributed_config=config))

        none = identity(None)
        pair_01 = identity(DistributedConfig(device_ids=[0, 1]))
        pair_23 = identity(DistributedConfig(device_ids=[2, 3]))
        pair_01_again = identity(DistributedConfig(device_ids=[0, 1]))

        assert len({none, pair_01, pair_23}) == 3, "device_ids must split the identity"
        assert pair_01 == pair_01_again, "equal config must be a cache hit"

    def test_same_strategy_is_cache_hit(self):
        strategy = OptimizationStrategy.Default
        assert run_config_compile_identity(
            RunConfig(platform="a2a3sim", strategy=strategy)
        ) == run_config_compile_identity(RunConfig(platform="a2a3sim", strategy=strategy))

    def test_strategy_splits_identity(self):
        """The strategy is a distinct identity slot, not a constant."""
        default = RunConfig(platform="a2a3sim", strategy=OptimizationStrategy.Default)
        other = RunConfig(platform="a2a3sim", strategy=OptimizationStrategy.Default)
        other.strategy = "custom"  # pyright: ignore[reportAttributeAccessIssue]
        assert run_config_compile_identity(default) != run_config_compile_identity(other)

    def test_identity_with_strategy_is_hashable(self):
        key = run_config_compile_identity(RunConfig(strategy=OptimizationStrategy.Default))
        assert {key: "value"}[key] == "value"

    def test_run_config_field_wins_over_default(self):
        config = RunConfig(platform="a2a3", memory_planner=MemoryPlanner.PTOAS)
        assert resolve_memory_planner(config) == MemoryPlanner.PTOAS
        assert ("memory_planner", str(MemoryPlanner.PTOAS)) in run_config_compile_identity(config)

    def test_unset_run_config_field_defers_to_context(self):
        config = RunConfig(platform="a2a3")
        assert config.memory_planner is None
        with passes.PassContext([], memory_planner=MemoryPlanner.PTOAS):
            assert resolve_memory_planner(config) == MemoryPlanner.PTOAS
            assert ("memory_planner", str(MemoryPlanner.PTOAS)) in run_config_compile_identity(config)


class TestResolveRuntime:
    """The runtime the key reads must match the one ``ir.compile`` will use."""

    def test_defaults_to_tensormap_and_ringbuffer(self):
        assert resolve_runtime_kind() == RuntimeKind.TENSORMAP_AND_RINGBUFFER

    def test_reads_the_active_pass_context(self):
        # The runtime is PassContext-only — RunConfig does not carry it — so the
        # context is the sole source the cache key can consult.
        with passes.PassContext([], runtime=RuntimeKind.HOST_BUILD_GRAPH):
            assert resolve_runtime_kind() == RuntimeKind.HOST_BUILD_GRAPH
        assert resolve_runtime_kind() == RuntimeKind.TENSORMAP_AND_RINGBUFFER  # pyright: ignore[reportArgumentType]


class TestResolveMemoryPlanner:
    """The planner the key reads must match the one ``ir.compile`` will use."""

    def test_defaults_to_pypto(self):
        assert resolve_memory_planner(None) == MemoryPlanner.PYPTO

    @pytest.mark.parametrize("planner", [MemoryPlanner.DSA_RP, MemoryPlanner.PTOAS])
    def test_reads_the_active_pass_context(self, planner):
        """The planner usually arrives through ``PassContext``, not RunConfig."""
        with passes.PassContext([], memory_planner=planner):
            assert resolve_memory_planner(None) == planner
        assert resolve_memory_planner(None) == MemoryPlanner.PYPTO


def test_software_pipeline_cache_option_follows_active_context():
    assert resolve_enable_software_pipeline() is False
    assert resolve_enable_software_pipeline(True) is True
    with passes.PassContext([], enable_buffer_ir=False, enable_software_pipeline=True):
        assert resolve_enable_software_pipeline() is True
        with passes.PassContext([]):
            assert resolve_enable_software_pipeline() is False
        assert resolve_enable_software_pipeline() is True
    assert resolve_enable_software_pipeline() is False


def test_software_pipeline_opt_in_disables_default_buffer_ir_without_context(monkeypatch):
    """``ir.compile()`` leaves Buffer IR on unless software pipelining lowers Tile IR.

    The suite runs inside an ambient ``PassContext``, so the no-context branch
    needs the resolver's context source stubbed out.
    """

    class _NoPassContext:
        @staticmethod
        def current() -> None:
            return None

    from pypto.runtime import jit_cache  # noqa: PLC0415

    monkeypatch.setattr(jit_cache, "PassContext", _NoPassContext)
    assert resolve_enable_buffer_ir() is True
    assert resolve_enable_buffer_ir(resolve_enable_software_pipeline(True)) is False


def test_buffer_ir_cache_option_follows_active_context():
    assert resolve_enable_buffer_ir() is True
    with passes.PassContext([], enable_buffer_ir=False):
        assert resolve_enable_buffer_ir() is False
        with passes.PassContext([]):
            assert resolve_enable_buffer_ir() is True
        assert resolve_enable_buffer_ir() is False
    assert resolve_enable_buffer_ir() is True


class TestResolveEnablePyptoL0cDoubleBuffer:
    """The legacy-PyPTO dbC=2 opt-in must match what ``ir.compile`` inherits."""

    def test_defaults_to_off(self):
        assert resolve_enable_pypto_l0c_double_buffer() is False

    def test_reads_the_active_pass_context(self):
        """The flag is PassContext-only; keying on RunConfig would miss it."""
        with passes.PassContext([], enable_pypto_l0c_double_buffer=True):
            assert resolve_enable_pypto_l0c_double_buffer() is True
        assert resolve_enable_pypto_l0c_double_buffer() is False


class TestSpecializerUsesSharedIdentity:
    """The AST specializer's key must embed exactly the shared identity."""

    def test_compile_opts_end_with_the_shared_identity(self):
        key = make_cache_key(
            source_hash="abc",
            param_names=[],
            tensor_shapes={},
            tensor_dtypes={},
            dynamic_dims=set(),
            analyze_auto_scopes_for_deps=True,
            memory_planner=MemoryPlanner.DSA_RP,
            enable_pypto_l0c_double_buffer=True,
            runtime=RuntimeKind.HOST_BUILD_GRAPH,
            enable_buffer_ir=False,
            emit_source_loc=False,
        )
        assert key.compile_opts is not None
        shared = _identity(
            analyze_auto_scopes_for_deps=True,
            memory_planner=MemoryPlanner.DSA_RP,
            enable_pypto_l0c_double_buffer=True,
            runtime=RuntimeKind.HOST_BUILD_GRAPH,
            enable_buffer_ir=False,
            emit_source_loc=False,
        )
        assert key.compile_opts == (("scalar_semantics", SCALAR_SEMANTICS), ("dep_layouts", ())) + shared

    def test_specializer_key_splits_where_the_shared_identity_splits(self):
        def key(**options):
            return make_cache_key(
                source_hash="abc",
                param_names=[],
                tensor_shapes={},
                tensor_dtypes={},
                dynamic_dims=set(),
                **options,
            )

        assert key() != key(runtime=RuntimeKind.HOST_BUILD_GRAPH)
        assert key() != key(memory_planner=MemoryPlanner.PTOAS)
        assert key() != key(enable_buffer_ir=False)
        assert key() != key(emit_source_loc=False)


class TestResolveCompileRequestEnvPolicy:
    """An explicit environment request bypasses the cache; a disabled flag does not.

    The bypass half decides whether a call must run the compiler. ``PYPTO_EMIT_DEBUG_RUNNER``
    and ``PYPTO_REBUILD_FROM_PTO`` follow the shared environment policy: only an *enabled*
    spelling is a request, so an explicitly disabled flag keeps serving warm artifacts.
    ``PYPTO_PROG_BUILD_DIR`` selects placement only and must never force a recompile.
    The suite runs inside an ambient instrumented ``PassContext`` (which correctly
    bypasses on its own), so these tests stub the resolver's context source to reach
    the no-context branch the environment policy lives in.
    """

    @pytest.fixture(autouse=True)
    def _no_outer_context(self, monkeypatch):
        """Stub the outer-context source the bypass check reads.

        The suite runs inside an ambient instrumented ``PassContext`` (which
        correctly bypasses on its own — an instrumented context must actually
        run). The environment policy under test lives in the no-context
        branch, so the conflict validator — the one place the resolver
        consults the outer context — reports none here.
        """

        from pypto.runtime import jit_cache  # noqa: PLC0415

        monkeypatch.setattr(jit_cache, "_validate_pass_context_conflicts", lambda **_kwargs: None)

    @staticmethod
    def _clean_env(monkeypatch):
        monkeypatch.delenv("PYPTO_EMIT_DEBUG_RUNNER", raising=False)
        monkeypatch.delenv("PYPTO_REBUILD_FROM_PTO", raising=False)
        monkeypatch.delenv("PYPTO_PROG_BUILD_DIR", raising=False)

    def test_no_environment_request_serves_the_cache(self, monkeypatch):
        self._clean_env(monkeypatch)
        _, bypass = resolve_compile_request(None)
        assert bypass is False

    @pytest.mark.parametrize("value", ["0", "false", "no"])
    @pytest.mark.parametrize("name", ["PYPTO_EMIT_DEBUG_RUNNER", "PYPTO_REBUILD_FROM_PTO"])
    def test_a_disabled_flag_keeps_the_warm_cache(self, monkeypatch, name, value):
        self._clean_env(monkeypatch)
        monkeypatch.setenv(name, value)
        _, bypass = resolve_compile_request(None)
        assert bypass is False

    @pytest.mark.parametrize("value", ["1", "true", ""])
    @pytest.mark.parametrize("name", ["PYPTO_EMIT_DEBUG_RUNNER", "PYPTO_REBUILD_FROM_PTO"])
    def test_an_enabled_flag_compiles_fresh(self, monkeypatch, name, value):
        self._clean_env(monkeypatch)
        monkeypatch.setenv(name, value)
        _, bypass = resolve_compile_request(None)
        assert bypass is True

    def test_a_build_directory_selects_placement_without_bypassing(self, monkeypatch, tmp_path):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("PYPTO_PROG_BUILD_DIR", str(tmp_path))
        _, bypass = resolve_compile_request(None)
        assert bypass is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
