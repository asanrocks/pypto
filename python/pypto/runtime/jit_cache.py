# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""In-process specialization keys for @pl.jit.

The tracing ``pl.jit_trace`` and the AST specializer (``pl.jit``)
compile through the same ``ir.compile`` pipeline,
so "which call gets which artifact" is one contract, not two: every
artifact-affecting dimension must split the cache identically on both
lanes. This module owns the shared pieces:

- :func:`compile_option_identity` — the compile options that change the
  artifact (AUTO-scope analysis, memory planner, dbC, runtime ABI, Buffer
  IR, source locations), canonicalized. Every one of them is read per
  compile, so two ``PassContext``-wrapped calls that differ in any of them
  must never be handed the same artifact.
- :func:`run_config_compile_identity` — the full identity a
  ``RunConfig``-carrying call resolves to (RunConfig fields, active
  PassContext, environment).
- :class:`TensorKeyInfo` — the per-tensor signature component, with
  dynamic extents keyed as ``None`` so one artifact serves every concrete
  value.
- :func:`resolve_runtime_kind` / :func:`resolve_enable_buffer_ir` — the
  PassContext-only knobs ``ir.compile`` inherits.

Provenance (which sources and globals an artifact was built from) stays
lane-owned: the specializer folds it into its ``source_hash``, the tracing
jit bakes host values at trace time. Everything else about the key is
here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pypto._environment import explicit_env_request
from pypto.backend.pto_backend import emit_source_loc_default
from pypto.compile_profiling import get_active_profiler
from pypto.ir.compile import _validate_pass_context_conflicts
from pypto.ir.pass_manager import OptimizationStrategy, PassDumpLevel, coerce_dump_level
from pypto.pypto_core import DataType
from pypto.pypto_core.passes import (
    DiagnosticCheck,
    DiagnosticCheckSet,
    MemoryPlanner,
    PassContext,
    RuntimeKind,
    get_default_diagnostic_phase,
    get_default_verification_level,
    runtime_kind_to_name,
)

if TYPE_CHECKING:
    from pypto.pypto_core.ir import TensorLayout
    from pypto.runtime.runner import RunConfig


@dataclass(frozen=True)
class TensorKeyInfo:
    """Per-tensor component of a jit cache key.

    Attributes:
        name: Parameter name.
        shape: Shape tuple with None for dynamic dimensions.
        dtype: DataType of the tensor.
        layout: Annotated tensor layout, or None when the annotation omits it.
            Included because a layout may reach the annotation through a
            closure variable (``L = pl.NZ; ... pl.Tensor[[...], pl.FP32, L]``),
            leaving the source text — and thus a source hash — unchanged
            while the compiled artifact differs.
    """

    name: str
    shape: tuple[int | None, ...]
    dtype: DataType
    layout: TensorLayout | None = None


def resolve_runtime_kind() -> RuntimeKind:
    """Resolve the target Simpler runtime ABI for a cache key.

    Like the dbC flag, the runtime is selected by wrapping a call in
    ``with PassContext([], runtime=...)``, which ``ir.compile()`` inherits.
    ``RunConfig`` does not carry it, so the active context is the only
    source. Keying on it stops a ``host_build_graph`` call from reusing an
    artifact compiled for ``tensormap_and_ringbuffer``, whose
    ``kernel_config.py`` names a runtime no matching worker would bind.
    """
    ctx = PassContext.current()
    return ctx.get_runtime() if ctx is not None else RuntimeKind.TENSORMAP_AND_RINGBUFFER


def resolve_enable_software_pipeline(explicit: bool | None = None) -> bool:
    """Resolve the explicit option, active context, then the default for caching.

    Mirrors ``ir.compile()``'s own precedence: callers pass their already-
    resolved explicit value (the ``RunConfig`` field on the jit paths), the
    active ``PassContext`` answers when they did not, and the opt-in defaults
    to disabled.
    """
    if explicit is not None:
        return explicit
    ctx = PassContext.current()
    return ctx.get_enable_software_pipeline() if ctx is not None else False


def resolve_enable_buffer_ir(software_pipeline: bool = False) -> bool:
    """Resolve the staged Buffer IR option ``ir.compile()`` inherits.

    Mirrors ``ir.compile()``'s own precedence: an active ``PassContext`` owns
    the selection; without one, software pipelining lowers Tile IR, so its
    opt-in disables the default Buffer IR selection.
    """
    ctx = PassContext.current()
    if ctx is not None:
        return ctx.get_enable_buffer_ir()
    return not software_pipeline


def compile_option_identity(
    *,
    analyze_auto_scopes_for_deps: bool,
    memory_planner: MemoryPlanner | None,
    enable_pypto_l0c_double_buffer: bool,
    runtime: RuntimeKind,
    enable_buffer_ir: bool,
    emit_source_loc: bool,
    enable_software_pipeline: bool,
) -> tuple[Any, ...]:
    """The canonical compile-option tuple every jit cache key embeds.

    Each component changes the compiled artifact, so two calls differing
    in any of them must never share a cache entry. Callers pass
    already-resolved values — their resolution sources differ (RunConfig
    and PassContext for the tracing lane, compile kwargs for the
    specializer); :func:`run_config_compile_identity` resolves and calls
    this for the RunConfig-carrying path.

    The dbC flag is canonicalized against the planner: DSA_RP and PTOAS
    enable dbC=2 automatically, so the legacy-PyPTO opt-in is inert there
    and must not split the key.
    """
    effective_pypto_dbc = enable_pypto_l0c_double_buffer and memory_planner in (None, MemoryPlanner.PYPTO)
    return (
        ("analyze_auto_scopes_for_deps", analyze_auto_scopes_for_deps),
        ("memory_planner", None if memory_planner is None else str(memory_planner)),
        ("enable_pypto_l0c_double_buffer", effective_pypto_dbc),
        ("runtime", runtime_kind_to_name(runtime)),
        ("enable_buffer_ir", enable_buffer_ir),
        ("buffer_ir_version", 10 if enable_buffer_ir else 0),
        ("emit_source_loc", emit_source_loc),
        ("enable_software_pipeline", enable_software_pipeline),
    )


def run_config_compile_identity(run_config: RunConfig | None) -> tuple[Any, ...]:
    """The hashable compile-identity tuple of a jit call, for artifact caches.

    Every field here changes the compiled artifact (or which artifact type a
    call must receive), so a cache that reuses compiled programs must key on
    all of them — two calls that differ in any component must never share an
    entry. Diagnostic-only knobs (``dump_passes``, ``profiling``,
    ``output_dir``) are deliberately absent: they do not change the artifact
    and would only split the cache.

    ``run_config=None`` normalizes to the no-config defaults, but the
    PassContext-derived values (memory planner, L0C double-buffer, runtime
    ABI, Buffer IR) are still resolved — ``ir.compile()`` inherits them from
    the active context, so an ambient ``PassContext`` must split the cache
    exactly like an explicit one. ``emit_source_loc`` follows the
    ``PYPTO_EMIT_PTO_LOC`` environment for the same reason: codegen reads
    it per compile.
    """
    from pypto.runtime.runner import (  # noqa: PLC0415 — torch is a runner dep
        freeze_config_value,
        resolve_enable_pypto_l0c_double_buffer,
        resolve_memory_planner,
    )

    platform = run_config.platform if run_config is not None else None
    strategy = run_config.strategy if run_config is not None else OptimizationStrategy.Default
    distributed = (
        freeze_config_value(run_config.distributed_config)
        if run_config is not None and run_config.distributed_config is not None
        else None
    )
    analyze = run_config.analyze_auto_scopes_for_deps if run_config is not None else False
    return (
        platform,
        strategy,
        distributed,
        *compile_option_identity(
            analyze_auto_scopes_for_deps=analyze,
            memory_planner=resolve_memory_planner(run_config),
            enable_pypto_l0c_double_buffer=resolve_enable_pypto_l0c_double_buffer(),
            runtime=resolve_runtime_kind(),
            enable_buffer_ir=resolve_enable_buffer_ir(
                resolve_enable_software_pipeline(
                    run_config.enable_software_pipeline if run_config is not None else None
                )
            ),
            emit_source_loc=emit_source_loc_default(),
            enable_software_pipeline=resolve_enable_software_pipeline(
                run_config.enable_software_pipeline if run_config is not None else None
            ),
        ),
    )


def resolve_compile_request(run_config: RunConfig | None) -> tuple[dict[str, Any], bool]:
    """Resolve compiler arguments and whether this call requires fresh compilation.

    Shared by every jit entrance: both lanes compile through ``ir.compile``, so
    "which request runs the compiler" is one contract, not two. Diagnostics
    are requests to *run* the compiler, so the bypass is decided here, before
    any cache lookup — a warm artifact entry must never swallow a dump,
    profile, or output-directory request.

    Returns the ``ir.compile()`` keyword arguments (the canonical
    :meth:`RunConfig.compile_kwargs` mapping, with ``dump_passes`` coerced and
    ``emit_source_loc`` resolved) and whether the caller must compile fresh:

    - a diagnostic request: pass dumps (any non-NONE level), PTOAS dumps,
      compile profiling, an active profiler, an explicit output directory, a
      diagnostic phase, or disabled diagnostics;
    - an explicit environment request: an enabled ``PYPTO_EMIT_DEBUG_RUNNER``
      or ``PYPTO_REBUILD_FROM_PTO``. Placement-only settings
      (``PYPTO_PROG_BUILD_DIR``) move artifacts but must not force a recompile;
    - an outer :class:`~pypto.pypto_core.passes.PassContext` carrying
      instruments or non-default verification/diagnostic settings — those must
      actually run, and an ordinary planner/runtime context may reuse artifacts.

    A ``run_config=None`` call resolves against :class:`CompileOptions`
    defaults; callers that defer to ``ir.compile()``'s own defaults with no
    config still consult the bypass half.
    """
    if run_config is None:
        from pypto.runtime import CompileOptions  # noqa: PLC0415

        kwargs = CompileOptions().as_compile_kwargs()
    else:
        kwargs = run_config.compile_kwargs()
    kwargs["dump_passes"] = coerce_dump_level(kwargs["dump_passes"])
    kwargs["emit_source_loc"] = emit_source_loc_default()
    outer = _validate_pass_context_conflicts(
        operation="compile",
        verification_level=kwargs.get("verification_level"),
        diagnostic_phase=kwargs.get("diagnostic_phase"),
        memory_planner=kwargs.get("memory_planner"),
        runtime=kwargs.get("runtime"),
        enable_software_pipeline=kwargs.get("enable_software_pipeline"),
    )
    bypass = (
        kwargs["dump_passes"] is not PassDumpLevel.NONE
        or kwargs["dump_ptoas_passes"]
        or kwargs["profiling"]
        or get_active_profiler() is not None
        or kwargs.get("output_dir") is not None
        or explicit_env_request(os.environ.get("PYPTO_EMIT_DEBUG_RUNNER"))
        or explicit_env_request(os.environ.get("PYPTO_REBUILD_FROM_PTO"))
        or kwargs.get("verification_level") is not None
        or kwargs["diagnostic_phase"] is not None
        or kwargs["disabled_diagnostics"] is not None
    )
    if not bypass and outer is not None:
        # Match the pipeline defaults: ordinary planner/runtime contexts can
        # reuse artifacts, but instruments and custom checks must actually run.
        default_disabled = DiagnosticCheckSet()
        default_disabled.insert(DiagnosticCheck.UnusedControlFlowResult)
        bypass = (
            bool(outer.get_instruments())
            or outer.get_verification_level() != get_default_verification_level()
            or outer.get_diagnostic_phase() != get_default_diagnostic_phase()
            or outer.get_disabled_diagnostics() != default_disabled
        )
    return kwargs, bypass


__all__ = [
    "TensorKeyInfo",
    "compile_option_identity",
    "resolve_compile_request",
    "resolve_enable_buffer_ir",
    "resolve_runtime_kind",
    "run_config_compile_identity",
]
