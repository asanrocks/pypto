# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared PTO runtime-scope and structural host constructs for PyPTO."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pypto.ir.builder import IRBuilder
    from pypto.language.protocol import TraceContext
    from pypto.language.typing import Tensor

from pypto.language.diagnostics import ParserSyntaxError
from pypto.language.typing import Array, Scalar, Tensor
from pypto.pypto_core import ir as _ir

from ..authoring import ConstructContext, construct
from .optimizations import CrossCoreSlot, Optimization, Split


def _explicit_dump_attrs(dumps: list[Tensor] | None, owner: str) -> list[tuple[str, object]]:
    """Build the ``dump_vars`` attr list from an explicit ``dumps=`` kwarg.

    The mirror of the AST parser's ``_parse_at_dumps_kwarg`` attrs arm, shared
    by the scope kinds whose only printed surface for a ``dump_vars`` attr is
    this kwarg (``pl.graph`` / ``pl.cluster``; ``pl.at`` collects its own and
    also takes ``no_dep_args=``). Entries must be staged (outer-scope) facade
    values; ``unwrap`` yields the Var the parser's name lookup would resolve.
    Returns ``[]`` for an empty list so callers treat it as a no-op.
    """
    if not dumps:
        return []
    for t in dumps:
        if not hasattr(t, "unwrap"):
            raise TypeError(f"{owner}(dumps=[...]) entries must be staged (outer-scope) values, got {t!r}")
    return [("dump_vars", [t.unwrap() for t in dumps])]


def _merge_sticky_dump_vars(
    attrs: list[tuple[str, object]] | None,
    tagged: list[_ir.Var],
) -> list[tuple[str, object]] | None:
    """Merge forward-sticky ``pl.dump_tag`` tensors into scope attrs."""
    if not tagged:
        return attrs
    new_attrs: list[tuple[str, object]] = list(attrs) if attrs else []
    for i, (key, value) in enumerate(new_attrs):
        if key == "dump_vars":
            assert isinstance(value, list)
            existing = cast(list[_ir.Var], value)
            merged = list(existing)
            seen = {id(var) for var in merged}
            for var in tagged:
                if id(var) not in seen:
                    merged.append(var)
                    seen.add(id(var))
            new_attrs[i] = ("dump_vars", merged)
            return new_attrs
    insert_at = next(
        (i for i, (key, _) in enumerate(new_attrs) if key in {"task_id_var", "slot_num"}),
        len(new_attrs),
    )
    new_attrs.insert(insert_at, ("dump_vars", list(tagged)))
    return new_attrs


class ScopeMode(Enum):
    """Dependency-tracking mode of a PTO runtime scope (``SIMPLER_SCOPE``).

    - ``AUTO``: OverlapMap auto-dependency tracking is on (``SIMPLER_SCOPE()``).
    - ``MANUAL``: auto tracking is off; the user declares every edge via
      ``pl.submit(..., deps=[...])`` (``SIMPLER_SCOPE(SIMPLER_SCOPE(ScopeMode::MANUAL))``).
    """

    AUTO = 0
    MANUAL = 1


def _reject_nested_manual_scope(ctx: ConstructContext) -> None:
    """Reject a MANUAL runtime scope opened inside another MANUAL one.

    The runtime forbids the nesting, so the tracing lane enforces it where the
    parser does: at the scope's opening. The parser counts
    ``_manual_scope_depth``; here the open-scope mirror answers the same
    question, which also covers an in-trace expansion whose body opens a manual
    scope (it borrows the parent builder, so it sees the enclosing scope)."""
    if ctx.is_inside_manual_scope():
        raise ParserSyntaxError(
            "a manual scope may not be nested inside another manual scope",
            hint="Flatten the nested manual scope into the enclosing one, or move it outside.",
        )


@construct
def scope(ctx: ConstructContext, mode: ScopeMode = ScopeMode.AUTO):
    """Context manager marking a PTO runtime scope (``SIMPLER_SCOPE``) region.

    A PTO runtime scope is a resource-management and dependency-tracking boundary in
    the simpler runtime: it bounds OverlapMap auto-dependency tracking and gives
    a per-scope HeapRing level (nested scopes reclaim memory independently). The
    simpler runtime provides an implicit top-level scope, so writing scopes is
    a **tuning / control** mechanism, never a correctness requirement.

    By default the compiler inserts AUTO scopes for you (function body + each
    ``for`` / ``if`` body). To place scopes yourself, set
    ``@pl.function(auto_scope=False)`` and use this context manager (and the
    ``pl.range(..., scope=...)`` sugar). See :class:`ScopeMode` for AUTO vs
    MANUAL.

    Usage::

        with pl.scope():                          # AUTO
            out = self.kernel(a, b, out)

        with pl.scope(mode=pl.ScopeMode.MANUAL):  # MANUAL — user owns deps
            out, tid = pl.submit(self.stage1, x, out)

    Rules:
      - Must appear inside an Orchestration function (not InCore).
      - ``mode=AUTO`` is only allowed under ``@pl.function(auto_scope=False)``
        (in the default ``auto_scope=True`` the compiler owns AUTO placement).
      - ``mode=MANUAL`` is allowed in either mode (it is a dependency-semantics
        choice, not ring tuning).
      - A MANUAL scope may not nest inside another MANUAL scope (runtime
        forbids).
      - AUTO scope may not nest inside a MANUAL scope (runtime forbids).
    """
    if ctx.binding_name is not None:
        raise TypeError("with pl.scope(...) as ...: is not supported — a runtime scope captures no value")
    if mode == ScopeMode.AUTO and ctx.auto_scope_enabled():
        raise ParserSyntaxError(
            "a hand-placed AUTO 'with pl.scope()' requires @pl.function(auto_scope=False)"
        )
    if mode == ScopeMode.MANUAL:
        _reject_nested_manual_scope(ctx)
    elif mode == ScopeMode.AUTO and ctx.is_inside_manual_scope():
        raise ParserSyntaxError(
            "an AUTO scope may not be nested inside a manual scope",
            hint="The runtime forbids AUTO scope nested in MANUAL scope; move it outside.",
        )
    region = ctx.open_scope(_ir.ScopeKind.Runtime, attrs=[], defer=False, manual=(mode == ScopeMode.MANUAL))
    yield None
    ctx.exit(region)


#: Function types whose body is a device-side kernel, where a Graph region (a
#: topology of task *launches*) cannot live — mirrors the AST parser's
#: ``_GRAPH_FORBIDDEN_FUNC_TYPES``.
_GRAPH_FORBIDDEN_FUNC_TYPES = frozenset(
    {
        _ir.FunctionType.InCore,
        _ir.FunctionType.AIC,
        _ir.FunctionType.AIV,
        _ir.FunctionType.Group,
        _ir.FunctionType.Spmd,
    }
)


@construct
def graph(ctx: ConstructContext, name: str, *, dumps: list[Tensor] | None = None):
    """Mark a repeated region of orchestration as one recordable graph.

    Under ``runtime="host_build_graph"`` the runtime records the region's task
    topology the first time it executes and replays the recording on every
    later execution. A decoder stack of N structurally identical layers
    therefore costs one graph build instead of N.

    ``pl.graph`` is the in-place form of ``@pl.jit.graph``: OutlineGraphScopes
    extracts the region into a Graph function named after ``name``, so the two
    surfaces compile to the same thing.

    The recorded topology is fixed after the first execution: only tensor
    addresses and boundary scalars are refreshed per replay. LegalizeGraphBoundary
    proves that statically and rejects at compile time whatever it cannot prove.

    Args:
        name: Region name. Must be a valid Python identifier; it becomes the
            outlined function's name and hence the runtime's graph key, so keep
            it stable across edits.
        dumps: Optional list of outer-scope tensors to mark for selective tensor
            dump. The graph scope's ``dump_vars`` surface — symmetric with
            ``pl.at(dumps=...)`` and the print -> reparse spelling for a set
            ``pl.dump_tag`` attached at scope entry.

    Returns:
        Context manager for the Graph scope.

    Examples:
        >>> for layer in pl.range(40):
        ...     with pl.graph("decoder_layer"):
        ...         h = attention(x, wq, h)
        ...         x = mlp(h, w1, x)
    """
    # The ctor pins its user boundary to one non-empty string literal: the
    # name is the graph key, so no path auto-generates one (an auto name
    # would silently change whenever an unrelated region was added).
    # ``type(...) is not str`` rather than ``isinstance``: the annotation
    # already promises ``str``, so the check exists only for untyped
    # callers — and an exact-type test keeps it honest under strict typing.
    if type(name) is not str:
        raise TypeError(
            f"pl.graph() region name must be a string literal, got {name!r} ({type(name).__name__})"
        )
    if not name:
        raise ValueError(
            "pl.graph() region name must not be empty — the name becomes the "
            "recorded graph's symbol: 'with pl.graph(\"decoder_layer\"):'"
        )
    if ctx.binding_name is not None:
        raise TypeError(
            "with pl.graph(...) as ...: is not supported — a Graph scope captures no producer TaskId"
        )
    func_type = ctx.function_type()
    if func_type in _GRAPH_FORBIDDEN_FUNC_TYPES:
        raise ParserSyntaxError(
            f"pl.graph() is not valid inside a {func_type.name} function",
            hint="A Graph region records orchestration task launches, so it belongs in the "
            "@pl.jit entry (or another @pl.jit.graph function), not in a kernel body. To make a "
            "whole function a Graph, decorate it with @pl.jit.graph instead.",
        )
    for device_kind, construct_name in (
        (_ir.ScopeKind.InCore, "pl.at(level=...)"),
        (_ir.ScopeKind.Hierarchy, "pl.at(level=..., role=...)"),
        (_ir.ScopeKind.Cluster, "pl.cluster()"),
        (_ir.ScopeKind.Spmd, "pl.spmd(...)"),
    ):
        if ctx.is_inside_scope(device_kind):
            raise ParserSyntaxError(
                f"pl.graph() cannot be nested inside `{construct_name}`",
                hint=(
                    "A Graph region records a topology of task launches, so it must sit "
                    "around the dispatches rather than inside one. Move the "
                    "`with pl.graph(...):` outside."
                    if device_kind is not _ir.ScopeKind.Hierarchy
                    else "Graph regions inside a distributed hierarchy region are not "
                    "supported yet. Move the `with pl.graph(...):` outside."
                ),
            )
    if ctx.is_inside_scope(_ir.ScopeKind.Graph):
        raise ParserSyntaxError(
            "pl.graph() cannot be nested inside another pl.graph()",
            hint="The host_build_graph runtime cannot record a graph inside a graph. Mark "
            "either the outer region or the inner one, not both.",
        )
    region = ctx.open_scope(
        _ir.ScopeKind.Graph,
        attrs=_merge_sticky_dump_vars(_explicit_dump_attrs(dumps, "pl.graph"), ctx.sticky_dump_vars()) or [],
        defer=False,
        name_hint=name,
    )
    yield None
    ctx.exit(region)


@construct
def manual_scope(ctx: ConstructContext):
    """Alias for ``pl.scope(mode=pl.ScopeMode.MANUAL)``.

    Turns OverlapMap auto dep-tracking off for a region.

    Inside this block, the simpler runtime skips OverlapMap lookup and insert
    for every kernel submit, so the user takes full responsibility for
    declaring task-to-task ordering edges. ``manual_scope`` is the
    coarsest-grained of the runtime's auto-dep-tracking opt-outs; finer
    granularities are available and compose with auto scope:

      - ``with pl.manual_scope():`` — this construct; whole-region opt-out.
      - ``pl.create_tensor(..., manual_dep=True)`` — opt out a single tensor
        for its entire lifetime (any task referencing it skips OverlapMap).
      - ``pl.no_dep(arg)`` at a kernel-call arg position — opt out a single
        tensor for a single task only.

    Manual dependency edges (``pl.submit(..., deps=[...])``) are **orthogonal**
    to all of the above: the runtime adds them on top of whatever auto-tracked
    deps remain (final fanin = auto ∪ explicit), so ``deps=`` works in auto
    scope too. Use ``manual_scope`` only when you want full ownership of the
    dep graph; otherwise stay in auto scope and use ``pl.submit(..., deps=)``
    as a precision tool that patches the edges auto cannot infer.

    Usage::

        # Full-manual region.
        with pl.manual_scope():
            scratch, tid = pl.submit(self.stage1, x, scratch)
            out, _       = pl.submit(self.stage2, scratch, out, deps=[tid])

        # Auto scope with explicit-edge patching (no manual_scope needed).
        a, a_tid = pl.submit(self.k1, x)
        b, _     = pl.submit(self.k2, x, deps=[a_tid])

    Restrictions:
      - Must appear inside an Orchestration function (not InCore).
      - Cannot be nested inside another ``manual_scope`` (runtime forbids).
    """
    if ctx.binding_name is not None:
        raise TypeError("with pl.manual_scope() as ...: is not supported — a runtime scope captures no value")
    _reject_nested_manual_scope(ctx)
    region = ctx.open_scope(_ir.ScopeKind.Runtime, attrs=[], defer=False, manual=True)
    yield None
    ctx.exit(region)


class reg_scope:
    """Implicitly capture inputs for a non-nested A5 register region.

    Register and local scalar values cannot escape; store results can.
    """

    def __enter__(self) -> None:
        raise RuntimeError("pl.reg_scope() is only valid inside a parsed PyPTO function")

    def __exit__(self, *args: object) -> None:
        pass


@construct
def cluster(ctx: ConstructContext, *, name_hint: str = "", dumps: list[Tensor] | None = None):
    """Mark a region of code as belonging to a Cluster execution context.

    A cluster groups co-scheduled AIC (Cube) and AIV (Vector) kernels that
    share the same physical cluster resources. The OutlineClusterScopes pass
    extracts Cluster scopes into separate Group-typed functions.

    Args:
        name_hint: Optional name hint for the outlined function.
        dumps: Optional list of outer-scope tensors to mark for selective tensor
            dump — the cluster scope's ``dump_vars`` surface and its print ->
            reparse spelling (symmetric with ``pl.at(dumps=...)``).

    Returns:
        Context manager for Cluster scope

    Examples:
        >>> with pl.cluster():
        ...     with pl.at(level=pl.Level.CORE_GROUP):
        ...         y = pl.add(x, x)
    """
    if ctx.binding_name is not None:
        raise TypeError(
            "with pl.cluster(...) as ...: is not supported — a Cluster scope captures no producer TaskId"
        )
    region = ctx.open_scope(
        _ir.ScopeKind.Cluster,
        attrs=_merge_sticky_dump_vars(_explicit_dump_attrs(dumps, "pl.cluster"), ctx.sticky_dump_vars())
        or [],
        defer=False,
        name_hint=name_hint,
    )
    yield None
    ctx.exit(region)


def _split_spmd_for_loop_name_hints(name_hint: str) -> tuple[str, str]:
    """Map one ``for i in pl.spmd(..., name_hint=...)`` hint to Spmd vs InCore
    names — the mirror of the AST parser's helper of the same name (the stdlib
    may not import the parser, since the parser imports the stdlib). The
    for-form auto-wraps an inner ``InCoreScopeStmt``; the outer
    ``SpmdScopeStmt`` hint is consumed by ``OutlineClusterScopes`` (``_spmd_``
    suffix on the wrapper).

    - ``name_hint="q_proj"`` → Spmd ``q_proj_spmd``, InCore ``q_proj``
    - ``name_hint="q_proj_spmd"`` → Spmd ``q_proj_spmd``, InCore ``q_proj``
    """
    if not name_hint:
        return "", ""
    if name_hint.endswith("_spmd"):
        return name_hint, name_hint[: -len("_spmd")]
    return f"{name_hint}_spmd", name_hint


def _fold_optimization_entries(
    optimizations: Sequence[Optimization] | None,
    owner: str,
    *,
    split_gate: bool | None,
) -> tuple[_ir.SplitMode | None, int | None]:
    """Fold ``optimizations=[...]`` entries to the split mode and the
    cross-core slot count — the tracing mirror of the AST parser's
    ``_parse_optimizations_list`` (``pl.at`` and ``pl.spmd`` share the entry
    contract: the parser routes both through one helper, and so does the
    tracing side).

    At most one ``pl.split(...)`` and one ``pl.cross_core_slot(...)``; the
    deprecated ``pl.split(..., slot_num=N)`` and ``pl.cross_core_slot(slot_num=N)``
    both feed the same ``slot_num`` scope attr, so stating both is the loud
    "sets the slot count twice" error regardless of entry order. The entry
    constructors already reject non-positive counts.

    Args:
        optimizations: The ``optimizations=`` entries.
        owner: API name used in error messages (``"pl.at"`` / ``"pl.spmd"``).
        split_gate: Whether a split entry is legal here — ``True``/``False``
            gates on the ``pl.at`` level (``None`` never gates, for ``pl.spmd``
            whose nested InCore always carries the split).

    Returns:
        ``(split_mode, slot_num)`` for the scope's ``split=`` field and
        ``slot_num`` scope attr.
    """
    split_mode: _ir.SplitMode | None = None
    slot_num: int | None = None
    seen_split = False
    seen_cross_core_slot = False
    # Whether slot_num came from the deprecated pl.split(slot_num=) spelling,
    # so a list carrying both sources for one value is rejected instead of
    # letting entry order decide (mirrors the parser's tracker).
    slot_num_from_split = False
    for opt in optimizations or ():
        if isinstance(opt, Split):
            if split_gate is False:
                raise TypeError("split mode is only supported with level=pl.Level.CORE_GROUP")
            if seen_split:
                raise TypeError(
                    f"Duplicate pl.split(...) entries in {owner}(optimizations=[...]) "
                    "— at most one split mode per scope"
                )
            seen_split = True
            split_mode = opt.mode
            if opt.slot_num is not None:
                if seen_cross_core_slot:
                    raise TypeError(
                        f"{owner}(optimizations=[...]) sets the cross-core slot count "
                        "twice: via the deprecated pl.split(slot_num=...) and via "
                        "pl.cross_core_slot(...) — keep only pl.cross_core_slot(slot_num=N)"
                    )
                slot_num = opt.slot_num  # pl.split(mode, slot_num=N) → scope attr
                slot_num_from_split = True
        elif isinstance(opt, CrossCoreSlot):
            if type(opt.slot_num) is not int or opt.slot_num <= 0:
                # The constructor validates, but the dataclass is directly
                # constructible — keep the parser's positivity check at the fold.
                raise TypeError(f"pl.cross_core_slot() requires a positive ring depth, got {opt.slot_num!r}")
            if seen_cross_core_slot:
                raise TypeError(
                    f"Duplicate pl.cross_core_slot(...) entries in {owner}(optimizations=[...]) "
                    "— at most one ring depth per scope"
                )
            if slot_num_from_split:
                raise TypeError(
                    f"{owner}(optimizations=[...]) sets the cross-core slot count "
                    "twice: via the deprecated pl.split(slot_num=...) and via "
                    "pl.cross_core_slot(...) — drop slot_num= from pl.split(...) "
                    "and keep pl.cross_core_slot(slot_num=N)"
                )
            seen_cross_core_slot = True
            slot_num = opt.slot_num  # pl.cross_core_slot(slot_num=N) → scope attr
        else:
            raise TypeError(
                f"Unsupported entry in {owner}(optimizations=[...]) — entries "
                f"must be pl.split(...) or pl.cross_core_slot(...), got {opt!r}"
            )
    return split_mode, slot_num


def _parse_spmd_optimizations(
    optimizations: Sequence[Optimization] | None,
) -> tuple[_ir.SplitMode | None, int | None]:
    """Parse ``pl.spmd(optimizations=[pl.split(mode), pl.cross_core_slot(slot_num=N)])``.

    Delegates to the shared entry fold — the mirror of the AST parser's
    ``_parse_spmd_optimizations_list``, which accepts the same entries as
    ``pl.at`` (the nested InCore region always carries the split, so no
    level gate applies). Returns ``(split_mode, split_slot_num)`` for the
    inner ``InCoreScopeStmt`` (``split=`` field + ``slot_num`` scope attr).
    """
    return _fold_optimization_entries(optimizations, "pl.spmd", split_gate=None)


def _pin_spmd_core_num(core_num: object) -> int | _ir.Expr:
    """Pin ``pl.spmd(...)``'s ``core_num`` to ``int | Scalar | ir.Expr``.

    A staged Scalar reaches the SPMD head at the operand boundary — unwrap
    it to its ir.Expr so the trace-protocol object owns the plain ``int |
    ir.Expr`` core_num it was designed for (the printed round-trip form
    ``pl.spmd(pl.const(8, pl.INDEX) // ...)`` re-parses through this same
    unwrap).
    """
    if isinstance(core_num, Scalar):
        return core_num.unwrap()
    if isinstance(core_num, bool) or not isinstance(core_num, (int, _ir.Expr)):
        raise ValueError(f"core_num must be a positive integer or ir.Expr, got {core_num!r}")
    if isinstance(core_num, int) and core_num <= 0:
        raise ValueError(f"core_num must be a positive integer, got {core_num!r}")
    return core_num


def _spmd_submit_only_scope_attrs(
    allow_early_resolve: bool, predicate: Scalar | Tensor | None
) -> list[tuple[str, Any]]:
    """The Submit-only ``pl.spmd`` kwargs, in canonical order — shared by both forms.

    ``allow_early_resolve`` / ``predicate`` ride on the ``SpmdScopeStmt``
    until the Spmd outliner moves them onto the synthesised ``Submit``.
    Both the with-form (its recipe) and the for-form (``begin``) build their
    scope attrs through this one helper, so a kwarg added here reaches both
    spellings."""
    attrs: list[tuple[str, Any]] = []
    if allow_early_resolve:
        attrs.append(("allow_early_resolve", True))
    if predicate is not None:
        if not hasattr(predicate, "unwrap"):
            raise TypeError(f"pl.spmd(predicate=...) must be a staged value, got {predicate!r}")
        attrs.append(("predicate", predicate.unwrap()))
    return attrs


def _reject_spmd_submit_only_kwargs_in_cluster(
    ctx: ConstructContext,
    dep_edges: list[_ir.Var],
    allow_early_resolve: bool,
    predicate: Scalar | Tensor | None,
) -> None:
    """Reject Submit-only ``pl.spmd()`` kwargs on a cluster-nested scope (both forms).

    Mirrors the parser's ``_reject_spmd_submit_only_kwargs_in_cluster``
    (``deps=`` first, the parser's report order): a cluster-nested pl.spmd
    is unwrapped into the Group function by ``OutlineClusterScopes`` and
    never lowers to a ``Submit``, so any of the three kwargs would be
    silently lost. Without the trace-time guard the attrs reach the pass
    pipeline and fail there as an internal error (``UnwrapNestedSpmd``
    re-asserts) — late and far less diagnosable."""
    if not ctx.is_inside_scope(_ir.ScopeKind.Cluster):
        return
    if dep_edges:
        kwarg, lost = "deps=[...]", "the dependency edges"
    elif allow_early_resolve:
        kwarg, lost = "allow_early_resolve=True", "the early-dispatch hint"
    elif predicate is not None:
        kwarg, lost = "predicate=(...)", "the dispatch predicate"
    else:
        return
    raise TypeError(
        f"`pl.spmd(..., {kwarg})` cannot be nested inside `pl.cluster()` — "
        "a cluster-nested pl.spmd is unwrapped into the Group function and never "
        f"produces a Submit, so {lost} would be lost. Use a standalone "
        f"`with pl.spmd(..., {kwarg}):` (implicit cluster) to keep it."
    )


def _spmd_block_idx_expr() -> _ir.Expr:
    """The per-block index read — the for-form carrier's prologue value."""
    from pypto.language.op import tile_ops  # noqa: PLC0415 - avoid the op-module cycle

    return tile_ops.get_block_idx().unwrap()


@construct
def _spmd_scope(
    ctx: ConstructContext,
    core_num: int | Scalar | _ir.Expr,
    *,
    sync_start: bool,
    name_hint: str,
    optimizations: Sequence[Optimization] | None,
    deps: Sequence[Scalar | Array | None] | None,
    allow_early_resolve: bool,
    predicate: Scalar | Tensor | None,
):
    """The ``pl.spmd`` lowering — one recipe for both statement spellings.

    Both spellings share the whole body: the ``SpmdScopeStmt`` with its attrs
    in canonical order (``manual_dep_edges``, ``task_id_var``,
    ``allow_early_resolve``, ``predicate``), then the nested InCore carrier
    whose begin is deferred so the body's leading ``pl.set_cache_policy(...)``
    declarations join its attrs. They differ only at the edges:

    * the **with-form** may capture the dispatch TaskId (``as tid`` → the
      ``task_id_var`` attr and the capture binding) and decides the carrier's
      open trigger from the body shape — a split hint, a cross-core slot, or
      a *captured* body that is not one call-valued statement wraps the
      dispatch body in an InCore scope opening at the body's first statement;
      the uncaptured form keeps the deferred carrier, opening dynamically on
      the body's per-block-index read (a dispatch body stays bare).
    * the **for-form** captures nothing, always wraps, and mints the loop
      variable bound to ``pl.tile.get_block_idx()`` as the carrier's deferred
      prologue — its first statement — so the representative body reads the
      index through the loop variable.
    """
    split_mode, split_slot_num = _parse_spmd_optimizations(optimizations)
    core = _pin_spmd_core_num(core_num)
    for_form = ctx.spelling == "for"
    # Form-surface discipline, mirroring the parser: ``deps=`` is accepted
    # on every form — an uncaptured scope still lowers to a Submit, the
    # Spmd outliner synthesising the unused TaskId Var exactly as
    # ``pl.at(..., deps=[...])`` does — while the Submit-only kwargs
    # (``deps=`` included) are rejected inside ``pl.cluster()``, and so is
    # ``as tid`` itself (nothing cluster-nested lowers to a Submit).
    dep_edges = ctx.lower_deps(deps, "pl.spmd") if deps else []
    _reject_spmd_submit_only_kwargs_in_cluster(ctx, dep_edges, allow_early_resolve, predicate)
    # The for-form's binding name is the loop target — never a capture.
    as_name = None if for_form else ctx.binding_name
    loop_var = ctx.loop_var() if for_form else None
    if as_name and ctx.is_inside_scope(_ir.ScopeKind.Cluster):
        raise TypeError(
            "`with pl.spmd(...) as tid:` cannot capture a TaskId when nested "
            "inside `pl.cluster()` — a cluster-nested pl.spmd is unwrapped into "
            "the Group function and never produces a Submit. Use a standalone "
            "`with pl.spmd(...) as tid:` (implicit cluster) to capture the "
            "dispatch TaskId."
        )
    # The wrap decision (see the docstring): the deferred carrier's open
    # trigger. A wrapped carrier opens at the body's first emission; the plain
    # one waits for the block-index read. The for-form always wraps — its
    # body is the representative iteration of an outlined per-block kernel.
    incore_wrapped = for_form or (
        split_mode is not None
        or split_slot_num is not None
        or (as_name is not None and not ctx.single_call_body)
    )
    scope_attrs: list[tuple[str, Any]] = []
    if dep_edges:
        scope_attrs.append(("manual_dep_edges", dep_edges))
    as_binding = None
    if as_name:
        # the TaskId-capture channel (mirrors AtContext.__trace_enter__ /
        # the AST's _parse_spmd_scope_with_tid): the kit creates the Var,
        # emits the placeholder, and hands back the deferred binding proxy.
        as_binding, tid_attr = ctx.capture_task_id(as_name)
        scope_attrs.append(tid_attr)
    scope_attrs.extend(_spmd_submit_only_scope_attrs(allow_early_resolve, predicate))
    # The split name hints are pessimistic for the uncaptured form: the
    # carrier may still open on the block-index read, and the Spmd scope's
    # name is fixed at begin — a bare uncaptured body keeps the user's
    # unsplit hint only when it is one call-valued statement (the shape
    # that can never open the carrier).
    if incore_wrapped or not ctx.single_call_body:
        spmd_name_hint, incore_name_hint = _split_spmd_for_loop_name_hints(name_hint)
    else:
        spmd_name_hint = name_hint
        incore_name_hint = name_hint
    incore_attrs: list[tuple[str, Any]] | None = None
    if split_slot_num is not None:
        incore_attrs = [("slot_num", split_slot_num)]
    # Forward-sticky dump tags attach to the scope that owns the outlined
    # dispatch: the InCore wrapper when wrapped, else the Spmd scope.
    if incore_wrapped:
        incore_attrs = _merge_sticky_dump_vars(incore_attrs, ctx.sticky_dump_vars())
    else:
        scope_attrs = _merge_sticky_dump_vars(scope_attrs, ctx.sticky_dump_vars()) or []

    spmd_region = ctx.open_scope(
        _ir.ScopeKind.Spmd,
        attrs=scope_attrs,
        defer=False,
        core_num=core,
        sync_start=sync_start,
        name_hint=spmd_name_hint,
    )
    carrier = ctx.open_carrier(
        _ir.ScopeKind.InCore,
        attrs=incore_attrs or [],
        # The carrier owns the body-decided open, so it — not the dispatch
        # scope around it — is the recipe's structural surface.
        primary=True,
        flush_on_emit=incore_wrapped,
        split=split_mode,
        name_hint=incore_name_hint,
        # The for-form's deferred prologue: ``i = pl.tile.get_block_idx()``
        # is the carrier's first statement (built here for span parity,
        # emitted at open).
        prologue=None if loop_var is None else [(loop_var, _spmd_block_idx_expr())],
    )
    yield loop_var if for_form else as_binding
    if (
        not for_form
        and carrier._carrier_pending
        and carrier._cache_decls
        and not carrier._carrier_flush_on_emit
    ):
        raise ParserSyntaxError(
            "pl.set_cache_policy() has nothing to attach to on this Spmd scope",
            hint=(
                "This `pl.spmd(...)` body dispatches a pre-defined kernel, so the GM "
                "reads happen inside that callee, not here. Declare the policy in the "
                "callee — a `with pl.at(level=pl.Level.CORE_GROUP):` scope in its "
                "body, or pl.load(..., cache=pl.CachePolicy.BYPASS) on the access "
                "itself. An *inline* pl.spmd body does accept the marker: it is "
                "carried by the InCore scope the body is outlined into."
            ),
        )
    ctx.exit(carrier)
    ctx.exit(spmd_region)


class spmd:
    """Dispatch a kernel with SPMD (Single Program Multiple Data) multi-block execution.

    The first argument is the number of blocks and is positional — mirroring
    ``range(n)``. Loop start is fixed at 0 and step at 1; each block gets an
    index ``i`` in ``[0, core_num)``.

    Usage forms:

    1. ``with pl.spmd(n):`` — body is either a single call to a pre-defined
       InCore kernel (direct dispatch), or an inline multi-statement block that
       is auto-outlined into a synthetic InCore kernel (like the loop form, minus
       the auto-bound index). An inline body must read the per-block index via
       ``pl.tile.get_block_idx()`` — without it every block runs identical work.
       Captures no producer TaskId (use form 3 for that). Can stand alone
       (implicit cluster) or nest inside ``pl.cluster()``.

    2. ``for i in pl.spmd(n):`` — loop-style. The iteration variable binds
       the per-block index (equivalent to ``pl.tile.get_block_idx()``); the
       body is auto-outlined into a synthetic InCore function, so inline
       tile/tensor ops work without a separate ``@pl.function(type=InCore)``
       declaration.

    3. ``with pl.spmd(n, deps=[...]) as tid:`` — same body shapes as form 1, and
       additionally captures the grid dispatch's producer ``Scalar[TASK_ID]`` in
       ``tid`` (mirroring ``with pl.at(...) as tid:``), usable as a ``deps=`` edge
       on later tasks, stored into a ``pl.array.create(N, pl.TASK_ID)``, or
       crossing into ``pl.manual_scope``. TaskId capture is the only thing this
       adds over form 1 — it is orthogonal to the inline body.

    Optional ``optimizations=[pl.split(mode)]`` applies to the inner InCore scope
    (auto-generated for the for-form and the ``as tid`` form, wrapped around the
    call for the plain with-form).

    Args:
        core_num: Number of blocks for SPMD dispatch. Positional; accepts a
            Python ``int`` or any ``ir.Expr`` of integer type. Closure-captured
            integer constants and closure arithmetic are folded to ``ConstInt``
            by the parser and ``Simplify``; non-foldable expressions flow
            through to codegen unchanged.
        sync_start: If True, all blocks start execution simultaneously (default: False).
        name_hint: Optional name hint for the outlined function.
        optimizations: Optional list literal containing only ``pl.split(mode)``
            entries (the parser inspects the AST).
        deps: Optional explicit producer-edge list (TaskId Vars and/or ``None``
            sentinels), accepted only with the ``as tid`` form. Lowered to the
            outlined Submit's ``manual_dep_edges``; codegen packs it into a
            ``set_dependencies(...)`` invocation (union'd with auto-deps).
        allow_early_resolve: Opt the grid dispatch in as a speculative
            early-dispatch producer (simpler#1065). Same hint as
            ``pl.submit(..., allow_early_resolve=True)`` / ``pl.at(...,
            allow_early_resolve=True)``: the scheduler may pre-stage this task's
            consumers before it completes. Forces the dispatch to lower to an
            ``ir.Submit`` (even without ``as tid``) so the flag rides to codegen,
            where it emits ``Arg::set_allow_early_resolve(true)``. Pure scheduling
            hint. Rejected on a ``pl.cluster()``-nested ``pl.spmd`` (such a scope
            is unwrapped into the Group function and never produces a Submit, so
            the hint would be lost).
        predicate: Optional **dispatch predicate** — a single comparison of one
            tensor element against an integer literal, e.g.
            ``predicate=(row_count[e] > 0)``. Same surface, validation and
            lowering as ``pl.spmd_submit(..., predicate=...)``: the scheduler
            evaluates it at the dispatch point (after this scope's dependencies
            are satisfied, so the value is current) and retires the task inline —
            never dispatching it to a core — when it is false, while still
            settling fanin/fanout so downstream consumers unlock. The comparison
            is parsed but **never evaluated** in orchestration; reading it there
            would stall on ``wait_for_tensor_ready``, exactly what the predicate
            avoids. Like ``allow_early_resolve`` it forces the dispatch to lower
            to an ``ir.Submit`` (even without ``as tid``) and is rejected on a
            ``pl.cluster()``-nested ``pl.spmd``.

            **Contract:** the operand tensor's producing task must be one of
            ``deps=`` — otherwise the predicate may read a stale value. ``deps=``
            is only available on the ``as tid`` form, so a predicate over a
            tensor produced in the same function needs that form. The parser
            makes a best-effort check (see ``pl.spmd_submit``); getting ``deps=``
            right remains the author's responsibility.

    Returns:
        Context manager / loop iterator for the SPMD scope.

    Examples:
        >>> # Single-kernel context-manager form (direct dispatch)
        >>> with pl.spmd(4):
        ...     out = self.kernel(a, b, out)
        >>>
        >>> # Inline context-manager form — no TaskId; read the block index yourself
        >>> with pl.spmd(4):
        ...     i = pl.tile.get_block_idx()
        ...     offset = i * 128
        ...     tile_a = pl.load(a, [offset, 0], [128, 128])
        ...     tile_b = pl.load(b, [offset, 0], [128, 128])
        ...     out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
        >>>
        >>> # Loop form — body runs per-block with i = tile.get_block_idx()
        >>> for i in pl.spmd(4):
        ...     offset = i * 128
        ...     tile_a = pl.load(a, [offset, 0], [128, 128])
        ...     tile_b = pl.load(b, [offset, 0], [128, 128])
        ...     out = pl.store(pl.add(tile_a, tile_b), [offset, 0], out)
        >>>
        >>> # Capture-form: inline body + producer TaskId for explicit dep wiring
        >>> with pl.spmd(4, name_hint="stage1") as tid_a:
        ...     i = pl.tile.get_block_idx()
        ...     offset = i * 128
        ...     out = pl.store(pl.add(a[offset], b[offset]), [offset, 0], out)
        >>> with pl.spmd(4, name_hint="stage2", deps=[tid_a]) as tid_b:
        ...     i = pl.tile.get_block_idx()
        ...     out2 = pl.store(pl.relu(out[i * 128]), [i * 128, 0], out2)
        >>>
        >>> # With-form with split hint on the inner InCore wrapper
        >>> with pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        ...     out = self.kernel(a, b, out)
        >>>
        >>> # SPMD inside cluster (mixed kernel)
        >>> with pl.cluster():
        ...     with pl.spmd(4, sync_start=True):
        ...         out = self.kernel(a, b, out)
        >>>
        >>> # Dispatch predicate — skip the expert entirely when its row count is 0
        >>> with pl.spmd(1) as gate_tid:
        ...     row_count = self.gate(row_count)
        >>> with pl.spmd(4, deps=[gate_tid], predicate=(row_count[0, 0] > 0)) as tid:
        ...     out = self.expert(x, out)
    """

    def __init__(
        self,
        core_num: int | Scalar | _ir.Expr,
        *,
        sync_start: bool = False,
        name_hint: str = "",
        optimizations: Sequence[Optimization] | None = None,
        deps: Sequence[Scalar | Array | None] | None = None,
        allow_early_resolve: bool = False,
        predicate: Scalar | Tensor | None = None,
    ) -> None:
        core = _pin_spmd_core_num(core_num)
        self.core_num = core
        self.sync_start = sync_start
        self.name_hint = name_hint
        self.optimizations = optimizations
        self.deps = deps
        self.allow_early_resolve = allow_early_resolve
        self.predicate = predicate
        # The active recipe instance — the one lowering in either spelling
        # (``_spmd_scope``). Created at the trace hook, single-use; the
        # structural probes (``_carrier_pending``, ``_pending_begin``, ...)
        # forward to it through ``__getattr__`` so the Frame sees the same
        # answers either spelling gives.
        self._recipe: Any = None

    def _recipe_instance(self) -> Any:
        """One fresh single-use recipe instance carrying this call's arguments."""
        return _spmd_scope(
            self.core_num,
            sync_start=self.sync_start,
            name_hint=self.name_hint,
            optimizations=self.optimizations,
            deps=self.deps,
            allow_early_resolve=self.allow_early_resolve,
            predicate=self.predicate,
        )

    # —— the traced protocol (both spellings forward to the recipe) ——

    def __trace_enter__(
        self, ctx: TraceContext, as_name: str | None = None, single_call_body: bool = False
    ) -> Any:
        """Enter the with-form: the recipe instance owns the whole lowering."""
        self._recipe = self._recipe_instance()
        return self._recipe.__trace_enter__(ctx, as_name, single_call_body)

    def __trace_iter__(self, ctx: TraceContext) -> Scalar:
        """Enter the for-form: the recipe instance drives from the target binding."""
        self._recipe = self._recipe_instance()
        return self._recipe.__trace_iter__(ctx)

    def begin(
        self,
        builder: IRBuilder,
        name: str,
        already_in_incore: bool = False,
        sticky_dump_vars: list[_ir.Var] | None = None,
    ) -> _ir.Var:
        """The ``_ForDescriptor`` hook: forward to the recipe's for-arm."""
        return self._recipe.begin(builder, name, already_in_incore, sticky_dump_vars)

    def __trace_exit__(self, ctx: TraceContext) -> object:
        """The close hook (either spelling): the recipe's post-yield code runs here."""
        return self._recipe.__trace_exit__(ctx)

    # —— the deferred-scope structural surface (forwarded; must sit on the
    # class because the runtime-checkable protocol isinstance probes with
    # getattr_static, which bypasses __getattr__) ——

    def open_deferred_scope(self, builder: IRBuilder) -> None:
        """Open the carrier — its first statement (or its close) arrived."""
        self._recipe.open_deferred_scope(builder)

    def open_pending_carrier(self, builder: IRBuilder) -> None:
        """Open the carrier — the body read the per-block index."""
        self._recipe.open_pending_carrier(builder)

    def attach_cache_policy(self, var: _ir.Var, policy: int) -> None:
        """Buffer one ``pl.set_cache_policy`` declaration on the pending carrier."""
        self._recipe.attach_cache_policy(var, policy)

    def __getattr__(self, name: str) -> Any:
        """Forward the structural probes to the active recipe instance.

        While a spelling runs, the deferred carrier and its state live on the
        recipe's primary region; the Frame probes *this* construct for them
        (``_carrier_pending``, ``_carrier_flush_on_emit``, ``_pending_begin``,
        the recipe's constructor-surface facts). Before a hook runs there is
        no instance, so every probe falls through as absent — the same answer
        an ordinary construct without the attribute gives."""
        active = self.__dict__.get("_recipe")
        if active is None:
            raise AttributeError(name)
        return getattr(active, name)

    # —— host guards (the traced protocol is the only entrance) ——

    def __enter__(self) -> Any:
        # The static contract is the traced ``as tid`` binding — the
        # ``Scalar[TASK_ID]`` the carrier protocol mints — not the raising
        # host path (a NoReturn here would mark kernel bodies unreachable).
        raise RuntimeError(
            "pl.spmd(...) with-form is only valid inside a decorated "
            "@pl.program / @pl.function body (the tracing frontend routes it through "
            "the __trace_enter__ trace-protocol hook). If you're seeing this, the "
            "surrounding function was not decorated."
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    def __iter__(self) -> Iterator[Scalar]:
        # Lets `for i in pl.spmd(...)` type-check and parse at the Python
        # level; the yielded type is the traced loop variable — a
        # ``Scalar[INDEX]`` bound to ``pl.tile.get_block_idx()`` (see
        # ``begin``). The @pl.program / @pl.function decorators intercept
        # the AST, so this method should never actually run — if it does,
        # the caller is using pl.spmd() outside the DSL interception path.
        # Raise loudly rather than silently yielding zero iterations.
        raise RuntimeError(
            "pl.spmd(...) loop form is only valid inside a @pl.program / "
            "@pl.function body; the parser replaces the for-loop with an "
            "SpmdScopeStmt. If you're seeing this, the surrounding function "
            "was not decorated."
        )


def _split_aiv_lane_idx_expr() -> _ir.Expr:
    """The AIV lane index read — the split_aiv region's prologue value."""
    from pypto.language.op import tile_ops  # noqa: PLC0415 - avoid the op-module cycle

    return tile_ops.get_subblock_idx().unwrap()


@construct
def _split_aiv_scope(ctx: ConstructContext, mode: _ir.SplitMode):
    """The ``for aiv_id in pl.split_aiv(...)`` lowering — an immediate-open for-recipe.

    Opens the ``ScopeKind.SplitAiv`` region with the selected split mode and
    binds the loop variable to ``pl.tile.get_subblock_idx()`` as the region's
    prologue. A top-level form synthesizes an InCore wrapper first (keeping
    the region eligible for ``OutlineIncoreScopes`` — else it has no enclosing
    InCore to outline) with the forward-sticky dump tags merged onto it; a
    form already inside InCore — or inside an InCore-typed function body,
    which *is* the region — emits in place. The lowering pass consumes this
    region before code generation.

    A function-level AUTO split (``optimizations=[pl.split(...)``) and this
    region are mutually exclusive AIV-split mechanisms (RFC #1820) —
    downstream lowering takes the per-region path and would silently drop the
    function-level split. **Any** ``pl.split(...)`` is rejected,
    ``SplitMode.NONE`` included: the literal NONE spelling is only observable
    here (the builder's user-split mirror), never in the IR.
    """
    user_split = ctx.enclosing_user_split()
    if user_split is not None:
        raise ParserSyntaxError(
            f"scope combines a function-level pl.split(pl.SplitMode.{user_split.name}) "
            "(optimizations=[pl.split(...)]) with a pl.split_aiv region; these are mutually "
            "exclusive AIV-split mechanisms — the function-level split would be silently "
            "dropped (the per-region split governs the lanes)",
            hint="Remove optimizations=[pl.split(...)] or the pl.split_aiv region. To pin a "
            "custom cross-core slot count, use optimizations=[pl.cross_core_slot(slot_num=N)], "
            "which is orthogonal to splitting.",
        )
    loop_var = ctx.loop_var()
    already_in_incore = ctx.is_inside_incore() or ctx.function_type() == _ir.FunctionType.InCore
    wrapper = None
    if not already_in_incore:
        wrapper = ctx.open_scope(
            _ir.ScopeKind.InCore,
            attrs=_merge_sticky_dump_vars(None, ctx.sticky_dump_vars()) or [],
            defer=False,
        )
    region = ctx.open_scope(
        _ir.ScopeKind.SplitAiv,
        attrs=[],
        defer=False,
        split=mode,
        prologue=[(loop_var, _split_aiv_lane_idx_expr())],
    )
    yield loop_var
    ctx.exit(region)
    if wrapper is not None:
        ctx.exit(wrapper)


class split_aiv:
    """Open an explicit AIV-split region as an SPMD-style loop.

    Usage::

        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            ...  # body runs per AIV lane; aiv_id = pl.tile.get_subblock_idx()

    The loop builds a first-class :class:`~pypto.pypto_core.ir.SplitAivScopeStmt`
    region carrying the requested ``SplitMode``. Because it is a structural node
    (not a whole-InCore-scope flag), it is **nestable**: the region may appear
    inside a ``pl.range`` / ``pl.pipeline`` loop or an ``if``, and sibling regions
    may carry **different** modes (multi-mode), each lowered independently. A
    top-level ``for aiv_id in pl.split_aiv(...)`` is wrapped in an enclosing
    ``InCore`` scope so OutlineIncoreScopes can outline it. The loop variable
    binds the AIV lane index (equivalent to ``pl.tile.get_subblock_idx()``).

    Two dispatch modes:

    * **Data-parallel** (``UP_DOWN`` / ``LEFT_RIGHT``): the region's vector
      compute is **halved** on the split axis (rows / cols), so each lane
      processes one half of every tile.
    * **Task-parallel** (``NONE``): **no halving** — both lanes run the *full*
      body for disjoint work the author dispatches via ``aiv_id`` (e.g. an
      ``aiv_id``-strided loop). Use this when the tiles cannot be halved
      (unit dims) or a reduction must stay full-width. ``tile.aiv_shard`` /
      ``tile.aic_gather`` are rejected here — there is no split axis.

    The region survives parse -> SSA -> ResolveBackendOpLayouts as a structural
    node (printer emits ``for aiv_id in pl.split_aiv(...):`` so parse->print->parse
    is a fixpoint), then is lowered by LowerAutoVectorSplit (pass 20) and erased by
    ExpandMixedKernel; it never reaches codegen.

    Args:
        n: The AIV sub-core count. Positional; hardware-fixed at 2 (the two AIV
            lanes of one AICore). Every other value is rejected by the parser.
        mode: Required dispatch mode, no silent default. ``pl.SplitMode.NONE``
            (task-parallel, no halving), ``pl.SplitMode.UP_DOWN`` or
            ``pl.SplitMode.LEFT_RIGHT`` (data-parallel halving).

    Returns:
        Loop iterator for the explicit AIV-split region.

    Examples:
        >>> for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        ...     offset = aiv_id * 128
        ...     t = pl.load(a, [offset, 0], [128, 128])
        ...     out = pl.store(t, [offset, 0], out)
    """

    def __init__(self, n: object, mode: object) -> None:
        # The ctor pins its user boundary to exactly (2, pl.SplitMode).
        if isinstance(n, bool) or not isinstance(n, int):
            raise ValueError(f"pl.split_aiv(n): n must be the integer 2, got {n!r}")
        if n != 2:
            raise ValueError(f"pl.split_aiv(n): n must be the integer 2 (the two AIV lanes), got {n}")
        if not isinstance(mode, _ir.SplitMode):
            raise ValueError(f"pl.split_aiv(mode=...): mode must be a pl.SplitMode, got {mode!r}")
        self.n = n
        self.mode = mode
        # The active recipe instance — the one lowering the for-form
        # (``_split_aiv_scope``). Created at the trace hook, single-use.
        self._recipe: Any = None

    @property
    def split_aiv_mode(self) -> _ir.SplitMode:
        """The split mode read by PtoForConstruct at target activation.

        The property lives on this object, not in ambient state — and stays
        class-level because the ``_SplitAivDescriptor`` protocol isinstance
        probes with ``getattr_static``, which bypasses ``__getattr__``. It is
        what routes the transfer rule (``break``/``continue`` shadowing) and
        the ``split_aiv`` interface tag."""
        return self.mode

    def _recipe_instance(self) -> Any:
        """One fresh single-use recipe instance carrying this call's arguments."""
        return _split_aiv_scope(self.mode)

    # —— the traced for-protocol (forwarded to the recipe) ——

    def __trace_iter__(self, ctx: TraceContext) -> Scalar:
        """Enter the for-form: the recipe instance drives from the target binding."""
        self._recipe = self._recipe_instance()
        return self._recipe.__trace_iter__(ctx)

    def begin(
        self,
        builder: IRBuilder,
        name: str,
        already_in_incore: bool = False,
        sticky_dump_vars: list[_ir.Var] | None = None,
    ) -> _ir.Var:
        """The ``_ForDescriptor`` hook: forward to the recipe's for-arm."""
        return self._recipe.begin(builder, name, already_in_incore, sticky_dump_vars)

    def __trace_exit__(self, ctx: TraceContext) -> None:
        """The close hook: the recipe's post-yield code runs here."""
        return self._recipe.__trace_exit__(ctx)

    def __iter__(self) -> Iterator[Scalar]:
        # The executed path routes the for through rt.begin_for →
        # try_open_for (the __trace_iter__ protocol), which returns the
        # one-element iteration surface the host for iterates — so this
        # __iter__ is only reached by direct scripts outside decoration.
        # The yielded type is the traced loop variable — a ``Scalar[INDEX]``
        # bound to ``pl.tile.get_subblock_idx()`` (see ``begin``).
        # Raise loudly rather than silently yielding zero iterations
        # (mirrors SpmdContext).
        raise RuntimeError(
            "pl.split_aiv(...) loop form is only valid inside a decorated "
            "@pl.program / @pl.function body (the tracing frontend routes it through "
            "the __trace_iter__ trace-protocol hook). If you're seeing this, the "
            "surrounding function was not decorated."
        )


def _at_scope_attrs(
    ctx: ConstructContext,
    deps: list[Scalar | Tensor | Array | None] | None,
    no_dep_args: list[Tensor] | None,
    dumps: list[Tensor] | None,
) -> list[tuple[str, Any]]:
    """Thread ``pl.at`` keyword values onto the resulting scope.

    The outliner translates ``deps=`` to ``manual_dep_edges``
    (Form-1 element reads desugar through the kit's ``lower_deps``),
    ``no_dep_args=`` to ``arg_direction_overrides_vars``, and ``dumps=``
    to ``dump_vars`` so the tracing path matches the parser-owned
    representation."""
    attrs: list[tuple[str, Any]] = []
    if deps:
        # ``deps=`` takes producer TaskIds (Scalar[TASK_ID] facades from
        # ``pl.submit``); ``None`` entries are the conditional-edge
        # sentinel — skipped rather than coerced, so a partial deps list
        # doesn't break collection.
        edges = ctx.lower_deps(deps, "pl.at")
        if edges:
            attrs.append(("manual_dep_edges", edges))
    for name, tensors in (
        ("arg_direction_overrides_vars", no_dep_args),
        ("dump_vars", dumps),
    ):
        if not tensors:
            continue
        for t in tensors:
            if not hasattr(t, "unwrap"):
                raise TypeError(
                    f"pl.at(no_dep_args/dumps=[...]) entries must be staged (outer-scope) values, got {t!r}"
                )
        attrs.append((name, [t.unwrap() for t in tensors]))
    return _merge_sticky_dump_vars(attrs, ctx.sticky_dump_vars()) or []


@construct
def at(  # noqa: PLR0913 — the public pl.at(...) kwarg surface, unchanged
    ctx: ConstructContext,
    level: _ir.Level,
    role: _ir.Role | None = None,
    *,
    optimizations: Sequence[Optimization] | None = None,
    # ``deps=`` threads any symbolic outer-scope value (a TaskId Scalar,
    # a TaskId Array, or a producing Tensor) onto ``manual_dep_edges`` —
    # the validation below only requires the ``unwrap`` facade protocol.
    deps: list[Scalar | Tensor | Array | None] | None = None,
    no_dep_args: list[Tensor] | None = None,
    dumps: list[Tensor] | None = None,
    allow_early_resolve: bool = False,
    predicate: Scalar | Tensor | None = None,
    name_hint: str = "",
    windowize: bool = False,
):
    """Mark a region of code for execution at a specific hierarchy level.

    A construct recipe: the pre-yield code is the hand-written ``__trace_enter__``
    in the original order (attr collection, TaskId capture, the parser-parity
    validations), the ``yield`` is the body boundary, and the close opens the
    deferred scope if the body emitted nothing and ends it.

       With ``level=pl.Level.CORE_GROUP``, the ``optimizations=`` list controls
       the resulting scope kind:

       - no entries → ``ScopeStmt(InCore)``
       - ``pl.split(mode)`` → ``ScopeStmt(InCore, split=mode)``

       For all other levels, this creates a Hierarchy scope.

       Args:
           level: Target hierarchy level (e.g. pl.Level.HOST, pl.Level.CORE_GROUP).
           role: Function role (Orchestrator or Worker). Default: None.
           optimizations: Optional list literal of optimization entries. Each
               entry must be ``pl.split(mode)`` — written inline at the call
               site, since the DSL parser inspects the AST and does not accept
               dynamically built variables here.
           deps: Optional explicit producer-edge list (TaskId Vars and/or
               ``None`` sentinels). Lowered to the resulting Call's
               ``manual_dep_edges`` attr, which codegen packs into a
               ``set_dependencies(...)`` invocation. Operates at the *TaskId*
               level — orthogonal to ``no_dep_args=``, which operates at the
               *arg-slot* level on captured tensors.
           no_dep_args: Optional list literal of outer-scope tensor names
               captured by the scope body. Each entry must be a bare tensor
               name; the parser resolves it to a Var, the outliner translates
               the Var list into positional indices into the synthesised
               Call's args, and ``DeriveCallDirections`` overwrites those
               slots to ``ArgDirection.NoDep``. Equivalent to wrapping the
               same tensor with ``pl.no_dep(t)`` at an explicit kernel call
               site — use this form when the kernel call is synthesised by
               the ``pl.at`` outliner and there is no syntactic call-arg slot
               to wrap. Legal for both read-only captures and captures that
               the scope body mutates via ``pl.assemble`` / ``pl.store`` (the
               outliner classifies the latter as ``InOut`` on the synthesised
               kernel, and ``NoDep`` overrides ``InOut`` just as it overrides
               ``Input``); the user is asserting that sibling fan-outs touch
               disjoint regions of the tensor and therefore do not need
               OverlapMap dep tracking. Note: ``deps=`` takes TaskIds, while
               ``no_dep_args=`` takes tensors — they describe different things.
           dumps: Optional list literal of outer-scope tensor names to mark for
               selective tensor dump on the synthesised kernel dispatch. The
               scope-level selective-dump surface, symmetric with ``deps=``: the
               outliner translates the captured-tensor entries into the dispatch's
               ``dump_vars`` by Var identity. Equivalent to declaring the same
               tensors with ``pl.dump_tag(t)`` before the scope; use ``dumps=``
               when you want the dump targets listed explicitly at the scope.
           allow_early_resolve: Opt the outlined dispatch in as a speculative
               early-dispatch producer (simpler#1065). Same hint as
               ``pl.submit(..., allow_early_resolve=True)``: the scheduler may
               pre-stage this task's consumers before it completes. Forces the
               scope to lower to an ``ir.Submit`` (even without ``as tid``) so the
               flag can ride to codegen, where it emits
               ``Arg::set_allow_early_resolve(true)``. Pure scheduling hint.
           predicate: Optional **dispatch predicate** — a single comparison of
               one tensor element against an integer literal, e.g.
    ``predicate=(rc[0, 0] > 0)``. Same surface and lowering as
               ``pl.spmd_submit(..., predicate=...)`` / the ``pl.spmd`` scope
               form: the scheduler evaluates it at the dispatch point (after this
               scope's dependencies are satisfied, so the value is current) and
               retires the task inline — never dispatching it to a core — when it
               is false, while still settling fanin/fanout so downstream
               consumers unlock. Like ``allow_early_resolve`` it forces the scope
               to lower to an ``ir.Submit`` (even without ``as tid``); the
               outliner moves it onto ``Submit::predicate_``. Placement-gated like
               the parser's ``_parse_at_predicate``: ``CORE_GROUP`` level only,
               not nested inside ``pl.cluster()`` / ``pl.spmd()`` / another
               CORE_GROUP ``pl.at`` (nesting inside a *non*-CORE_GROUP ``pl.at``
               is fine — a Hierarchy scope is not a task wrapper), and the
               enclosing function must be one ``OutlineIncoreScopes`` rewrites.
               The comparison shape itself is enforced by orchestration codegen
               (the tracing-wide rule shared with the call forms), and the
               operand's producer must be one of ``deps=`` — nothing checks that
               contract under tracing, so it stays the author's responsibility.
           name_hint: Optional name hint for the outlined function (must be
               a valid identifier).
           windowize: Explicitly allow local windowization for the outlined InCore
               kernel. The default is False.

       Returns:
           Context manager for the appropriate scope.

       Examples:
           >>> # InCore scope:
           >>> with pl.at(level=pl.Level.CORE_GROUP):
           ...     y = pl.ops.add(x, x)

           >>> # InCore scope with split hint:
           >>> with pl.at(level=pl.Level.CORE_GROUP,
           ...            optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
           ...     y = pl.ops.add(x, x)

           >>> # Hierarchy scope (unchanged behavior):
           >>> with pl.at(level=pl.Level.HOST, role=pl.Role.SubWorker):
           ...     y = pl.add(x, x)

           >>> # Dispatch predicate — skip the outlined task when the gate is 0
           >>> with pl.at(level=pl.Level.CORE_GROUP, predicate=(rc[0, 0] > 0)):
           ...     y = pl.ops.add(x, x)
    """
    # —— the scope attrs, in the parser's canonical order ——
    # (manual_dep_edges, arg_direction_overrides_vars, dump_vars via the
    # collect below; task_id_var; allow_early_resolve; predicate; slot_num;
    # windowize — print → reparse must reproduce the list byte-for-byte).
    scope_attrs = _at_scope_attrs(ctx, deps, no_dep_args, dumps)

    as_binding = None
    if ctx.binding_name:
        # The TaskId-capture channel: the kit mints the ``Scalar[TASK_ID]``
        # Var, emits the transient placeholder one stmt before the scope, and
        # hands back the binding proxy plus the ``task_id_var`` attr entry.
        as_binding, tid_attr = ctx.capture_task_id(ctx.binding_name)
        scope_attrs.append(tid_attr)

    if allow_early_resolve:
        scope_attrs.append(("allow_early_resolve", True))
    is_core_group = level == _ir.Level.CORE_GROUP
    split_mode: _ir.SplitMode | None = None
    # Parser-location validations the traced frontend re-encodes at the
    # trace-protocol boundary (the AST parser's ``pl.at`` arg-surface discipline):
    # optimizations must be a list literal, at most one split entry, split
    # only with CORE_GROUP, role only for hierarchy levels, and a HOST-level
    # scope never takes a SubWorker role (inline ``pl.at`` regions are not
    # kernels — ``@pl.function`` declares workers).
    if not isinstance(optimizations, (list, tuple, type(None))):
        raise TypeError(
            "pl.at(optimizations=...) must be a list literal, got "
            f"{optimizations!r} ({type(optimizations).__name__})"
        )
    if level == _ir.Level.HOST and role == _ir.Role.SubWorker:
        raise TypeError(
            "pl.at(level=pl.Level.HOST, role=pl.Role.SubWorker) is not supported "
            "— an inline pl.at region is not a kernel; declare a worker with "
            "@pl.function instead"
        )
    split_mode, split_slot_num = _fold_optimization_entries(optimizations, "pl.at", split_gate=is_core_group)
    if level == _ir.Level.CORE_GROUP and role is not None:
        raise TypeError("role= is not supported with level=pl.Level.CORE_GROUP")
    if predicate is not None:
        # The dispatch-predicate placement gates, mirroring the parser's
        # ``_parse_at_predicate`` (rejected here rather than miscompiled):
        # a predicate only has a runtime carrier where the scope becomes an
        # independently submitted L0 task — a CORE_GROUP scope that
        # ``OutlineIncoreScopes`` turns into a ``Submit``.
        if not is_core_group:
            raise TypeError(
                "pl.at(predicate=...) is only supported with level=pl.Level.CORE_GROUP "
                "— a non-CORE_GROUP pl.at builds a Hierarchy scope, which is never "
                "outlined into a task dispatch, so the dispatch predicate would be lost. "
                "Use pl.at(level=pl.Level.CORE_GROUP, predicate=...) or "
                "pl.spmd_submit(..., predicate=...)."
            )
        for kind, owner in (
            (_ir.ScopeKind.Cluster, "pl.cluster()"),
            (_ir.ScopeKind.Spmd, "pl.spmd()"),
            (_ir.ScopeKind.InCore, "another pl.at(level=pl.Level.CORE_GROUP)"),
        ):
            if ctx.is_inside_scope(kind):
                raise TypeError(
                    f"`pl.at(..., predicate=(...))` cannot be nested inside `{owner}` — "
                    "the inner dispatch is folded into the enclosing task, so the "
                    "dispatch predicate would be lost. Move the predicate to the "
                    "enclosing dispatch, or use a top-level "
                    "`with pl.at(level=pl.Level.CORE_GROUP, predicate=...):`."
                )
        # ``OutlineIncoreScopes`` only rewrites Opaque / orchestration-like
        # bodies (an ``Inline`` helper is accepted — its call site is only
        # revealed by ``InlineFunctions``, and the outlining passes
        # re-check the placement they can actually see). A scope written in
        # a Group / Spmd / InCore-family kernel body is never outlined into
        # a dispatch at all.
        func_type = ctx.function_type()
        if func_type not in (
            _ir.FunctionType.Opaque,
            _ir.FunctionType.Inline,
        ) and not _ir.is_orchestration_like(func_type):
            raise TypeError(
                "`pl.at(..., predicate=(...))` is not supported inside a "
                f"{func_type.name} function — the scope is never outlined into its own "
                "task dispatch there, so the dispatch predicate would be lost. Write "
                "the predicated `with pl.at(level=pl.Level.CORE_GROUP, predicate=...):` "
                "in the orchestration function that launches this kernel."
            )
        if not hasattr(predicate, "unwrap"):
            raise TypeError(f"pl.at(predicate=...) must be a staged value, got {predicate!r}")
        # Appended after ``allow_early_resolve`` and before ``slot_num`` /
        # ``windowize`` — the parser's canonical attr order, so a
        # print -> reparse cycle reproduces the attr list byte-for-byte.
        scope_attrs.append(("predicate", predicate.unwrap()))

    if split_slot_num is not None:
        # slot_num lands as the "slot_num" scope attr on the InCore
        # scope (mirrors the AST parser's
        # _append_split_slot_num_attr — appended last so
        # print→reparse reproduces the attr order;
        # ExpandMixedKernel sizes the cross-core ring from it).
        scope_attrs.append(("slot_num", split_slot_num))
    if windowize:
        # ``windowize=`` allows local windowization for the outlined InCore
        # kernel — appended after slot_num, mirroring the parser's attr
        # order so a structural comparison is order-stable.
        scope_attrs.append(("windowize", True))

    # —— open the deferred scope; the body lands inside it ——
    if not is_core_group:
        region = ctx.open_scope(
            _ir.ScopeKind.Hierarchy,
            attrs=scope_attrs,
            level=level,
            role=role,
            name_hint=name_hint,
        )
    else:
        region = ctx.open_scope(
            _ir.ScopeKind.InCore,
            attrs=scope_attrs,
            split=split_mode,
            name_hint=name_hint,
        )
    yield as_binding
    ctx.exit(region)


__all__ = [
    "ScopeMode",
    "manual_scope",
    "reg_scope",
    "scope",
    "at",
    "cluster",
    "spmd",
    "split_aiv",
]
