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
    from pypto.pypto_core import ir

from pypto.language.diagnostics import ParserSyntaxError
from pypto.language.typing import Scalar, Tensor
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir

from .loops import _LoopTargetBinding
from .optimizations import CrossCoreSlot, Optimization, Split

# Validated at import so a bare name comparison can never drift from the
# registry (see .claude/rules/operator-identity-checks.md).
_ARRAY_GET_ELEMENT = _ir.get_op("array.get_element").name


def _merge_cache_policy_attr(
    attrs: list[tuple[str, Any]] | None, decls: list[tuple[_ir.Var, int]]
) -> list[tuple[str, Any]]:
    """Insert the ``cache_policy_vars`` attr at the parser's position.

    Right before the first ``task_id_var`` / ``slot_num`` entry, else appended
    — ``structural_equal`` compares attrs positionally, so first trace and
    print -> reparse must agree on the order (mirrors the AST parser's
    ``_merge_cache_policy_attr``)."""
    new_attrs: list[tuple[str, Any]] = list(attrs) if attrs else []
    insert_at = next(
        (i for i, (k, _) in enumerate(new_attrs) if k in {"task_id_var", "slot_num"}),
        len(new_attrs),
    )
    new_attrs.insert(insert_at, ("cache_policy_vars", decls))
    return new_attrs


def _is_task_id_dep_var(expr: _ir.Expr) -> bool:
    """Is ``expr`` a whole-value dependency Var (a TaskId scalar or a
    TASK_ID array)? The direct ``manual_dep_edges`` lane — the AST parser's
    ``_is_dep_var_type``."""
    if not isinstance(expr, _ir.Var):
        return False
    t = expr.type
    return (
        isinstance(t, _ir.ScalarType)
        and t.dtype == DataType.TASK_ID
        or (isinstance(t, _ir.ArrayType) and t.dtype == DataType.TASK_ID)
    )


def _is_per_element_task_id_read(expr: _ir.Expr) -> bool:
    """Is ``expr`` an ``array.get_element(arr, idx)`` reading a TASK_ID?

    The Form-1 deps shape — the AST parser's ``_is_per_element_task_id_read``.
    """
    if not isinstance(expr, _ir.Call) or expr.op.name != _ARRAY_GET_ELEMENT:
        return False
    if not isinstance(expr.type, _ir.ScalarType) or expr.type.dtype != DataType.TASK_ID:
        return False
    if not expr.args:
        return False
    arr_type = expr.args[0].type
    return isinstance(arr_type, _ir.ArrayType) and arr_type.dtype == DataType.TASK_ID


def _lower_scope_deps(deps: Sequence[object], construct: str, builder: IRBuilder) -> list[_ir.Var]:
    """Lower a ``deps=[...]`` list to ``manual_dep_edges`` Vars (Form-1 desugar).

    The AST parser's ``_parse_submit_deps_kwarg`` discipline: a bare TaskId /
    whole-array Var rides the direct lane verbatim; an element read of a
    TASK_ID array (``tids[g]`` — the DSL's documented shape for handing a
    TaskId between orchestration phases) cannot ride ``manual_dep_edges``
    directly (the attr holds Vars), so the whole list is desugared into a
    fresh ``_submit_deps_buf`` TASK_ID array — ``array.create`` followed by
    one ``array.update_element`` per entry — and that array Var becomes the
    single edge, so the existing whole-array dep codegen fires unchanged.

    ``None`` entries are the conditional-edge sentinel and are dropped. Mixing
    a whole-array entry with element reads is rejected (the synthesizer would
    feed an ArrayType Var into ``update_element``'s scalar value slot — the
    parser refuses the same mixed form).
    """
    from pypto.language.facade import is_facade  # noqa: PLC0415 - the facade/typing cycle

    entries: list[_ir.Expr] = []
    needs_desugar = False
    saw_whole_array = False
    for d in deps:
        if d is None:
            continue
        if not is_facade(d):
            raise TypeError(f"{construct}(deps=[...]) entries must be staged (outer-scope) values, got {d!r}")
        expr = d.unwrap()
        if _is_task_id_dep_var(expr):
            if isinstance(expr.type, _ir.ArrayType):
                saw_whole_array = True
                if needs_desugar:
                    raise TypeError(
                        f"{construct}(deps=[...]) cannot mix a whole TASK_ID array with "
                        f"element reads — desugaring would feed the array into the "
                        f"synthesized buffer's scalar slot; depend on the array alone "
                        f"or on its elements"
                    )
            entries.append(expr)
        elif _is_per_element_task_id_read(expr):
            needs_desugar = True
            if saw_whole_array:
                raise TypeError(
                    f"{construct}(deps=[...]) cannot mix a whole TASK_ID array with "
                    f"element reads — desugaring would feed the array into the "
                    f"synthesized buffer's scalar slot; depend on the array alone "
                    f"or on its elements"
                )
            entries.append(expr)
        else:
            raise ParserSyntaxError(
                f"{construct}(deps=[...]) entries must be a TaskId variable, a "
                f"TASK_ID array element (e.g. arr[i]), or None — got {expr!r}",
                hint=(
                    "Bind a TaskId with `_, tid = pl.submit(self.producer, ...)` "
                    "or `with pl.at(...) as tid:`, read a slot with `arr[i]` from a "
                    "`pl.array.create(N, pl.TASK_ID)`, or pass None for no producer."
                ),
            )
    if not entries:
        return []
    if not needs_desugar:
        # The direct lane: every entry is already a Var (auto-mapping absorbs
        # per-trace Var identity in comparisons against the parser's IR).
        return [e for e in entries if isinstance(e, _ir.Var)]
    # Form 1: synthesize the deps buffer in the enclosing region — before the
    # scope opens, exactly where the parser emits it.
    span = _ir.Span.unknown()
    size = _ir.ConstInt(len(entries), DataType.INDEX, span)
    create = _ir.create_op_call("array.create", [size], {"dtype": DataType.TASK_ID}, span)
    buf = builder.let("_submit_deps_buf", create)
    for i, entry in enumerate(entries):
        idx = _ir.ConstInt(i, DataType.INDEX, span)
        update = _ir.create_op_call("array.update_element", [buf, idx, entry], {}, span)
        buf = builder.let("_submit_deps_buf", update)
    return [buf]


def _buffer_cache_policy(decls: list[tuple[_ir.Var, int]], var: _ir.Var, policy: int) -> None:
    """Buffer one declaration under the parser's redundancy rule.

    A repeat naming the same binding with the same policy is redundant — keep
    the first so the attr stays a set of distinct tensors. A repeat naming a
    *different* policy is a contradiction, not a redundancy: silently keeping
    the first would resolve it toward BYPASS, the direction that also asserts
    the coherency contract the second statement retracts."""
    existing = next((p for v, p in decls if v is var), None)
    if existing is None:
        decls.append((var, policy))
    elif existing != policy:
        raise ParserSyntaxError(
            f"pl.set_cache_policy() declares conflicting policies for '{var.name_hint}' "
            f"in one scope: {_ir.CachePolicy(existing).name} then {_ir.CachePolicy(policy).name}",
            hint=(
                "One tensor takes one policy per scope. Drop the redundant "
                "declaration, or move the differing one into its own scope. To vary "
                "the policy per access, use pl.load(..., cache=...) instead."
            ),
        )


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
            raise TypeError(
                f"{owner}(dumps=[...]) entries must be staged (outer-scope) values, got {t!r}"
            )
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


def _reject_nested_manual_scope(ctx: TraceContext) -> None:
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


class scope:
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

    def __init__(self, mode: ScopeMode = ScopeMode.AUTO):
        self.mode = mode
        self._begun: bool = False  # whether open fired (never-begun no-op for auto-unwind)

    def __trace_enter__(
        self,
        ctx: TraceContext,
        as_name: str | None = None,
        single_call_body: bool = False,
        reads_block_idx: bool = False,
    ) -> None:
        """Enter the PTO runtime scope through the trace-enter hook.

        `Frame` recognizes this trace context manager by protocol membership
        and passes the tracing context. The hook emits ``ScopeKind.Runtime``
        with the selected mode; builder access is write-only through
        ``begin_scope``/``end_scope``. The corresponding close hook ends it."""
        if as_name is not None:
            raise TypeError("with pl.scope(...) as ...: is not supported — a runtime scope captures no value")
        if self.mode == ScopeMode.AUTO and ctx.auto_scope_enabled():
            raise ParserSyntaxError(
                "a hand-placed AUTO 'with pl.scope()' requires @pl.function(auto_scope=False)"
            )
        if self.mode == ScopeMode.MANUAL:
            _reject_nested_manual_scope(ctx)
        elif self.mode == ScopeMode.AUTO and ctx.is_inside_manual_scope():
            raise ParserSyntaxError(
                "an AUTO scope may not be nested inside a manual scope",
                hint="The runtime forbids AUTO scope nested in MANUAL scope; move it outside.",
            )
        ctx.builder.begin_scope(_ir.ScopeKind.Runtime, manual=(self.mode == ScopeMode.MANUAL))
        self._begun = True

    def __trace_exit__(self, ctx: TraceContext) -> None:
        """Exit the PTO runtime scope through the tracing context.

        This is the exact PtoWithConstruct close; no-op when never opened."""
        if self._begun:
            ctx.builder.end_scope()
            self._begun = False

    def __enter__(self) -> None:
        # The static contract mirrors the traced ``with`` (the body executes;
        # the carrier protocol routes through ``__trace_enter__``) — a
        # NoReturn here would poison control-flow analysis of kernel bodies.
        raise RuntimeError(
            "pl.scope(...) with-form is only valid inside a decorated "
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
        # Returning None (falsy) is "do not suppress" — identical to the
        # former ``return False`` at runtime, but the ``None`` annotation
        # keeps pyright from considering an exception-suppressed continuation
        # path (which made ``return``-carrying ``with`` bodies look like
        # functions that miss a return).
        pass


class graph:
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

    #: Function types whose body is a device-side kernel, where a Graph
    #: region (a topology of task *launches*) cannot live — mirrors the AST
    #: parser's ``_GRAPH_FORBIDDEN_FUNC_TYPES``.
    _FORBIDDEN_FUNC_TYPES = frozenset(
        {
            _ir.FunctionType.InCore,
            _ir.FunctionType.AIC,
            _ir.FunctionType.AIV,
            _ir.FunctionType.Group,
            _ir.FunctionType.Spmd,
        }
    )

    def __init__(self, name: str, *, dumps: list[Tensor] | None = None) -> None:
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
        self.name = name
        # Explicit selective-dump targets — the scope-level ``dumps=`` surface
        # and the print -> reparse spelling for a ``dump_vars`` attr.
        self.dumps = dumps
        self._begun: bool = False  # whether open fired (never-begun no-op for auto-unwind)

    def __trace_enter__(
        self,
        ctx: TraceContext,
        as_name: str | None = None,
        single_call_body: bool = False,
        reads_block_idx: bool = False,
    ) -> Any:
        """Enter the Graph scope through the trace-enter hook.

        Emits ``GraphScopeStmt`` through write-only ``begin_scope``/``end_scope``
        operations. ``as_name`` is rejected because a Graph scope captures no
        producer TaskId. Placement rejections mirror the AST parser's: a Graph
        region records a topology of task launches, so it must sit around the
        dispatches, never inside a device scope, inside another Graph, or in a
        device-side kernel body. The forward-sticky ``pl.dump_tag`` set merges
        onto the scope at entry, mirroring the cluster hook and the parser's
        ``_parse_scope_body`` merge."""
        if as_name is not None:
            raise TypeError(
                "with pl.graph(...) as ...: is not supported — a Graph scope captures no producer TaskId"
            )
        func_type = ctx.function_type()
        if func_type in self._FORBIDDEN_FUNC_TYPES:
            raise ParserSyntaxError(
                f"pl.graph() is not valid inside a {func_type.name} function",
                hint="A Graph region records orchestration task launches, so it belongs in the "
                "@pl.jit entry (or another @pl.jit.graph function), not in a kernel body. To make a "
                "whole function a Graph, decorate it with @pl.jit.graph instead.",
            )
        for device_kind, construct in (
            (_ir.ScopeKind.InCore, "pl.at(level=...)"),
            (_ir.ScopeKind.Hierarchy, "pl.at(level=..., role=...)"),
            (_ir.ScopeKind.Cluster, "pl.cluster()"),
            (_ir.ScopeKind.Spmd, "pl.spmd(...)"),
        ):
            if ctx.is_inside_scope(device_kind):
                raise ParserSyntaxError(
                    f"pl.graph() cannot be nested inside `{construct}`",
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
        ctx.builder.begin_scope(
            _ir.ScopeKind.Graph,
            name_hint=self.name,
            attrs=_merge_sticky_dump_vars(
                _explicit_dump_attrs(self.dumps, "pl.graph"), ctx.sticky_dump_vars()
            ),
        )
        self._begun = True

    def __trace_exit__(self, ctx: TraceContext) -> None:
        """Exit the Graph scope through the tracing context.

        No-op when never opened."""
        if self._begun:
            ctx.builder.end_scope()
            self._begun = False

    def __enter__(self) -> None:
        raise RuntimeError(
            "pl.graph(...) with-form is only valid inside a decorated "
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
        """Exit the Graph scope context (direct-script path)."""
        pass


class manual_scope:
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

    def __init__(self) -> None:
        self._begun: bool = False  # whether open fired (never-begun no-op for auto-unwind)

    def __trace_enter__(
        self,
        ctx: TraceContext,
        as_name: str | None = None,
        single_call_body: bool = False,
        reads_block_idx: bool = False,
    ) -> None:
        """Enter the MANUAL PTO runtime scope through the trace-enter hook.

        This mirrors ``scope.__trace_enter__`` with ``manual=True``; the
        printed ``with pl.scope(mode=MANUAL):`` form uses the same protocol
        object. The MANUAL-in-MANUAL rejection is the same runtime rule the
        parser enforces through ``_manual_scope_depth``."""
        if as_name is not None:
            raise TypeError(
                "with pl.manual_scope() as ...: is not supported — a runtime scope captures no value"
            )
        _reject_nested_manual_scope(ctx)
        ctx.builder.begin_scope(_ir.ScopeKind.Runtime, manual=True)
        self._begun = True

    def __trace_exit__(self, ctx: TraceContext) -> None:
        """Exit the MANUAL PTO runtime scope through the tracing context.

        No-op when never opened."""
        if self._begun:
            ctx.builder.end_scope()
            self._begun = False

    def __enter__(self) -> None:
        raise RuntimeError(
            "pl.manual_scope() with-form is only valid inside a decorated "
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
        # Returning None (falsy) is "do not suppress" — identical to the
        # former ``return False`` at runtime, but the ``None`` annotation
        # keeps pyright from considering an exception-suppressed continuation
        # path (which made ``return``-carrying ``with`` bodies look like
        # functions that miss a return).
        pass


class cluster:
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

    def __init__(self, *, name_hint: str = "", dumps: list[Tensor] | None = None) -> None:
        self.name_hint = name_hint
        # Explicit selective-dump targets — the scope-level ``dumps=`` surface
        # and the print -> reparse spelling for a ``dump_vars`` attr (F39).
        self.dumps = dumps
        self._begun: bool = False  # whether open fired (never-begun no-op for auto-unwind)

    def __trace_enter__(
        self,
        ctx: TraceContext,
        as_name: str | None = None,
        single_call_body: bool = False,
        reads_block_idx: bool = False,
    ) -> Any:
        """Enter the Cluster scope through the trace-enter hook.

        `Frame` recognizes this trace context manager by protocol membership
        and passes the tracing context. The hook emits ``ClusterScopeStmt``
        through write-only ``begin_scope``/``end_scope`` operations. ``as_name``
        is rejected because a Cluster scope has no producer TaskId to capture."""
        if as_name is not None:
            raise TypeError(
                "with pl.cluster(...) as ...: is not supported — a Cluster scope captures no producer TaskId"
            )
        ctx.builder.begin_scope(
            _ir.ScopeKind.Cluster,
            name_hint=self.name_hint,
            attrs=_merge_sticky_dump_vars(
                _explicit_dump_attrs(self.dumps, "pl.cluster"), ctx.sticky_dump_vars()
            ),
        )
        self._begun = True

    def __trace_exit__(self, ctx: TraceContext) -> None:
        """Exit the Cluster scope through the tracing context.

        No-op when never opened."""
        if self._begun:
            ctx.builder.end_scope()
            self._begun = False

    def __enter__(self) -> None:
        raise RuntimeError(
            "pl.cluster(...) with-form is only valid inside a decorated "
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
        """Exit the Cluster scope context (direct-script path)."""
        pass


def _split_spmd_for_loop_name_hints(name_hint: str) -> tuple[str, str]:
    """Map one ``for i in pl.spmd(..., name_hint=...)`` hint to Spmd vs InCore
    names — the mirror of the AST parser's helper of the same name (dsl_api
    cannot import from ast_parser — the parser imports the stdlib). The
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
        deps: list[Scalar | None] | None = None,
        allow_early_resolve: bool = False,
        predicate: Scalar | None = None,
    ) -> None:
        core = _pin_spmd_core_num(core_num)
        self.core_num = core
        self.sync_start = sync_start
        self.name_hint = name_hint
        self.optimizations = optimizations
        self.deps = deps
        self.allow_early_resolve = allow_early_resolve
        self.predicate = predicate
        self._begun: bool = False  # whether open fired (never-begun no-op for auto-unwind)
        self._loop_form: bool = False  # True after begin() (the for-form) — closes two scopes
        self._incore_wrapped: bool = False  # the with-form's inner InCore wrapper
        # The plain with-form defers its InCore-carrier decision to the body: the
        # carrier opens at the first per-block-index read (the parser classifies the
        # same fact statically — `_spmd_body_reads_block_idx`). The wrapped
        # with-form and the for-form defer too — for the cache-policy
        # declarations (see ``at._scope_pending``) — so every form shares one
        # open path (``open_deferred_scope``); only the plain form opts out of
        # the first-emission flush (``_carrier_flush_on_emit``).
        self._carrier_pending: bool = False
        self._carrier_name_hint: str = ""
        self._carrier_attrs: list[tuple[str, Any]] | None = None
        self._carrier_split: _ir.SplitMode | None = None
        self._carrier_flush_on_emit: bool = True
        # The for-form's deferred prologue: ``i = pl.tile.get_block_idx()`` is
        # the InCore carrier's first statement. Built at ``begin`` (span
        # parity with an immediate emission), emitted at open.
        self._carrier_prologue: tuple[_ir.Var, _ir.Expr] | None = None
        self._cache_decls: list[tuple[_ir.Var, int]] = []

    def _submit_only_scope_attrs(self) -> list[tuple[str, object]]:
        """The Submit-only kwargs, in canonical order — shared by both forms.

        ``allow_early_resolve`` / ``predicate`` ride on the ``SpmdScopeStmt``
        until the Spmd outliner moves them onto the synthesised ``Submit``.
        Both the with-form (``__trace_enter__``) and the for-form (``begin``)
        build their scope attrs through this one helper, so the two
        spellings cannot drift — the for-form used to silently drop both
        (the DSV4 parity verifier's ``expert_shared`` divergence)."""
        attrs: list[tuple[str, object]] = []
        if self.allow_early_resolve:
            attrs.append(("allow_early_resolve", True))
        if self.predicate is not None:
            if not hasattr(self.predicate, "unwrap"):
                raise TypeError(f"pl.spmd(predicate=...) must be a staged value, got {self.predicate!r}")
            attrs.append(("predicate", self.predicate.unwrap()))
        return attrs

    def _reject_deps_without_capture(self) -> None:
        """Reject ``deps=`` on a form that captures no TaskId (both call sites).

        Mirrors the parser's ``allow_deps=False`` path: gated by keyword
        presence, not the resolved list — ``deps=[]`` / ``deps=[None]`` are
        rejected too (a truthiness check would silently accept them)."""
        raise TypeError(
            "pl.spmd() does not accept 'deps=' here — use `with pl.spmd(n, "
            "deps=[...]) as tid:` (the with-form) to declare explicit TaskId "
            "deps, or `out, tid = pl.spmd_submit(..., deps=[...])` for the "
            "single-call form."
        )

    def _reject_submit_only_kwargs_in_cluster(self, builder: IRBuilder) -> None:
        """Reject Submit-only kwargs on a cluster-nested ``pl.spmd`` (both forms).

        Mirrors the parser's ``_reject_spmd_submit_only_kwargs_in_cluster``:
        a cluster-nested pl.spmd is unwrapped into the Group function by
        ``OutlineClusterScopes`` and never lowers to a ``Submit``, so the
        kwargs would be silently lost. Without the trace-time guard the attrs
        reach the pass pipeline and fail there as an internal error
        (``UnwrapNestedSpmd`` re-asserts) — late and far less diagnosable."""
        if not builder.is_inside_scope(_ir.ScopeKind.Cluster):
            return
        if self.allow_early_resolve:
            kwarg, lost = "allow_early_resolve=True", "the early-dispatch hint"
        elif self.predicate is not None:
            kwarg, lost = "predicate=(...)", "the dispatch predicate"
        else:
            return
        raise TypeError(
            f"`pl.spmd(..., {kwarg})` cannot be nested inside `pl.cluster()` — "
            "a cluster-nested pl.spmd is unwrapped into the Group function and never "
            f"produces a Submit, so {lost} would be lost. Use a standalone "
            f"`with pl.spmd(..., {kwarg}):` (implicit cluster) to keep it."
        )

    def __trace_enter__(
        self,
        ctx: TraceContext,
        as_name: str | None = None,
        single_call_body: bool = False,
        reads_block_idx: bool = False,
    ) -> Any:
        """Enter the SPMD scope through the trace-enter hook.

        The hook emits ``SpmdScopeStmt`` through write-only builder operations.
        ``as_name`` requests the TaskId-capture binding proxy (the
        ``task_id_var`` attr); capture is orthogonal to the body shape.
        ``single_call_body`` and ``reads_block_idx`` — the rewriter's static
        mirrors of the parser's ``_emit_spmd_body`` rule — pick the body shape,
        in the parser's order: a body that reads the per-block index anywhere
        is *inline* and wraps the whole body in the nested InCore scope (split
        name hints) for ``OutlineIncoreScopes`` to outline; else a single
        call-valued statement without a split hint is the direct-dispatch shape
        (``SpmdScopeStmt(<the call stmt>)``, the user's unsplit hint); else the
        deferred carrier waits on the read trigger. Scope attributes are
        emitted in their canonical order: ``manual_dep_edges``,
        ``task_id_var``, ``allow_early_resolve``, and ``predicate``."""
        split_mode, split_slot_num = _parse_spmd_optimizations(self.optimizations)
        builder = ctx.builder
        # Form-surface discipline, mirroring the parser: ``deps=`` is legal only
        # on the ``as tid`` capture form (gated by keyword presence, not the
        # resolved list — ``deps=[]`` / ``deps=[None]`` are rejected too), the
        # Submit-only kwargs are rejected inside ``pl.cluster()``, and so is
        # ``as tid`` itself (nothing cluster-nested lowers to a Submit).
        if self.deps is not None and as_name is None:
            self._reject_deps_without_capture()
        self._reject_submit_only_kwargs_in_cluster(builder)
        if as_name and builder.is_inside_scope(_ir.ScopeKind.Cluster):
            raise TypeError(
                "`with pl.spmd(...) as tid:` cannot capture a TaskId when nested "
                "inside `pl.cluster()` — a cluster-nested pl.spmd is unwrapped into "
                "the Group function and never produces a Submit. Use a standalone "
                "`with pl.spmd(...) as tid:` (implicit cluster) to capture the "
                "dispatch TaskId."
            )
        scope_attrs: list[tuple[str, Any]] = []
        if self.deps:
            edges = _lower_scope_deps(self.deps, "pl.spmd", builder)
            if edges:
                scope_attrs.append(("manual_dep_edges", edges))
        as_binding = None
        if as_name:
            # the TaskId-capture channel (mirrors AtContext.__trace_enter__ /
            # the AST's _parse_spmd_scope_with_tid): create the Var, emit the
            # placeholder, stamp task_id_var, hand back the deferred binding proxy.
            tid_var = builder.var(as_name, _ir.ScalarType(DataType.TASK_ID))
            builder.assign(
                tid_var,
                _ir.create_op_call("system.task_invalid", [], {}, _ir.Span.unknown()),
            )
            scope_attrs.append(("task_id_var", tid_var))
            as_binding = _TaskIdBinding(tid_var)
        scope_attrs.extend(self._submit_only_scope_attrs())
        spmd_name_hint: str
        incore_name_hint: str
        # The wrap decision mirrors the parser's ``_emit_spmd_body``, which is
        # static and body-wide: a body that reads the per-block index anywhere
        # (``reads_block_idx`` — the defect the review pinned as F23; the
        # parser's trigger is ``_spmd_body_reads_block_idx``'s whole-suite walk,
        # not the first top-level read), a split hint, a cross-core slot count,
        # or a *captured* body that is not one call-valued statement wraps the
        # body in an InCore scope that opens at the body's first statement. The
        # remaining uncaptured form keeps the deferred carrier: it opens if the
        # runtime read probe fires (a spelling the AST walk cannot see, e.g. an
        # aliased ``get_block_idx``), while a dispatch body whose callee reads
        # the index internally, or a bare nested ``pl.at`` providing its own
        # region, never opens it — the statements land directly in the Spmd
        # scope. TaskId capture on a single-call body only adds the
        # ``task_id_var`` attr — orthogonal to the shape.
        self._incore_wrapped = (
            reads_block_idx
            or split_mode is not None
            or split_slot_num is not None
            or (as_name is not None and not single_call_body)
        )
        # The split name hints are pessimistic for the uncaptured form: the
        # carrier may still open on the block-index read, and the Spmd scope's
        # name is fixed at begin — a bare uncaptured body keeps the user's
        # unsplit hint only when it is one call-valued statement (the shape
        # that can never open the carrier). The read fact wins over that shape:
        # the parser classifies such a body as inline and splits the hints.
        if self._incore_wrapped or not single_call_body:
            spmd_name_hint, incore_name_hint = _split_spmd_for_loop_name_hints(self.name_hint)
        else:
            spmd_name_hint = self.name_hint
            incore_name_hint = self.name_hint
        incore_attrs: list[tuple[str, Any]] | None = None
        if split_slot_num is not None:
            incore_attrs = [("slot_num", split_slot_num)]
        # Forward-sticky dump tags attach to the scope that owns the outlined
        # dispatch: the InCore wrapper when wrapped, else the Spmd scope.
        if self._incore_wrapped:
            incore_attrs = _merge_sticky_dump_vars(incore_attrs, ctx.sticky_dump_vars())
        else:
            scope_attrs = _merge_sticky_dump_vars(scope_attrs, ctx.sticky_dump_vars()) or []
        builder.begin_scope(
            _ir.ScopeKind.Spmd,
            core_num=self.core_num,
            sync_start=self.sync_start,
            name_hint=spmd_name_hint,
            attrs=scope_attrs or None,
        )
        # Every form's carrier begin is deferred
        # (``open_deferred_scope``) so the body's leading cache-policy
        # declarations can join its attrs; the unwrapped plain form
        # additionally opts out of the first-emission flush — its open decision
        # then belongs to the per-block-index read trigger.
        self._carrier_pending = True
        self._carrier_name_hint = incore_name_hint
        self._carrier_attrs = incore_attrs
        self._carrier_split = split_mode
        self._carrier_flush_on_emit = self._incore_wrapped
        self._begun = True
        self._loop_form = False
        return as_binding

    def attach_cache_policy(self, var: _ir.Var, policy: int) -> None:
        """Buffer one ``pl.set_cache_policy`` declaration for the pending carrier.

        Called by the Frame when the marker reaches the carrier's body top
        level (see ``_DeferredDeclarableScope``); folded into the carrier's
        ``cache_policy_vars`` attr at open."""
        _buffer_cache_policy(self._cache_decls, var, policy)

    def open_deferred_scope(self, builder: IRBuilder) -> None:
        """Open the deferred InCore carrier — its first statement arrived (or
        it closed).

        Shared by every form: the wrapped with-form and the for-form flush
        here at the body's first IR-emitting statement, and the plain
        with-form's block-index trigger routes here through
        ``open_pending_carrier``. The buffered cache-policy declarations join
        the attrs at the parser's position, and the for-form's deferred
        prologue (``i = pl.tile.get_block_idx()``) is the carrier's first
        statement. No-op once the carrier opened or the form never deferred
        one."""
        if not self._carrier_pending:
            return
        self._carrier_pending = False
        attrs = self._carrier_attrs
        if self._cache_decls:
            attrs = _merge_cache_policy_attr(attrs, self._cache_decls)
        builder.begin_scope(
            _ir.ScopeKind.InCore,
            split=self._carrier_split,
            name_hint=self._carrier_name_hint,
            attrs=attrs,
        )
        if self._carrier_prologue is not None:
            loop_var, block_idx = self._carrier_prologue
            builder.assign(loop_var, block_idx, _ir.Span.unknown())
        # The for-form closes its carrier through the ``_loop_form`` arm; the
        # with-forms through ``_incore_wrapped`` — setting both would end the
        # carrier twice.
        self._incore_wrapped = not self._loop_form

    def open_pending_carrier(self, builder: IRBuilder) -> None:
        """Open the deferred InCore carrier — the body read the per-block index.

        Called by the Frame when a ``tile.get_block_idx`` value is bound at the
        top level of this with-form's body: the body does its own per-block
        work and must be outlined into a per-block kernel, so the statements
        land inside an ``InCoreScopeStmt`` — the parser's inline-body shape
        (its ``_emit_spmd_body`` wraps on the same fact, read statically). A
        no-op once the carrier opened or the form never deferred one."""
        self.open_deferred_scope(builder)

    def cancel_pending_carrier(self) -> None:
        """Abandon the deferred InCore carrier — the body supplied its own.

        Called by the Frame when the body opens a CORE_GROUP region of its own
        (a bare nested ``pl.at(level=pl.Level.CORE_GROUP, ...)``, which *is* the
        parser's carrier for that shape): the region nests directly in the Spmd
        scope, so this construct's carrier must not open — neither at the body's
        first statement nor at its own close (the close hook begins a
        still-pending flushable carrier, which is what left a spurious empty
        InCore behind). No-op once the carrier opened or the form never
        deferred one. Buffered cache-policy declarations keep the old path: a
        carrier with declarations still begins at the close so they have a
        home (and the plain never-wrapped form keeps its loud rejection).
        """
        if not self._carrier_pending or self._cache_decls:
            return
        self._carrier_pending = False
        self._incore_wrapped = False

    def __trace_iter__(self, ctx: TraceContext) -> Scalar:
        """Return a loop-target binding proxy through the trace-iter hook.

        The proxy defers ``for i in pl.spmd(...)`` construction until the
        first target binding, when ``begin`` emits the SPMD scope and its
        auto-outlined InCore region."""
        del ctx  # the context is consumed by the binding proxy's __trace_bind__, not here
        return _LoopTargetBinding(self)

    def begin(
        self,
        builder: IRBuilder,
        name: str,
        already_in_incore: bool = False,
        sticky_dump_vars: list[_ir.Var] | None = None,
    ) -> _ir.Var:
        """Construct the SPMD loop at the first target binding.

        Emit ``SpmdScopeStmt`` and defer the auto-outlined ``InCoreScopeStmt``
        carrier: the carrier's begin waits for the body's first IR-emitting
        statement (the Frame's pre-emit flush) or this construct's close, so
        the body's leading ``pl.set_cache_policy(...)`` declarations can join
        the carrier's attrs — the declarations attach to the carrier, the
        scope the loop body lowers to. The loop variable binds to
        ``pl.tile.get_block_idx()`` as the carrier's first statement (built
        here for span parity, emitted at open). Split optimization attributes
        apply to the nested InCore scope, and forward-sticky dump tags merge
        onto that InCore scope. Return the loop-variable ``ir.Var``
        that the Frame registers."""
        del already_in_incore  # the for-form always wraps in its own InCore carrier
        split_mode, split_slot_num = _parse_spmd_optimizations(self.optimizations)
        from pypto.language.op import tile_ops as _tile_ops  # noqa: PLC0415 - avoid the op-module cycle

        # The for-form's mirror of the parser's ``allow_deps=False``: the loop
        # form captures no TaskId, so ``deps=`` has nowhere to hang its edges.
        # The cluster guard mirrors the with-form's.
        if self.deps is not None:
            self._reject_deps_without_capture()
        self._reject_submit_only_kwargs_in_cluster(builder)
        spmd_name_hint, incore_name_hint = _split_spmd_for_loop_name_hints(self.name_hint)
        incore_attrs: list[tuple[str, Any]] | None = None
        if split_slot_num is not None:
            incore_attrs = [("slot_num", split_slot_num)]
        incore_attrs = _merge_sticky_dump_vars(incore_attrs, sticky_dump_vars or [])
        loop_var = builder.var(name, _ir.ScalarType(DataType.INDEX), _ir.Span.unknown())
        block_idx = _tile_ops.get_block_idx().unwrap()
        builder.begin_scope(
            _ir.ScopeKind.Spmd,
            core_num=self.core_num,
            sync_start=self.sync_start,
            name_hint=spmd_name_hint,
            attrs=self._submit_only_scope_attrs() or None,
        )
        self._carrier_split = split_mode
        self._carrier_attrs = incore_attrs
        self._carrier_name_hint = incore_name_hint
        self._carrier_prologue = (loop_var, block_idx)
        self._carrier_pending = True
        self._begun = True
        self._loop_form = True
        return loop_var

    def __trace_exit__(self, ctx: TraceContext) -> object:
        """Finalize the SPMD scope(s) through the tracing context.

        A carrier still pending at exit had an emission-free body — a
        declaration-only or empty body — so it begins here (an empty
        InCoreScopeStmt) before the close, unless it is the plain with-form's
        never-triggered carrier carrying buffered declarations: a dispatch
        body has no InCore to attach them to, which is the loud rejection
        (the parser rejects the same body at its scope-kind check). The loop
        form opens nested Spmd and InCore scopes, so it closes InCore first
        and then Spmd. No-op when never opened."""
        if self._carrier_pending and self._cache_decls and not self._loop_form and not self._incore_wrapped:
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
        # Only the flushable forms begin at exit (an emission-free body —
        # declaration-only or empty): the plain with-form's carrier that never
        # triggered stays closed — a dispatch body is the bare shape
        # OutlineIncoreScopes leaves behind.
        if self._carrier_pending and self._carrier_flush_on_emit:
            self.open_deferred_scope(ctx.builder)
        if self._begun:
            if self._loop_form:
                ctx.builder.end_scope()  # the InCore wrapper closes before the Spmd scope
            if self._incore_wrapped:
                ctx.builder.end_scope()  # the with-form's InCore wrapper, same LIFO order
            result = ctx.builder.end_scope()
            self._begun = False
            return result
        return None

    def __enter__(self) -> Scalar:
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
        self._begun: bool = False  # whether begin fired (never-begun no-op)
        self._wrapped: bool = False  # bare-top-level form synthesized an InCore wrapper

    @property
    def split_aiv_mode(self) -> _ir.SplitMode:
        """The split mode read by PtoForConstruct at target activation.

        The property lives on this trace iterable, not in ambient state."""
        return self.mode

    def __trace_iter__(self, ctx: TraceContext) -> Scalar:
        """Return a loop-target binding proxy through the trace-iter hook.

        The proxy defers ``SplitAivScopeStmt`` construction until the first
        target binding, when ``begin`` receives the tracing context."""
        del ctx  # the context is consumed by the binding proxy's __trace_bind__, not here
        return _LoopTargetBinding(self)

    def begin(
        self,
        builder: IRBuilder,
        name: str,
        already_in_incore: bool = False,
        sticky_dump_vars: list[_ir.Var] | None = None,
    ) -> _ir.Var:
        """Construct the SplitAiv region at the first target binding.

        Open ``ScopeKind.SplitAiv`` with the selected split mode, bind the
        loop variable to ``pl.tile.get_subblock_idx()``, and expose the mode
        to operations in the region. A top-level form synthesizes an InCore
        wrapper; a form already inside InCore emits in place. Forward-sticky
        dump tags merge onto the synthesized wrapper. The lowering pass
        consumes this region before code generation.

        ``already_in_incore`` is a Frame-provided fact; the trace iterable
        does not inspect builder scope state."""
        from pypto.language.op import tile_ops as _tile_ops  # noqa: PLC0415 - avoid the op-module cycle

        # A function-level AUTO split (``optimizations=[pl.split(MODE)]``) and
        # an explicit ``pl.split_aiv`` region are mutually exclusive AIV-split
        # mechanisms (RFC #1820) — downstream lowering takes the per-region
        # path and would silently drop the function-level split. **Any**
        # ``pl.split(...)`` is rejected, ``SplitMode.NONE`` included: the
        # literal NONE spelling is only observable here (the builder's
        # user-split mirror), never in the IR. Mirrors the AST parser's
        # ``_reject_user_split_with_split_aiv_region``; OutlineIncoreScopes
        # remains the backstop for the modes that reach the IR.
        user_split = builder.enclosing_user_split()
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
        loop_var = builder.var(name, _ir.ScalarType(DataType.INDEX), _ir.Span.unknown())
        if already_in_incore:
            # FLATTEN: nested inside a CORE_GROUP InCore scope — emit the
            # region in place; it nests inside the open context.
            self._wrapped = False
        else:
            # Bare top-level form: synthesize an InCore wrapper first and
            # nest the region inside it (keeps it eligible for
            # OutlineIncoreScopes — else the region has no enclosing InCore
            # to outline). Forward-sticky dump tags merge onto the wrapper.
            builder.begin_scope(
                _ir.ScopeKind.InCore,
                _ir.Span.unknown(),
                attrs=_merge_sticky_dump_vars(None, sticky_dump_vars or []),
            )
            self._wrapped = True
        builder.begin_scope(_ir.ScopeKind.SplitAiv, _ir.Span.unknown(), split=self.mode)
        builder.assign(loop_var, _tile_ops.get_subblock_idx().unwrap(), _ir.Span.unknown())
        self._begun = True
        return loop_var

    def __trace_exit__(self, ctx: TraceContext) -> None:
        """Finalize the SplitAiv region through the tracing context.

        No-op when never begun. Close the SplitAiv region and, for a
        top-level form, its synthesized InCore wrapper."""
        if self._begun:
            ctx.builder.end_scope(_ir.Span.unknown())  # the SplitAiv region
            if self._wrapped:
                ctx.builder.end_scope(_ir.Span.unknown())  # the synthesized InCore wrapper
            self._begun = False

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


class _TaskIdBinding(Scalar):
    """A binding proxy for ``with pl.at(...) as tid``.

    The proxy carries ``__trace_bind__`` and returns the pre-created
    ``Scalar[TASK_ID]`` variable for the target name. The Sema registers that
    variable, so the target reads the same value the outliner rebinds to the
    Submit producer TaskId. The proxy is idempotent because the variable is
    intentionally reused on every call.
    """

    def __init__(self, tid_var: _ir.Var) -> None:
        super().__init__(expr=_ir.ConstInt(0, DataType.TASK_ID, _ir.Span.unknown()))
        self._tid_var = tid_var

    # The bind hook emits no IR, so the Frame's pre-emit flush must not run
    # here: a wrapped ``pl.spmd`` carrier is pending at the ``as tid`` target
    # bind, and flushing it would beat the body's own CORE_GROUP region to the
    # absorption check (F44). The var is pre-created at the scope's enter, so
    # nothing structural needs the scopes opened first.
    _emits_on_bind: bool = False

    def __trace_bind__(self, ctx: TraceContext, name: str) -> _ir.Var:
        del ctx, name  # the Var is pre-created; no construction needed
        return self._tid_var


class at:
    """Mark a region of code for execution at a specific hierarchy level.

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
           name_hint: Optional name hint for the outlined function (must be a
               valid identifier).
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

    def __init__(
        self,
        level: ir.Level,
        role: ir.Role | None = None,
        *,
        optimizations: Sequence[Optimization] | None = None,
        # ``deps=`` threads any symbolic outer-scope value (a TaskId Scalar
        # or a producing Tensor) onto ``manual_dep_edges`` — the validation
        # below only requires the ``unwrap`` facade protocol.
        deps: list[Scalar | Tensor | None] | None = None,
        no_dep_args: list[Tensor] | None = None,
        dumps: list[Tensor] | None = None,
        allow_early_resolve: bool = False,
        predicate: Scalar | None = None,
        name_hint: str = "",
        windowize: bool = False,
    ) -> None:
        self.level = level
        self.role = role
        self.optimizations = optimizations
        self.deps = deps
        self.no_dep_args = no_dep_args
        self.dumps = dumps
        self.allow_early_resolve = allow_early_resolve
        self.predicate = predicate
        self.name_hint = name_hint
        self.windowize = windowize
        self._begun: bool = False  # whether open fired (never-begun no-op for auto-unwind)
        # The deferred begin: a declarable scope does not begin at enter — its
        # attrs are fixed at ``begin_scope``, so the body's leading
        # ``pl.set_cache_policy(...)`` declarations must be able to join them.
        # The Frame opens it at the body's first IR-emitting statement (the
        # builder's pre-emit hook) or at this construct's own close.
        self._scope_pending: bool = False
        self._pending_begin: tuple[_ir.ScopeKind, dict[str, Any], list[tuple[str, Any]]] | None = None
        self._cache_decls: list[tuple[_ir.Var, int]] = []

    def _collect_scope_attrs(self, builder: IRBuilder) -> list[tuple[str, Any]]:
        """Thread ``pl.at`` keyword values onto the resulting scope.

        The outliner translates ``deps=`` to ``manual_dep_edges``
        (Form-1 element reads desugar through :func:`_lower_scope_deps`),
        ``no_dep_args=`` to ``arg_direction_overrides_vars``, and ``dumps=``
        to ``dump_vars`` so the tracing path matches the parser-owned
        representation."""
        attrs: list[tuple[str, Any]] = []
        if self.deps:
            # ``deps=`` takes producer TaskIds (Scalar[TASK_ID] facades from
            # ``pl.submit``); ``None`` entries are the conditional-edge
            # sentinel — skipped rather than coerced, so a partial deps list
            # doesn't break collection.
            edges = _lower_scope_deps(self.deps, "pl.at", builder)
            if edges:
                attrs.append(("manual_dep_edges", edges))
        for name, tensors in (
            ("arg_direction_overrides_vars", self.no_dep_args),
            ("dump_vars", self.dumps),
        ):
            if not tensors:
                continue
            for t in tensors:
                if not hasattr(t, "unwrap"):
                    raise TypeError(
                        f"pl.at(no_dep_args/dumps=[...]) entries must be "
                        f"staged (outer-scope) values, got {t!r}"
                    )
            attrs.append((name, [t.unwrap() for t in tensors]))
        return attrs

    @staticmethod
    def _parse_optimizations(
        opts: Sequence[Optimization] | None, is_core_group: bool
    ) -> tuple[_ir.SplitMode | None, int | None]:
        """Fold ``pl.at(optimizations=[...])`` entries to the split mode and
        the cross-core slot count — the shared entry fold with the CORE_GROUP
        gate on split entries (the mirror of the AST parser's
        ``_parse_optimizations_list`` with ``owner="pl.at"``)."""
        return _fold_optimization_entries(opts, "pl.at", split_gate=is_core_group)

    def __trace_enter__(
        self,
        ctx: TraceContext,
        as_name: str | None = None,
        single_call_body: bool = False,
        reads_block_idx: bool = False,
    ) -> Any:
        """Enter the ``pl.at`` region through the trace-enter hook.

        The hook validates ``deps``, ``role``, and split options, unwraps its
        own fields, merges forward-sticky dump tags, and emits the scope
        through write-only builder operations. ``as_name`` requests a
        deferred TaskId binding proxy."""
        builder = ctx.builder
        scope_attrs = (
            _merge_sticky_dump_vars(self._collect_scope_attrs(builder), ctx.sticky_dump_vars()) or []
        )
        as_binding = None
        if as_name:
            # The TaskId-capture channel: create the
            # ``Scalar[TASK_ID]`` Var, emit the transient
            # ``AssignStmt(tid, system.task_invalid())`` placeholder one stmt
            # BEFORE the scope (so ConvertToSSA has a def; the outliner drops
            # it and synthesises the real ``AssignStmt(tid, TupleGetItem(
            # ret_tmp, n_outputs))`` binding — mirrors ast_parser
            # ``_parse_at_meta``), stamp ``task_id_var`` on the ScopeStmt, and
            # hand back a binding proxy whose ``__trace_bind__`` registers the Var as
            # the as-name's staged local.
            tid_var = builder.var(as_name, _ir.ScalarType(DataType.TASK_ID))
            builder.assign(
                tid_var,
                _ir.create_op_call("system.task_invalid", [], {}, _ir.Span.unknown()),
            )
            scope_attrs.append(("task_id_var", tid_var))
            as_binding = _TaskIdBinding(tid_var)
        if self.allow_early_resolve:
            scope_attrs.append(("allow_early_resolve", True))
        level = self.level
        is_core_group = level == _ir.Level.CORE_GROUP
        split_mode: _ir.SplitMode | None = None
        # Parser-location validations the traced frontend re-encodes at the
        # trace-protocol boundary (the AST parser's ``pl.at`` arg-surface discipline):
        # optimizations must be a list literal, at most one split entry, split
        # only with CORE_GROUP, role only for hierarchy levels, and a HOST-level
        # scope never takes a SubWorker role (inline ``pl.at`` regions are not
        # kernels — ``@pl.function`` declares workers).
        opts = self.optimizations
        if not isinstance(opts, (list, tuple, type(None))):
            raise TypeError(
                f"pl.at(optimizations=...) must be a list literal, got {opts!r} ({type(opts).__name__})"
            )
        if self.level == _ir.Level.HOST and self.role == _ir.Role.SubWorker:
            raise TypeError(
                "pl.at(level=pl.Level.HOST, role=pl.Role.SubWorker) is not supported "
                "— an inline pl.at region is not a kernel; declare a worker with "
                "@pl.function instead"
            )
        split_mode, split_slot_num = self._parse_optimizations(opts, is_core_group)
        if self.level == _ir.Level.CORE_GROUP and self.role is not None:
            raise TypeError("role= is not supported with level=pl.Level.CORE_GROUP")
        if self.predicate is not None:
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
                if builder.is_inside_scope(kind):
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
            if not hasattr(self.predicate, "unwrap"):
                raise TypeError(f"pl.at(predicate=...) must be a staged value, got {self.predicate!r}")
            # Appended after ``allow_early_resolve`` and before ``slot_num`` /
            # ``windowize`` — the parser's canonical attr order, so a
            # print -> reparse cycle reproduces the attr list byte-for-byte.
            scope_attrs.append(("predicate", self.predicate.unwrap()))

        if split_slot_num is not None:
            # slot_num lands as the "slot_num" scope attr on the InCore
            # scope (mirrors the AST parser's
            # _append_split_slot_num_attr — appended last so
            # print→reparse reproduces the attr order;
            # ExpandMixedKernel sizes the cross-core ring from it).
            scope_attrs.append(("slot_num", split_slot_num))
        if self.windowize:
            # ``windowize=`` allows local windowization for the outlined InCore
            # kernel — appended after slot_num, mirroring the parser's attr
            # order so a structural comparison is order-stable.
            scope_attrs.append(("windowize", True))
        if not is_core_group:
            self._pending_begin = (
                _ir.ScopeKind.Hierarchy,
                {"level": level, "role": self.role, "name_hint": self.name_hint},
                scope_attrs,
            )
        else:
            self._pending_begin = (
                _ir.ScopeKind.InCore,
                {"split": split_mode, "name_hint": self.name_hint},
                scope_attrs,
            )
        self._scope_pending = True
        return as_binding

    def attach_cache_policy(self, var: _ir.Var, policy: int) -> None:
        """Buffer one ``pl.set_cache_policy`` declaration for the pending scope.

        Called by the Frame when the marker reaches this scope's body top
        level (see ``_DeferredDeclarableScope``); folded into the
        ``cache_policy_vars`` attr at ``open_deferred_scope``."""
        _buffer_cache_policy(self._cache_decls, var, policy)

    def open_deferred_scope(self, builder: IRBuilder) -> None:
        """Begin the deferred scope — its first statement arrived (or it closed).

        No-op once begun. The buffered cache-policy declarations join the
        attrs at the parser's position; the stored begin parameters are
        exactly what an immediate ``begin_scope`` would have received."""
        pending = self._pending_begin
        if pending is None:
            return
        self._scope_pending = False
        self._pending_begin = None
        kind, kwargs, attrs = pending
        if self._cache_decls:
            attrs = _merge_cache_policy_attr(attrs, self._cache_decls)
        builder.begin_scope(kind, attrs=attrs, **kwargs)
        self._begun = True

    def __trace_exit__(self, ctx: TraceContext) -> object:
        """Exit the ``pl.at`` scope through the tracing context.

        A scope still pending at exit had an emission-free body — begin it
        now (an empty ScopeStmt), then close. No-op when never opened."""
        if self._scope_pending:
            self.open_deferred_scope(ctx.builder)
        if self._begun:
            result = ctx.builder.end_scope()
            self._begun = False
            return result
        return None

    def __enter__(self) -> Scalar:
        # The static contract is the traced ``as tid`` binding (the
        # pre-created ``Scalar[TASK_ID]`` variable), mirroring the traced
        # semantics — not the raising host path.
        raise RuntimeError(
            "pl.at(...) with-form is only valid inside a decorated "
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


__all__ = [
    "ScopeMode",
    "manual_scope",
    "scope",
    "at",
    "cluster",
    "spmd",
    "split_aiv",
]
