# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Construct recipes — the authoring tier between kernel authors and the tracer.

A recipe is an ordinary generator function declaring how one construct lowers:
code before the ``yield`` opens the construct's regions, the ``yield`` hands
the body its value, and code after it finalizes. The tracer keeps everything
else — execution, stack discipline, IR construction:

.. code-block:: python

    @pl.authoring.construct
    def blocked_spmd(ctx, shape, tile, *, deps=None):
        blocks = tuple((s + t - 1) // t for s, t in zip(shape, tile))
        with ctx.region(pl.spmd(prod(blocks), deps=deps)) as tid:
            yield BlockGrid(shape, tile, ctx.resolve_binding(tid))

    with blocked_spmd((tokens, heads), (1, HEAD_TILE)) as block:
        token, head = block.axes
        block.region(query).load()

One generator serves both statement spellings. The ``with`` spelling enters at
``__trace_enter__`` and the ``yield`` value is the ``as``-target — a host-bound
handle (a :class:`ConstructHandle`) or a capture binding. The ``for`` spelling
(``for i in blocked_spmd(...)``) enters at the loop-target binding: the
generator starts there (the loop variable's source name is known only at that
point), the ``yield`` value is the loop variable — an ``ir.Var`` minted through
:meth:`ConstructContext.loop_var` — and the body that follows is the
representative iteration. The driver implements the whole trace protocol
(``__trace_enter__`` / ``__trace_iter__`` / ``begin`` / ``__trace_exit__``) for
the recipe, so neither spelling's conventions can be gotten wrong.

The layering rule: this module is the sanctioned client of the tracer's
internals (the Frame's carry guard and deferred-scope flush), and recipes are
clients of :class:`ConstructContext` only. The trace protocol
(``__trace_enter__`` / ``__trace_bind__`` / ``__trace_exit__``) remains the ABI
between this tier and the tracer, but recipe authors never implement it — the
driver below implements it *for* the recipe, so the private conventions of a
hand-written construct (deferred-carrier flags, binding decline, close
ordering) cannot be gotten wrong.

Composition is protocol-level, not interpreter-level: a recipe opening
``pl.spmd`` through :meth:`ConstructContext.region` drives that construct's
enter/exit hooks directly. Exactly one interpreter construct is installed —
the recipe's own — so stack-sensitive routing (break/continue selection,
staged-control-flow facts) sees the same shape as a hand-written ``with``.
The recipe instance delegates the deferred-scope structural surface
(``_carrier_pending``, ``open_pending_carrier``, ...) of its *primary* region —
the first region it entered — to the Frame's existing member checks, which is
what keeps the carrier and cache-policy machinery working unchanged.

The vocabulary stays closed where it must: recipes configure and compose
existing IR constructs; they cannot mint statements, scopes, or control flow.
A recipe that needs a new lowering shape is a framework change first.
"""

from __future__ import annotations

import functools
import inspect
import types
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, overload

from pypto.ir.builder import IRBuilder
from pypto.language.diagnostics import ParserSyntaxError
from pypto.language.protocol import (
    SupportsTraceBind,
    SupportsTraceEnter,
    SupportsTraceExit,
    TraceContext,
)
from pypto.language.tracing.semantics import PtoFrame, _enters_incore_scope
from pypto.pypto_core import DataType, ir

if TYPE_CHECKING:
    from pypto.language.authoring.dispatch import SubmitOperand
    from pypto.language.authoring.session import AuthoringSession

__all__ = ["CarrierRegion", "ConstructContext", "ConstructHandle", "LoopRegion", "ScopeRegion", "construct"]

#: The loop-kind vocabulary — the ``pl.range`` family's kinds mapped to the
#: ``ForStmt``'s ``ForKind`` (the same mapping ``RangeIterator.begin`` applies).
_LOOP_FOR_KINDS: dict[str, ir.ForKind] = {
    "range": ir.ForKind.Sequential,
    "parallel": ir.ForKind.Parallel,
    "unroll": ir.ForKind.Unroll,
    "pipeline": ir.ForKind.Pipeline,
}


def _pin_loop_bound(bound: object, which: str) -> Any:
    """A loop bound at the operand boundary: a ``Scalar`` facade unwraps to
    its Expr; a host ``int`` / raw ``ir.Expr`` passes through
    (``begin_for_loop`` materializes the int as a ``ConstInt(INDEX)``)."""
    from pypto.language.typing import Scalar  # noqa: PLC0415 - the facade/typing cycle

    if isinstance(bound, Scalar):
        return bound.unwrap()
    if isinstance(bound, bool) or not isinstance(bound, (int, ir.Expr)):
        raise TypeError(f"ctx.open_loop {which} bound must be a host int or a staged value, got {bound!r}")
    return bound


class _Unset:
    """The ``as_name`` sentinel distinguishing "not given" from an explicit ``None``."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


# -----------------------------------------------------------------------------------------------------------
# The scope-construct lowering helpers — shared by the kit's primitives and the
# stdlib scope constructs (pl.at / pl.spmd / ...). They live here because the
# kit's ScopeRegion and capture channel implement them; the constructs module
# imports them back. The one ``deps=`` lowering lives in
# :mod:`pypto.language.authoring.dispatch` (the trace lane's single
# implementation — the dispatch effect and ``ctx.lower_deps`` share it).
# -----------------------------------------------------------------------------------------------------------


def _merge_cache_policy_attr(
    attrs: list[tuple[str, Any]] | None, decls: list[tuple[ir.Var, int]]
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


def _buffer_cache_policy(decls: list[tuple[ir.Var, int]], var: ir.Var, policy: int) -> None:
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
            f"in one scope: {ir.CachePolicy(existing).name} then {ir.CachePolicy(policy).name}",
            hint=(
                "One tensor takes one policy per scope. Drop the redundant "
                "declaration, or move the differing one into its own scope. To vary "
                "the policy per access, use pl.load(..., cache=...) instead."
            ),
        )


class _TaskIdBinding:
    """A binding proxy for a ``with ... as tid`` capture.

    The proxy carries ``__trace_bind__`` and returns the pre-created
    ``Scalar[TASK_ID]`` variable for the target name. The Sema registers that
    variable, so the target reads the same value the outliner rebinds to the
    Submit producer TaskId. The proxy is idempotent because the variable is
    intentionally reused on every call.

    Not a facade subclass: the kit's capture channel hands it to the recipe
    unbound, and ``resolve_binding``/the Frame's bind dispatch read the Var
    through the hook.
    """

    __slots__ = ("_tid_var",)

    def __init__(self, tid_var: ir.Var) -> None:
        self._tid_var = tid_var

    def __trace_bind__(self, ctx: TraceContext, name: str) -> ir.Var:
        del ctx, name  # the Var is pre-created; no construction needed
        return self._tid_var


class ScopeRegion:
    """A scope a recipe opened directly — the framework's lowering primitive.

    :meth:`ConstructContext.open_scope` returns one. The begin is deferred by
    design (the declarable-scope contract): a ``ScopeStmt``'s attrs are fixed
    at ``begin_scope``, so the body's leading ``pl.set_cache_policy(...)``
    declarations must be able to join them — the Frame opens the region at
    the body's first IR-emitting statement (the pre-emit flush) or at its
    close, whichever comes first. The region implements the deferred-scope
    structural surface itself, so a recipe whose primary region is a
    ``ScopeRegion`` answers the Frame's member checks exactly the way a
    hand-written scope construct does.

    ``prologue`` statements are the region's own first emissions (a for-form
    binding its loop variable to the block or lane index) — emitted right
    after an immediate ``begin_scope``, or at a deferred open before any
    body statement lands.
    """

    def __init__(
        self,
        kind: ir.ScopeKind,
        begin_kwargs: dict[str, Any],
        attrs: list[tuple[str, Any]],
        builder: IRBuilder | None = None,
        prologue: list[tuple[ir.Var, ir.Expr]] | None = None,
    ) -> None:
        self._cache_decls: list[tuple[ir.Var, int]] = []
        self._begun: bool = False
        self._flush_on_emit: bool = True
        # Statements the region owns as its first emissions (built at
        # construction for span parity, emitted at open — immediate or
        # deferred alike).
        self._prologue: list[tuple[ir.Var, ir.Expr]] = list(prologue or ())
        if builder is None:
            # Deferred: the Frame owns the open (the pre-emit flush / the close),
            # and the region answers the declarable-scope protocol.
            self._pending_begin: tuple[ir.ScopeKind, dict[str, Any], list[tuple[str, Any]]] | None = (
                kind,
                begin_kwargs,
                attrs,
            )
            self._scope_pending: bool = True
        else:
            # Immediate: the scope opens at entry, exactly as a construct whose
            # attrs are fixed at enter does (no cache-policy surface).
            self._pending_begin = None
            self._scope_pending = False
            builder.begin_scope(kind, attrs=attrs, **begin_kwargs)
            self._emit_prologue(builder)
            self._begun = True

    def _emit_prologue(self, builder: IRBuilder) -> None:
        for var, expr in self._prologue:
            builder.assign(var, expr, ir.Span.unknown())

    # —— the deferred-scope structural surface ——

    def attach_cache_policy(self, var: ir.Var, policy: int) -> None:
        """Buffer one ``pl.set_cache_policy`` declaration for the pending scope.

        Called by the Frame when the marker reaches the scope's body top
        level; folded into the ``cache_policy_vars`` attr at open."""
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
        self._emit_prologue(builder)
        self._begun = True

    # —— the kit's close ——

    def __trace_exit__(self, ctx: TraceContext) -> object:
        """Close the scope: begin it if the body emitted nothing, then end it.

        Returns the closed ``ScopeStmt`` (the Frame attaches deferred
        construct comments to it). No-op when never begun.

        A region whose open decision belongs to a *body fact* rather than to
        the first emission (a carrier) does not begin here — see
        :class:`CarrierRegion`."""
        if self._scope_pending and self._flush_on_emit:
            self.open_deferred_scope(ctx.builder)
        if self._begun:
            result = ctx.builder.end_scope()
            self._begun = False
            return result
        return None


class ConstructHandle:
    """The base class for a recipe's rich ``as``-target (the body's handle).

    A handle is a host object by design — it carries *authoring* structure
    (axes, regions, metadata) the staged vocabulary cannot spell. Subclasses
    inherit the binding discipline: the handle declines staged binding, so the
    Frame host-binds it under the ``as`` name and the body reads it natively.
    A handle is also recognized as *not staged* (no facade, no trace
    protocol), which keeps it out of every staged-value path.
    """

    def __trace_bind__(self, ctx: TraceContext, name: str) -> None:
        """Decline staged binding: the Frame host-binds the handle under ``name``."""
        del ctx, name


class LoopRegion:
    """A recipe-authored ``ForStmt`` — the loop-tier lowering primitive.

    :meth:`ConstructContext.open_loop` returns one. The loop opens
    immediately (its parameters are fixed at entry — there is no
    cache-policy surface to defer for) and closes with ``end_for_loop``;
    the region composes like a scope region, so the recipe brackets its
    ``yield`` with the region's ``with`` form and the close runs at the
    recipe's exit. As a context manager it yields the loop variable.

    The loop-carried channels are the SSA pass's inference — the same
    inference that carries a ``pl.range`` accumulator: bind the initial
    value through :meth:`ConstructContext.bind` before the open, rebind the
    name in the body (a handle's ``bind``, or the body's own rebinding),
    and ``ConvertToSSA`` creates the ``iter_args`` / ``return_vars``.

    ``kind`` is the ``pl.range``-family loop-kind vocabulary (``range`` /
    ``parallel`` / ``unroll`` / ``pipeline``) — the fact the interpreter's
    transfer rules read (a ``parallel`` / ``unroll`` loop rejects
    ``break`` / ``continue``; a ``range`` loop accepts them).
    """

    def __init__(self, loop_var: ir.Var, kind: str, session: AuthoringSession) -> None:
        self._loop_var = loop_var
        self.kind = kind
        self._session = session
        self._closed = False

    @property
    def var(self) -> ir.Var:
        """The loop variable — the region's own value (a ``Scalar[INDEX]`` Var)."""
        return self._loop_var

    def __trace_exit__(self, ctx: TraceContext) -> object:
        """Close the loop (``end_for_loop``); returns the closed ``ForStmt``."""
        if not self._closed:
            self._closed = True
            return ctx.builder.end_for_loop()
        return None

    # —— the recipe-facing composition surface ——

    def __enter__(self) -> ir.Var:
        """The ``with ctx.open_loop(...) as i:`` form — the loop variable."""
        return self._loop_var

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        """Close the region on the success path (the LIFO-validated exit).

        During an exception unwind nothing runs — the trace is discarded
        with its one-shot Sema (the same doctrine as every staged region)."""
        del exc_val, exc_tb
        if exc_type is None:
            self._session.exit_region(self)
            self.__trace_exit__(self._session.caller_frame.trace_context())
        return False  # never suppress — a failing recipe fails the trace


class ConstructContext:
    """The capability object handed to a construct recipe as its first argument.

    The surface is closed and curated: region composition (enter/exit), one
    binding resolution, and read-only trace facts. It is not the stdlib
    :class:`~pypto.language.protocol.TraceContext` — recipe code never
    receives builder access, emission helpers, or the Frame.

    A context is a *view* over the authoring session
    (:class:`~pypto.language.authoring.session.AuthoringSession`) — the
    one-invocation-scoped owner of the region stack and the function units.
    The general recipe context subclasses this view and adds the function,
    body, and control effects; both spellings of composition (a construct's
    ``ctx.region(pl.spmd(...))`` and a recipe's) drive the same session state.
    """

    __slots__ = (
        "_session",
        "_binding_name",
        "_channel_names",
        "_single_call_body",
        "_spelling",
    )

    def __init__(
        self,
        session: AuthoringSession,
        binding_name: str | None,
        single_call_body: bool,
        spelling: str = "with",
    ) -> None:
        self._session = session
        self._binding_name = binding_name
        self._channel_names: set[str] = set()
        self._single_call_body = single_call_body
        self._spelling = spelling

    @property
    def _frame(self) -> PtoFrame:
        """The frame the session borrows — the sanctioned tracer-internal
        accessor (this module and the authoring runtime are the clients; recipe
        code never sees it)."""
        return self._session.caller_frame

    @property
    def _regions(self) -> list[object]:
        """The session's composed regions, outermost first (the construct
        driver and the kit's own close checks read it)."""
        return self._session._regions

    @property
    def _primary(self) -> object | None:
        """The session's primary region — the recipe's structural surface."""
        return self._session._primary

    @property
    def _primary_exit(self) -> object:
        """The primary region's close result (the closed scope statement)."""
        return self._session._primary_exit

    # —— composition ——

    def region(
        self,
        cm: object,
        *,
        as_name: str | None | _Unset = _UNSET,
        single_call_body: bool = False,
    ) -> _RegionScope:
        """A context manager composing one construct's region inside the recipe.

        ``with ctx.region(pl.spmd(...)) as tid:`` opens the construct's region
        at ``__enter__`` and closes it at ``__exit__``; the recipe's ``yield``
        sits inside the with-body, so the user's body executes inside the
        region and the close runs at the recipe's exit. ``as_name`` defaults
        to the recipe's own ``as`` target name — the composed construct's
        capture (e.g. the dispatch TaskId) mints its Var under the name the
        user bound. An explicit ``None`` opts out of capture.
        """
        return _RegionScope(self, cm, as_name, single_call_body)

    def enter(
        self,
        cm: object,
        *,
        as_name: str | None | _Unset = _UNSET,
        single_call_body: bool = False,
    ) -> Any:
        """Open one composed construct's region now (the explicit form of :meth:`region`).

        The pair ``enter(cm)`` / ``exit(cm)`` brackets the recipe's ``yield``
        the way the with-form's generator body does implicitly. Regions close
        in reverse open order (LIFO) — the same order the builder closes
        scopes in.
        """
        self._session._check_not_terminated("ctx.enter")
        if not isinstance(cm, SupportsTraceEnter):
            raise TypeError(
                f"ctx.enter: {cm!r} is not a trace construct — a composed region "
                "is an object implementing __trace_enter__ (pl.spmd, pl.at, another recipe)"
            )
        if not isinstance(cm, SupportsTraceExit):
            raise TypeError(
                f"ctx.enter: {type(cm).__name__} implements __trace_enter__ but not "
                "__trace_exit__ — the close hook is required to finalize the region it opens"
            )
        if any(region is cm for region in self._regions):
            raise TypeError(
                f"ctx.enter: this {type(cm).__name__} region is already open — a "
                "construct instance enters once per recipe use"
            )
        if as_name is _UNSET:
            as_name = self._binding_name
        assert not isinstance(as_name, _Unset)  # the sentinel resolved above
        frame = self._frame
        # The mediation the Frame performs for an ordinary ``with``: the carry
        # group must not be orphaned by an intervening region open, and a
        # pending declarable scope must open before the new region's enter
        # reads structural state or begins its own scope.
        frame._reject_pending_carry("a with")
        frame._flush_deferred_scopes(entering_incore=_enters_incore_scope(cm))
        binding = cm.__trace_enter__(frame.trace_context(), as_name, single_call_body)
        self._session.enter_region(cm)
        return binding

    def exit(self, cm: object) -> Any:
        """Close one composed region (a construct from :meth:`enter`/:meth:`region`,
        or a scope from :meth:`open_scope`).

        Returns the region's close result — for scope regions, the closed
        ``ScopeStmt``."""
        self._session.exit_region(cm)
        assert isinstance(cm, SupportsTraceExit)  # enter()/open_scope() checked the pairing protocol
        result: Any = cm.__trace_exit__(self._frame.trace_context())
        if cm is self._session._primary:
            self._session.note_primary_exit(result)
        return result

    def resolve_binding(self, binding: SupportsTraceBind, name: str | None = None) -> ir.Var | None:
        """Resolve one capture binding proxy to its ``ir.Var``.

        The capture channel of an ``as tid`` form: the enter hook returns a
        binding proxy whose Var the recipe wants to expose on its handle
        (e.g. a dispatch TaskId for a later ``deps=`` edge). The hook is pure
        for capture bindings — it returns the pre-created Var without
        registering anything — so calling it here has no binding side
        effect. ``name`` defaults to the recipe's own ``as`` name (the name
        the composed construct minted the Var under). Returns ``None`` when
        the proxy declines.
        """
        if not isinstance(binding, SupportsTraceBind):
            raise TypeError(
                f"ctx.resolve_binding: {binding!r} is not a binding proxy — the "
                "capture channel hands back the enter hook's return value"
            )
        if name is None:
            name = self._binding_name or ""
        result = binding.__trace_bind__(self._frame.trace_context(), name)
        if result is not None and not isinstance(result, ir.Var):
            raise TypeError(
                "ctx.resolve_binding: a capture binding resolves to an ir.Var, got "
                f"{result!r} — only as-tid capture proxies belong on this channel"
            )
        return result

    # —— the scope-construct lowering primitives ——

    def open_scope(
        self,
        kind: ir.ScopeKind,
        *,
        attrs: list[tuple[str, Any]],
        defer: bool = True,
        primary: bool = False,
        prologue: list[tuple[ir.Var, ir.Expr]] | None = None,
        **begin_kwargs: Any,
    ) -> ScopeRegion:
        """Open one scope directly — the recipe-authored lowering primitive.

        ``kind`` is drawn from the framework's closed scope vocabulary;
        ``begin_kwargs`` are the scope's own parameters (``name_hint``,
        ``level``/``role`` for Hierarchy scopes, ``split`` for InCore scopes,
        ``manual`` for Runtime scopes); ``attrs`` is the scope's attr list in
        canonical order — the recipe owns the order because it is the
        lowering policy.

        ``defer=True`` (the declarable-scope contract) defers the begin: the
        Frame opens the region at the body's first IR-emitting statement —
        folding in any buffered ``pl.set_cache_policy(...)`` declarations — or
        at :meth:`exit`, whichever comes first. That is also what makes the
        region a *declarable* scope, so only a construct that accepts
        cache-policy declarations may defer.

        ``defer=False`` opens the scope now, exactly as a construct whose attrs
        are fixed at entry does. Such a region accepts no cache-policy
        declaration — the Frame's routing never sees it, so a stray
        ``pl.set_cache_policy`` reports the position/kind rejection it always
        reported for that construct.

        ``prologue`` statements are the region's own first emissions (a
        for-form's loop-variable binding) — emitted right after an immediate
        begin, or at a deferred open before any body statement lands.

        The construct's own enter already flushed outer deferred scopes (so
        this recipe's pre-enter facts were visible to that flush), exactly as
        a hand-written scope construct's enter never flushes either.

        ``primary=True`` makes this region the recipe's structural surface —
        the one the Frame's deferred-scope and carrier probes see. The default
        (the first region opened) is right for a recipe whose single region
        *is* its lowering; a recipe that wraps its carrier in an outer scope
        (``pl.spmd``: a dispatch scope around an InCore carrier) names the
        carrier instead, because the carrier owns the body-decided open."""
        self._session._check_not_terminated("ctx.open_scope")
        region = ScopeRegion(
            kind,
            dict(begin_kwargs),
            attrs,
            builder=None if defer else self._frame.trace_context().builder,
            prologue=prologue,
        )
        self._session.register_scope_region(region, primary=primary)
        return region

    def open_carrier(
        self,
        kind: ir.ScopeKind,
        *,
        attrs: list[tuple[str, Any]],
        flush_on_emit: bool,
        prologue: list[tuple[ir.Var, ir.Expr]] | None = None,
        primary: bool = False,
        **begin_kwargs: Any,
    ) -> CarrierRegion:
        """Open a deferred scope whose begin the body decides (:class:`CarrierRegion`).

        ``flush_on_emit=True`` opens at the body's first IR-emitting statement
        (a wrapped body); ``False`` waits for the Frame's block-index trigger —
        the body proves itself inline by reading ``pl.tile.get_block_idx()``,
        and a dispatch body never opens it. ``prologue`` statements are the
        carrier's own first emissions (a loop variable binding the block
        index). Everything else matches :meth:`open_scope` — including
        ``primary``, which a recipe that wraps its carrier names explicitly.
        """
        self._session._check_not_terminated("ctx.open_carrier")
        region = CarrierRegion(
            kind, dict(begin_kwargs), attrs, flush_on_emit=flush_on_emit, prologue=prologue
        )
        self._session.register_scope_region(region, primary=primary)
        return region

    def capture_task_id(self, name: str) -> tuple[_TaskIdBinding, tuple[str, ir.Var]]:
        """Mint the TaskId capture for an ``as tid`` form.

        Creates the ``Scalar[TASK_ID]`` Var, emits the transient
        ``AssignStmt(tid, system.task_invalid())`` placeholder one statement
        before the scope (so ConvertToSSA has a def; the outliner drops it and
        synthesises the real binding), and returns ``(binding, attr)``: the
        binding proxy to yield as the ``as``-target, and the ``task_id_var``
        attr entry for the recipe to append at its canonical attr position."""
        builder = self._frame.trace_context().builder
        tid_var = builder.var(name, ir.ScalarType(DataType.TASK_ID))
        builder.assign(tid_var, ir.create_op_call("system.task_invalid", [], {}, ir.Span.unknown()))
        return _TaskIdBinding(tid_var), ("task_id_var", tid_var)

    def lower_deps(self, deps: Sequence[SubmitOperand | None] | None, construct: str) -> list[ir.Var]:
        """Lower a ``deps=[...]`` list to ``manual_dep_edges`` Vars.

        The one ``deps=`` lowering on the trace lane —
        :func:`pypto.language.authoring.dispatch.coerce_deps`, the implementation
        the dispatch effect (``pl.submit`` / ``ctx.submit``) shares: a bare
        TaskId / whole-array Var rides the direct lane; an element read of a
        TASK_ID array desugars into one fresh synthesized array; ``None``
        entries are the conditional-edge sentinel. The synthesized lets land
        in the caller's frame at the current insertion point — before the
        recipe opens its region, exactly where the parser emits them."""
        from pypto.language.authoring.dispatch import coerce_deps  # noqa: PLC0415 - the op/typing cycle

        return coerce_deps(deps, construct, self._frame)

    def loop_var(self) -> ir.Var:
        """Mint the loop variable of a ``for``-spelling use — the recipe's ``yield`` value.

        The Var carries the loop target's source name and ``INDEX`` type. The
        driver hands it to the Frame's target binding, which registers it
        under the target name without a ``LetStmt`` — the loop variable is
        the region's own value, not a computed one. Only the ``for`` spelling
        mints one; a ``with``-spelling recipe has no loop target."""
        if self._spelling != "for":
            raise TypeError(
                "ctx.loop_var() belongs to the `for` spelling — a `with` recipe "
                "yields its as-target (a ConstructHandle or capture binding) instead"
            )
        if not self._binding_name:
            raise TypeError("ctx.loop_var() needs the loop target's name")
        return self._frame.trace_context().builder.var(
            self._binding_name, ir.ScalarType(DataType.INDEX), ir.Span.unknown()
        )

    def open_loop(
        self,
        start: Any,
        stop: Any,
        step: Any,
        *,
        kind: str = "range",
        attrs: dict[str, Any] | None = None,
        var_name: str | None = None,
    ) -> LoopRegion:
        """Open one for-loop region directly — the loop-tier lowering primitive.

        The ``pl.range``-family's ``begin`` as a recipe capability: mints the
        loop variable (``var_name``, or the ``as``/loop-target name, or ``i``)
        and opens ``begin_for_loop`` at the current insertion point, returning
        the :class:`LoopRegion` — bracket the recipe's ``yield`` with its
        ``with`` form and the close (``end_for_loop``) runs at the recipe's
        exit, in LIFO order with every other region.

        ``kind`` is the loop-kind vocabulary (``range`` / ``parallel`` /
        ``unroll`` / ``pipeline``) with the same transfer rules the
        ``pl.range`` family carries (a ``range`` loop accepts ``break`` /
        ``continue``; ``parallel`` / ``unroll`` reject them). ``attrs`` is the
        ``ForStmt`` attr dict (``pipeline``'s stage count rides here).

        The loop-carried channels are the SSA pass's inference — the same
        inference that carries a ``pl.range`` accumulator: bind the initial
        value through :meth:`bind` before the open, rebind the name in the
        body (a handle's ``bind``, or the body's own rebinding), and
        ``ConvertToSSA`` creates the ``iter_args`` / ``return_vars`` — order
        and types follow the rebindings, exactly as for a hand-written loop.
        """
        for_kind = _LOOP_FOR_KINDS.get(kind)
        if for_kind is None:
            raise TypeError(f"ctx.open_loop kind must be one of {sorted(_LOOP_FOR_KINDS)}, got {kind!r}")
        # The loop-var dtype follows the bounds — the same shared rule the
        # native ``pl.range`` family applies (an explicitly typed ConstInt
        # bound set rebuilds a typed loop var), not a hardcoded INDEX.
        from pypto.language.constructs.loops import (  # noqa: PLC0415 - the constructs cycle
            infer_loop_var_dtype,
        )

        start_bound = _pin_loop_bound(start, "start")
        stop_bound = _pin_loop_bound(stop, "stop")
        step_bound = _pin_loop_bound(step, "step")
        builder = self._frame.trace_context().builder
        # The ``for`` spelling's loop variable carries the loop target's name;
        # a ``with``-spelling recipe (an accumulator region) has no target, and
        # its ``as`` name belongs to the handle — the loop variable is ``i``
        # unless the recipe names it.
        default_name = self._binding_name if self._spelling == "for" else "i"
        name = var_name if var_name is not None else (default_name or "i")
        loop_var = builder.var(name, ir.ScalarType(infer_loop_var_dtype(start_bound, stop_bound, step_bound)))
        builder.begin_for_loop(
            loop_var,
            start_bound,
            stop_bound,
            step_bound,
            kind=for_kind,
            attrs=attrs,
        )
        region = LoopRegion(loop_var, kind, session=self._session)
        self._session.register_scope_region(region, primary=False)
        return region

    def bind(self, name: str, value: object) -> object:
        """Bind one staged value under ``name`` in the caller's trace.

        The binding-family effect: recipe code is ordinary Python, so an
        unpacking it performs never routes through the rewriter — this
        capability is that route. It lands in the Frame's own bind path, so
        the structural machinery fires exactly as it does for a rewritten
        kernel: a projection carrying ``__trace_bind__`` names its shared
        node once, the name maps to one mutable ``ir.Var`` (``let`` on first
        definition, ``assign`` on rebinding — the rebind is a loop's carried
        next value), and the returned facade wraps that Var — a value a
        later effect (``deps=[...]``, a composed region, a return) accepts
        directly.

        The channel-name discipline: the bound name is a caller-trace name
        and must be fresh — it must not collide with the recipe's own
        ``as`` target (or loop variable) or with the user's locals; a
        colliding bind is refused (see the checks below).
        """
        session = self._session
        session._require_live()
        frame = session._require_borrowed("bind")
        session._check_not_terminated("bind")
        if not isinstance(name, str) or not name:
            raise TypeError(f"bind: a name is required, got {name!r}")
        # The channel-name discipline: a channel is a caller-trace name and
        # must be fresh — it must not collide with the recipe's own target
        # (the ``as`` handle or the loop variable) or with a user local. A
        # colliding first bind would silently rebind the user's name (the
        # channel rides ordinary caller-trace locals), so it is refused here;
        # a rebind of a name this recipe itself first-bound is the carried
        # next value and stays open.
        if name == self._binding_name:
            raise TypeError(
                f"bind: {name!r} is this recipe's own target name — a channel name "
                "must be fresh (it must not collide with the `as` handle or the loop variable)"
            )
        if name not in self._channel_names:
            _, prior = frame._require_rt().current_binding(name)
            if prior is not None:
                raise TypeError(
                    f"bind: {name!r} collides with a local of the caller's trace — a "
                    "channel name must be fresh; rebinding a user local would "
                    "silently overwrite it"
                )
            self._channel_names.add(name)
        return frame.rt_bind(name, value)

    # —— read-only trace facts ——

    @property
    def binding_name(self) -> str | None:
        """The name the user bound the recipe under.

        The ``with`` spelling: the ``as`` target name (``None`` without ``as``).
        The ``for`` spelling: the loop target's name — always bound, and what
        :meth:`loop_var` mints its Var under."""
        return self._binding_name

    @property
    def spelling(self) -> str:
        """The statement spelling this use entered through: ``"with"`` or ``"for"``.

                A recipe serving both spellings branches on this fact — the same
        generator lowers ``with construct(...)`` and ``for i in construct(...)``.
                The yield contract differs per spelling: a handle or capture binding
                for ``with`` (:meth:`ConstructHandle`, a binding proxy, or ``None``),
                the loop variable for ``for`` (an ``ir.Var`` from :meth:`loop_var`)."""
        return self._spelling

    @property
    def single_call_body(self) -> bool:
        """The rewriter's fact that the recipe's with-body is one call-valued statement.

        Constructs whose IR shape depends on the body shape (``pl.spmd``'s
        direct-dispatch vs InCore-wrapped form) read this fact; a recipe
        composing one forwards it through
        :meth:`region(single_call_body=...) <ConstructContext.region>`."""
        return self._single_call_body

    def is_inside_incore(self) -> bool:
        """Whether the innermost region is an InCore region."""
        return self._frame.trace_context().is_inside_incore()

    def is_inside_scope(self, scope_kind: ir.ScopeKind) -> bool:
        """Whether the build position is nested inside an open scope of ``scope_kind``."""
        return self._frame.trace_context().is_inside_scope(scope_kind)

    def function_type(self) -> ir.FunctionType:
        """The active function's ``FunctionType`` (its body's kind)."""
        return self._frame.trace_context().function_type()

    def is_inside_manual_scope(self) -> bool:
        """Whether the build position is nested inside an open MANUAL runtime scope."""
        return self._frame.trace_context().is_inside_manual_scope()

    def auto_scope_enabled(self) -> bool:
        """Does the active function own automatic runtime-scope placement?"""
        return self._frame.trace_context().auto_scope_enabled()

    def enclosing_user_split(self) -> ir.SplitMode | None:
        """The innermost open InCore scope's user-declared ``pl.split(...)`` mode.

        The fact behind the AIV-split mutual-exclusion rule: a function-level
        split (``optimizations=[pl.split(mode)]``) and an explicit
        ``pl.split_aiv`` region are exclusive mechanisms, and a recipe nested
        inside a split-declaring scope must reject the combination. ``None``
        means no open scope declared one."""
        return self._frame.trace_context().builder.enclosing_user_split()

    def sticky_dump_vars(self) -> list[ir.Var]:
        """The function's forward-sticky dump-tagged tensor Vars."""
        return self._frame.trace_context().sticky_dump_vars()


class _RegionScope:
    """The composition context manager behind :meth:`ConstructContext.region`.

    ``__exit__`` drives the region's close on the success path only — during
    an exception unwind the trace is discarded with its one-shot Sema, and no
    staged finalization runs (the tracer's own doctrine for staged
    constructs; the recipe's generator is closed by Python's unwinding
    instead).
    """

    __slots__ = ("_kit", "_cm", "_as_name", "_single_call_body")

    def __init__(
        self, kit: ConstructContext, cm: object, as_name: str | None | _Unset, single_call_body: bool
    ) -> None:
        self._kit = kit
        self._cm = cm
        self._as_name = as_name
        self._single_call_body = single_call_body

    def __enter__(self) -> Any:
        return self._kit.enter(self._cm, as_name=self._as_name, single_call_body=self._single_call_body)

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        del exc_val, exc_tb
        if exc_type is None:
            self._kit.exit(self._cm)
        return False  # never suppress — a failing recipe fails the trace


class ConstructRecipe:
    """The product of :func:`construct`: a callable construct factory.

    Calling the recipe builds one single-use construct instance — the object
    a ``with`` statement enters. The factory itself is reusable and stateless.
    """

    def __init__(self, function: types.FunctionType) -> None:
        if not isinstance(function, types.FunctionType):
            raise TypeError(f"@pl.authoring.construct expects a plain function, got {function!r}")
        if not inspect.isgeneratorfunction(function):
            raise TypeError(
                f"@pl.authoring.construct expects a generator function — {function.__name__!r} has "
                "no yield, so it cannot mark the body boundary"
            )
        self._function = function
        self.__name__ = function.__name__
        self.__qualname__ = function.__qualname__
        self.__doc__ = function.__doc__
        self.__module__ = function.__module__
        self.__wrapped__ = function

    def __call__(self, *args: Any, **kwargs: Any) -> _RecipeInstance:
        # Bind eagerly so a bad call fails at the factory — a constructor-surface
        # TypeError, matching a hand-written construct's __init__ — and so every
        # bound argument is readable on the instance before it is entered (the
        # constructor surface: pl.at's `level` is the pre-enter fact the Frame's
        # InCore hint reads).
        # The leading ``ctx`` parameter is kit-injected at enter, so it binds to
        # a placeholder here; the user-facing surface is the remaining parameters.
        bound = inspect.signature(self._function).bind(None, *args, **kwargs)
        bound.apply_defaults()
        facts = dict(bound.arguments)
        facts.pop("ctx", None)
        return _RecipeInstance(self, args, kwargs, facts=facts)

    def __repr__(self) -> str:
        return f"<construct recipe {self.__name__}>"


class CarrierRegion(ScopeRegion):
    """A deferred scope whose open decision belongs to a body fact.

    The plain ``with pl.spmd(...)`` form opens its inner InCore carrier only
    when the body proves itself inline — a per-block-index read — because a
    dispatch body stays bare; the wrapped forms open at the body's first
    emission. Both spellings defer the begin (so leading
    ``pl.set_cache_policy(...)`` declarations join the attrs) but differ in
    *when*: ``flush_on_emit=False`` hands the decision to the block-index
    trigger, which the Frame drives through :meth:`open_pending_carrier`.

    The class also carries the plain attributes the Frame's member checks
    probe (``_carrier_pending``, ``_carrier_flush_on_emit``) — the sanctioned
    open-world getattr pattern — so a recipe whose primary region is a carrier
    answers exactly what a hand-written carrier construct answers.
    """

    def __init__(
        self,
        kind: ir.ScopeKind,
        begin_kwargs: dict[str, Any],
        attrs: list[tuple[str, Any]],
        *,
        flush_on_emit: bool,
        prologue: list[tuple[ir.Var, ir.Expr]] | None = None,
    ) -> None:
        super().__init__(kind, begin_kwargs, attrs, builder=None, prologue=prologue)
        self._flush_on_emit = flush_on_emit

    # The Frame's carrier member checks.
    @property
    def _carrier_pending(self) -> bool:
        return self._scope_pending

    @property
    def _carrier_flush_on_emit(self) -> bool:
        return self._flush_on_emit

    def open_pending_carrier(self, builder: IRBuilder) -> None:
        """Open the carrier — the body read the per-block index."""
        self.open_deferred_scope(builder)


class _RecipeInstance:
    """One use of a construct recipe — the traced ``with`` construct.

    Implements the trace protocol on the recipe's behalf: the enter hook runs
    the recipe's generator to its single ``yield`` and returns the handle as
    the ``as``-target; the exit hook resumes the generator so its close code
    runs in order. The instance delegates the deferred-scope structural
    surface to its primary region (see :mod:`pypto.language.protocol`,
    ``_DeferredDeclarableScope``) so the Frame's member checks — written
    against construct objects — keep working through the recipe.
    """

    def __init__(
        self, recipe: ConstructRecipe, args: tuple[Any, ...], kwargs: dict[str, Any], facts: dict[str, Any]
    ) -> None:
        self._recipe = recipe
        self._args = args
        self._kwargs = kwargs
        self._facts = facts
        self._kit: ConstructContext | None = None
        self._session: AuthoringSession | None = None
        self._generator: Any = None
        # The Frame of the trace that opened the ``for`` spelling — captured
        # at ``__trace_iter__`` so ``begin`` can build the kit: the descriptor
        # ABI hands ``begin`` a raw builder, not the TraceContext.
        self._for_frame: PtoFrame | None = None
        # The loop kind a ``for``-spelling use authored (set by ``begin`` when
        # the recipe's innermost region is a LoopRegion — see ``begin``). The
        # ``_LoopDescriptor`` protocol's member: with it set, the interpreter's
        # transfer selection treats this construct as the loop it is.
        self.kind: str = ""

    # —— the single-use guard ——

    def _reject_reuse(self) -> None:
        """One construct instance serves one statement — either spelling."""
        if self._generator is not None:
            raise TypeError(
                f"{self._recipe.__name__}(...) enters once — a construct instance is "
                "single-use (the same rule as pl.spmd)"
            )

    def _drive_to_yield(self, kit: ConstructContext) -> Any:
        """Run the recipe's generator to its single ``yield`` — shared by both spellings.

        The yielded value crosses back to the spelling's driver: the ``as``
        target for ``with``, the loop variable for ``for``. The carrier
        channel is injected as an instance attribute when the primary region
        has one — the runtime-checkable protocol isinstance probes with
        ``inspect.getattr_static``, which bypasses ``__getattr__``, so the
        injection is the only delegation it sees (see the note at
        :attr:`open_pending_carrier`). A scope-opened region has no carrier;
        the recipe then answers "not a carrier", exactly like a hand-written
        scope construct without one."""
        generator = self._recipe._function(kit, *self._args, **self._kwargs)
        try:
            handle = next(generator)
        except StopIteration:
            raise TypeError(
                f"@pl.authoring.construct {self._recipe.__name__!r} has no yield — a recipe "
                "yields exactly once, and the yield is the body boundary"
            ) from None
        self._generator = generator
        primary = kit._primary
        if primary is not None and hasattr(primary, "open_pending_carrier"):
            self.open_pending_carrier = getattr(primary, "open_pending_carrier")
        return handle

    # —— the traced with-protocol ——

    def __trace_enter__(
        self, ctx: TraceContext, as_name: str | None = None, single_call_body: bool = False
    ) -> Any:
        """Run the recipe to its yield: the handle becomes the ``as``-target."""
        self._reject_reuse()
        # The kit's sanctioned access: this module and the authoring runtime
        # are the tracer-internal clients; recipe code never sees the session.
        from pypto.language.authoring.session import AuthoringSession  # noqa: PLC0415 - the session cycle

        self._session = AuthoringSession(ctx._frame)
        kit = ConstructContext(self._session, as_name, single_call_body)
        self._kit = kit
        handle = self._drive_to_yield(kit)
        if as_name is None:
            if handle is not None:
                raise TypeError(
                    f"{self._recipe.__name__}(...) yields a handle — bind it: "
                    f"with {self._recipe.__name__}(...) as ..."
                )
            return None
        if handle is None:
            raise TypeError(
                f"{self._recipe.__name__}(...) yielded no handle for 'as {as_name}' — "
                "yield the handle (a ConstructHandle or binding proxy)"
            )
        if not isinstance(handle, SupportsTraceBind):
            raise TypeError(
                f"{self._recipe.__name__}(...) must yield a pl.authoring.ConstructHandle (or a "
                f"binding proxy) for 'as {as_name}', got {handle!r}"
            )
        return handle

    # —— the traced for-protocol ——

    def __trace_iter__(self, ctx: TraceContext) -> Any:
        """Open the ``for`` spelling: hand out the loop-target binding proxy.

        The shared proxy every trace iterable returns (the one ``pl.range``
        and ``pl.spmd`` hand out): it defers construction to the first
        target binding, where the loop variable's source name is known and
        the recipe's generator starts. The Frame is captured here because
        the descriptor ABI hands ``begin`` a raw builder — the kit builds
        its facts from the Frame, not from that builder."""
        self._reject_reuse()
        self._for_frame = ctx._frame
        from pypto.language.constructs.loops import _LoopTargetBinding  # noqa: PLC0415 - the constructs cycle

        return _LoopTargetBinding(self)

    def begin(
        self,
        builder: IRBuilder,
        name: str,
        already_in_incore: bool = False,
        sticky_dump_vars: list[ir.Var] | None = None,
    ) -> ir.Var:
        """Construct the loop at the first target binding — the ``_ForDescriptor`` hook.

        Drives the recipe's generator to its yield: the pre-yield code opens
        the loop's regions and the yielded value is the loop variable — an
        ``ir.Var`` minted through ``ctx.loop_var()`` — which the Frame
        registers under the target name without a ``LetStmt``. The kit reads
        its facts (``sticky_dump_vars``, the InCore position) from the Frame,
        so the descriptor ABI's builder parameters are not the recipe's
        surface. A failed drive discards the whole trace, so a recipe that
        yields a ``with``-shaped value (a handle, a capture binding) is loud."""
        del builder, already_in_incore, sticky_dump_vars
        self._reject_reuse()
        frame = self._for_frame
        if frame is None:
            raise RuntimeError(
                f"{self._recipe.__name__}(...).begin without __trace_iter__ — the for "
                "spelling opens through the trace-iterable hook only"
            )
        from pypto.language.authoring.session import AuthoringSession  # noqa: PLC0415 - the session cycle

        self._session = AuthoringSession(frame)
        kit = ConstructContext(self._session, name, single_call_body=False, spelling="for")
        self._kit = kit
        loop_var = self._drive_to_yield(kit)
        if not isinstance(loop_var, ir.Var):
            raise TypeError(
                f"@pl.authoring.construct {self._recipe.__name__!r} must yield the loop variable "
                f"(ctx.loop_var()) for the `for` spelling — got {loop_var!r}; a recipe "
                "that yields a handle or a capture binding serves the `with` spelling only"
            )
        # A recipe whose innermost region is a loop region authored a real
        # ``ForStmt``: the construct advertises the loop kind so the
        # interpreter's transfer rules treat it as the loop it is (break /
        # continue target it; parallel/unroll reject them, as ``pl.range``'s
        # family does). A scope-shaped recipe keeps the loop-protocol
        # default — its region crosses transfers, exactly as ``pl.spmd``'s
        # for-form does.
        innermost = kit._regions[-1] if kit._regions else None
        if isinstance(innermost, LoopRegion):
            self.kind = innermost.kind
        if any(getattr(region, "_scope_pending", False) for region in kit._regions) and not getattr(
            self, "_carrier_pending", False
        ):
            # The structural surface activate_target reads: a region the
            # recipe left pending must be exposed by it, or the Frame never
            # registers it and the body's first emission lands outside it —
            # the silent mis-lowering. Carriers are exposed through
            # primary=True; an immediate-open recipe (split_aiv) has nothing
            # pending and needs no surface at all.
            raise TypeError(
                f"@pl.authoring.construct {self._recipe.__name__!r} left a deferred region open that "
                "its structural surface does not expose — open the carrier with "
                "ctx.open_carrier(..., primary=True), or open the scopes immediately "
                "(defer=False)"
            )
        return loop_var

    def __trace_exit__(self, ctx: TraceContext) -> Any:
        """Resume the recipe past its yield so the close code runs in order.

        Returns the primary region's close result — the closed scope
        statement, matching what a hand-written construct's exit returns (the
        Frame attaches deferred construct comments to it).
        """
        del ctx
        kit = self._kit
        generator = self._generator
        self._generator = None
        if kit is None or generator is None:
            # Never begun (the enter hook failed and the construct was never
            # installed) — the no-op close of a never-begun construct.
            return None
        try:
            generator.send(None)
        except StopIteration:
            pass
        else:
            raise TypeError(
                f"@pl.authoring.construct {self._recipe.__name__!r} yielded twice — one yield "
                "is the body boundary; code after it finalizes and returns"
            )
        if kit._regions:
            raise TypeError(
                f"@pl.authoring.construct {self._recipe.__name__!r} left "
                f"{type(kit._regions[-1]).__name__} region(s) open — every ctx.enter "
                "pairs with a ctx.exit (a with ctx.region(...) closes itself)"
            )
        primary_exit = kit._session._primary_exit
        kit._session.close()
        return primary_exit

    # —— structural delegation to the primary region ——

    # The deferred-scope protocol is probed with ``inspect.getattr_static``
    # (runtime-checkable protocol ``isinstance``), which bypasses
    # ``__getattr__`` — so the protocol surface must sit on the class, each
    # method forwarding to the primary region at call time. The plain-attribute
    # probes (``_carrier_pending``, ``_scope_pending``, ``_pending_begin``,
    # ``_carrier_flush_on_emit``) go through runtime ``getattr`` and delegate
    # through ``__getattr__``. The one exception is the carrier channel:
    # ``open_pending_carrier`` exists only for recipes whose primary region
    # has one, so the enter hook injects it as an instance attribute — a
    # scope-opened recipe answers "not a carrier", exactly like a
    # hand-written scope construct without one.

    # Injected by ``__trace_enter__`` when the primary region has one (see
    # above). An annotation, not a class attribute: ``getattr_static`` must
    # not see it until the injection.
    open_pending_carrier: Any

    def open_deferred_scope(self, builder: Any) -> None:
        """Open the primary region's deferred scope (its first statement arrived)."""
        primary = self._require_primary("open_deferred_scope")
        primary.open_deferred_scope(builder)

    def attach_cache_policy(self, var: ir.Var, policy: int) -> None:
        """Buffer one ``pl.set_cache_policy`` declaration on the primary region."""
        primary = self._require_primary("attach_cache_policy")
        primary.attach_cache_policy(var, policy)

    def _require_primary(self, operation: str) -> Any:
        kit = self._kit
        primary = kit._primary if kit is not None else None
        if primary is None:
            raise TypeError(
                f"@pl.authoring.construct {self._recipe.__name__!r} cannot {operation}: no region "
                "was entered — a recipe deferring a scope opens one through "
                "ctx.region(...) / ctx.enter(...)"
            )
        return primary

    def __getattr__(self, name: str) -> Any:
        """Delegate facts and the primary region's structural surface.

        The recipe's declared ``facts`` answer first: constructor parameters
        the Frame may probe *before* enter (``pl.at``'s ``level``, read by the
        pre-enter InCore hint). Then the primary region's structural surface:
        the Frame's deferred-scope machinery probes the construct object for
        plain attributes (``_carrier_pending``, ``_scope_pending``,
        ``_pending_begin``, ``_carrier_flush_on_emit``) — the sanctioned
        open-world getattr pattern. Delegating to the primary region gives a
        composed construct or an opened scope the same answers its
        hand-written form gives (the protocol methods sit on this class — see
        the note above — while the attributes flow through here). Before the
        recipe opens a region there is no primary, so every probe falls
        through as absent — the same answer an ordinary construct without
        the attribute gives.
        """
        facts = self.__dict__.get("_facts")
        if facts and name in facts:
            return facts[name]
        kit = self.__dict__.get("_kit")
        primary = kit._primary if kit is not None else None
        if primary is None:
            raise AttributeError(name)
        return getattr(primary, name)

    # —— host guards (the traced protocol is the only entrance) ——

    def __enter__(self) -> Any:
        raise RuntimeError(
            f"{self._recipe.__name__}(...) is only valid inside a decorated "
            "@pl.program / @pl.function body (the tracing frontend routes it through "
            "the __trace_enter__ trace-protocol hook). If you're seeing this, the "
            "surrounding function was not decorated."
        )

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass

    def __iter__(self) -> Any:
        raise RuntimeError(
            f"{self._recipe.__name__}(...) is only valid inside a decorated "
            "@pl.program / @pl.function body (the tracing frontend routes the `for` "
            "spelling through the __trace_iter__ trace-protocol hook). If you're "
            "seeing this, the surrounding function was not decorated."
        )


@overload
def construct(function: types.FunctionType) -> ConstructRecipe: ...


@overload
def construct(function: None = None) -> Callable[[types.FunctionType], ConstructRecipe]: ...


def construct(
    function: types.FunctionType | None = None,
) -> ConstructRecipe | Callable[[types.FunctionType], ConstructRecipe]:
    """Declare a construct recipe — one ``with`` construct authored as a generator.

    The decorated function receives the :class:`ConstructContext` as its first
    argument, runs to its single ``yield`` when the construct is entered, and
    finishes when the construct exits:

    .. code-block:: python

        @pl.authoring.construct
        def blocked_spmd(ctx, shape, tile, *, name_hint=""):
            with ctx.region(pl.spmd(block_count(shape, tile), name_hint=name_hint)) as tid:
                yield BlockGrid(shape, tile, ctx.resolve_binding(tid))

    The bound constructor arguments are readable on the instance before it
    is entered (the constructor surface): ``pl.at``'s ``level`` is the
    pre-enter fact the Frame's InCore hint reads to decide whether an outer
    deferred carrier absorbs instead of opening, so it must be visible
    before any region exists (facts answer through the instance; regions
    answer only after enter).

    See :mod:`pypto.language.authoring.construct` for the contract and the
    layering rule.
    """
    if function is None:  # the parenthesized form: @pl.authoring.construct(...)
        return functools.partial(construct)
    if not isinstance(function, types.FunctionType):
        raise TypeError(f"@pl.authoring.construct expects a plain function, got {function!r}")
    return ConstructRecipe(function)
