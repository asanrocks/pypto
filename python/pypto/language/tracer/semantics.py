# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: E501

"""PtoSema and PtoFrame — the PTO plugin over the facades.

**PtoSema** is one function-trace compilation: it owns one `IRBuilder`, the
compilation configuration, the trace result, and creates one fresh
:class:`PtoFrame` (``begin_frame``/``bind_parameters``). A mutable PtoSema is
one-shot — reuse after success or failure is rejected, never reset; the
decorator adapter acquires a fresh Sema per trace.

**PtoFrame** handles bindings, expression statements, returns, staged
Construct construction, and the logical value algebra. It owns the mapping
from source names to mutable `ir.Var`s, pending comments, and detached
pending `iter_args`. It retains **no active-Construct ordering mirror** —
structural questions (staged control flow, nearest interface tags, pending
result constructs, and host-binding facts) are queries to the trace
interpreter.

**PtoConstructs** (``tracer/constructs.py``) own staged region data; the
`TraceContext` in ``language/protocol.py`` is the tracing context passed to
stdlib trace-protocol hooks — never a raw builder, the Sema, or the Frame
itself.

Value operations are the shared typed layer both frontends execute: a facade
wraps the ``ir.Expr`` an operation produces. PtoFrame produces **only the
structural IR**: statements land through the ``ir/builder.py`` API. Binding
uses a mutable `ir.Var`, emitting ``LetStmt`` for its first definition and
``AssignStmt`` for later definitions, then returns a facade wrapping that
variable.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from typing_extensions import TypeIs

from pypto._function_attrs import AUTO_SCOPE_ATTR  # noqa: PLC0415 - the single canonical attr-key source
from pypto.ir.builder import IRBuilder
from pypto.ir.utils import current_pinned_span, reset_pinned_span, set_pinned_span
from pypto.language.protocol import (
    ResultBindingTarget,
    SplitAivTarget,
    SupportsTraceBind,
    SupportsTraceEnter,
    SupportsTraceExit,
    SupportsTraceIter,
    SupportsTraceStmt,
    TraceContext,
    YieldTarget,
    _ConstructAttachingBinding,
    _DeferredDeclarableScope,
    _ExistingVarBinding,
    _ForDescriptor,
    _LoopDescriptor,
    _ResultBindingProxy,
    _RuntimeDeferredDeclarableScope,
    _RuntimeDeferredSpmdCarrier,
    is_staged,
    is_staged_annotation,
)
from pypto.language.tracer.collector.interface import (
    BindingResult,
    FinalResultBinding,
    ForOpen,
    HostBinding,
    PartialResultBinding,
    StagedBinding,
    WithOpen,
)
from pypto.language.tracer.collector.interpreter import IfConstruct, WhileConstruct, ZeroOperandYieldAction
from pypto.language.typing import (
    Scalar,
    Tensor,
    Tile,
    annotation_to_type,
    wrap_expr,
)
from pypto.language.typing._host import coerce_host_scalar
from pypto.language.typing.tensor import CachePolicyDecl, DumpTagMarker
from pypto.language.typing.type_utils import _UnwrapsToExpr
from pypto.pypto_core import DataType, ir

if TYPE_CHECKING:
    from pypto.language.intrin import _LeakBundle
    from pypto.language.tracer.collector.interpreter import Construct, Interpreter

__all__ = ["PtoFrame", "PtoSema", "is_staged"]

# The stdlib-yielded value kinds: a facade is staged; a bare literal, a raw
# ir.Expr, a list — anything else — is a host value stored in the
# interpreter's instrumented locals. A trace-protocol object such as a
# ``RangeIterator`` or ``AtContext`` participates in staging by **protocol
# membership** — the trace hook is the marker (no central registry).
# ``FACADE_TYPES`` is derived from the closed static union (:mod:`pypto.language.facade`).
from pypto.language.ambient import active_frame  # noqa: PLC0415 - the ambient channel's own funnel
from pypto.language.diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    SSAViolationError,
)
from pypto.language.facade import FACADE_TYPES, Facade  # noqa: PLC0415 - avoid the typing/init cycle
from pypto.language.typing.dynamic import DynVar  # noqa: PLC0415 - avoid the typing/init cycle


def _ambient_pre_emit_flush() -> None:
    """The builder's pre-emit hook: flush the *ambient* frame's deferred scopes.

    A borrowing expansion frame shares the root's builder but owns its own
    deferred-scope list (the spmd for-form carrier its body opens). Routing
    the flush to the root frame alone would leave the child's carrier
    unopened: its body's first statement would land in the Spmd scope ahead
    of the loop variable's ``get_block_idx`` definition — the free-var use
    the SSA verifier rejects. The ambient frame is exactly the frame whose
    statements are being emitted."""
    frame = active_frame()
    if frame is not None:
        frame._flush_deferred_scopes()


_FACADE_TYPES = FACADE_TYPES

# Validated at import so bare name comparisons can never drift from the
# registry (see .claude/rules/operator-identity-checks.md).
_ALLREDUCE = ir.get_op("pld.tensor.allreduce").name
_GET_BLOCK_IDX = ir.get_op("tile.get_block_idx").name


def _unwrap_facade(value: object) -> ir.Expr:
    """Unwrap a staged facade to its IR expression after an ``is_staged`` guard.

    A trace-protocol object without ``unwrap`` is rejected; such an object is
    not a binding value in a well-formed trace, and the check keeps the
    language boundary type-sound."""
    if not isinstance(value, _UnwrapsToExpr):
        raise TypeError(f"expected a staged facade with unwrap(), got {value!r}")
    return value.unwrap()


def _contains_allreduce(value: object) -> bool:
    """Find an allreduce result nested inside a host container."""
    if is_staged(value):
        expr = _unwrap_facade(value)
        return isinstance(expr, ir.Call) and expr.op.name == _ALLREDUCE
    if _is_sequence(value):
        return any(_contains_allreduce(item) for item in value)
    return False


def _is_sequence(v: object) -> TypeIs[Sequence[object]]:
    """Narrow a host container to its read-only sequence view (the
    isinstance-list/tuple check pyright cannot narrow by itself)."""
    return isinstance(v, (list, tuple))


def _base_facade_class(cls: type[object]) -> type[Facade] | None:
    """The closed-vocabulary base facade of ``cls``, or ``None`` outside it.

    ``FACADE_TYPES`` is the derived isinstance set (``tuple[type[object],
    ...]``), so the MRO walk cannot narrow past ``object``; the cast is
    confined to this one lookup, the shared owner of the claim. Both
    callers (the rewrap and the bind-path class query) route through it.
    """
    for base in cls.__mro__:
        if base in _FACADE_TYPES:
            return cast("type[Facade]", base)
    return None


class _BlockIdxReadProbe(ir.IRVisitor):
    """Expression-tree probe for a ``tile.get_block_idx`` call.

    Only consulted while a deferred spmd carrier is pending, so the per-bind
    walk never runs on the common path."""

    def __init__(self) -> None:
        super().__init__()
        self.found = False

    def visit_call(self, op: ir.Call) -> None:
        if op.op.name == _GET_BLOCK_IDX:
            self.found = True
        super().visit_call(op)


def _expr_reads_block_index(expr: ir.Expr) -> bool:
    """Whether ``expr``'s tree contains a per-block-index read."""
    probe = _BlockIdxReadProbe()
    probe.visit_expr(expr)
    return probe.found


def _enters_incore_scope(context: object) -> bool:
    """Whether a with-construct about to be entered opens its own InCore region.

    The ``pl.at`` family: a CORE_GROUP-level region is the InCore scope (a
    bare nested ``at`` under a wrapped ``pl.spmd`` absorbs the carrier — see
    ``_flush_deferred_scopes``). Other levels open Hierarchy scopes and never
    absorb."""
    level = getattr(context, "level", None)
    return level is not None and getattr(level, "name", "") == "CORE_GROUP"


class _DeferredScopeEntry(NamedTuple):
    """One pending declarable scope, in nesting (registration) order.

    ``construct`` is the with/for descriptor that deferred its scope begin
    (see ``_DeferredDeclarableScope``). ``flush_on_emit`` is False only for a
    plain ``with pl.spmd(...)`` carrier — its open decision belongs to the
    per-block-index read (a dispatch body stays bare), so the first-emission
    flush skips it."""

    construct: _DeferredDeclarableScope
    flush_on_emit: bool


# The position and rejection wording for ``pl.set_cache_policy`` mirrors the
# AST parser's (``_handle_set_cache_policy`` / ``_collect_scope_cache_policy_decls``);
# the two frontends keep their own copies because the shared language layer
# must not import either frontend and the parser keeps its constants on the
# parser class.
_CACHE_POLICY_POSITION_HINT = (
    "Write pl.set_cache_policy(t, pl.CachePolicy.BYPASS) as a standalone statement "
    "directly inside a `with pl.at(...):` / `with pl.spmd(...):` scope body (or a "
    "`for i in pl.spmd(...):` body), before the scope's first statement. It declares "
    "a property of the whole scope, so it cannot sit outside one, nor nested inside "
    "an `if` / `for` within one — a conditionally-executed declaration is a promise "
    "the compiler cannot check. To annotate a single access instead, use "
    "pl.load(..., cache=pl.CachePolicy.BYPASS)."
)
_CACHE_POLICY_ADJACENCY_HINT = (
    "The declaration joins the scope's attrs, which are fixed when the scope begins "
    "— write it as one of the scope's first statements (the printer re-emits "
    "declarations first, so printed output always satisfies this)."
)


def _rewrap_as_base_facade(facade_cls: type[Facade], expr: ir.Expr) -> Facade:
    """Re-wrap an IR expr as the expr-wrapping facade of the same class (the
    param facade after the param Var exists). Subclasses normalize to their
    ``_FACADE_TYPES`` ancestor (e.g. the consumed ``_LoopTargetBinding`` — a
    ``Scalar`` subclass that doubles as the loop-var facade — re-wraps as a
    plain ``Scalar``), because only the base facades accept ``expr=``.

    Loud on a class outside the closed union — the silent fall-through
    to ``facade_cls(expr=expr)`` would fabricate a facade the vocabulary
    never admitted."""
    base = _base_facade_class(facade_cls)
    if base is None:
        raise TypeError(f"not a facade class: {facade_cls!r} has no ancestor in the closed facade vocabulary")
    return base(expr=expr)


def _predicate_expr(pred: object) -> ir.Expr:
    """The if/while predicate: a staged value unwraps to its expression."""
    if isinstance(pred, Scalar):
        return pred.unwrap()
    raise TypeError(f"a staged predicate must be a pl.Scalar, got {pred!r} (bool predicates are host values)")


class PtoSema:
    """One function-trace compilation.

    Constructed per kernel by the decorator adapter (the function name and
    return type are the adapter's read of the kernel's annotations). The
    function context is opened by the decorator-adapter entrance
    (``bind_parameters``) or, for direct use, by ``begin_frame``; the frame's
    ``finish`` finalizes it on normal completion. One-shot: ``begin_frame``
    succeeds once and the Sema is never reset.
    """

    def __init__(
        self,
        name: str,
        return_type: ir.Type | None = None,
        function_type: ir.FunctionType = ir.FunctionType.Opaque,
        level: ir.Level | None = None,
        role: ir.Role | None = None,
        attrs: dict[str, Any] | None = None,
        strict_ssa: bool = False,
        requires_runtime_binding: bool = False,
        source_span: ir.Span | None = None,
    ) -> None:
        self._name = name
        self._return_type = return_type
        # The span-pin token: saved on the first row-span event, restored
        # when the frame finishes (see note_span).
        self._span_token: contextvars.Token[ir.Span | None] | None = None
        # The text-parser strict_ssa toggle (round-trip dialect, NOT ergonomic): when on, ``bind``
        # creates a fresh ``let`` per binding and raises on a rebind of an
        # already-bound name (the guardrail) — opting out of mutable-local
        # assignment.
        self._strict_ssa = strict_ssa
        # The function-classification facts: the type is
        # known at Marker creation and flows Marker → PtoSema.__init__ →
        # begin_function(type=…) at construction.
        self._function_type = function_type
        self._level = level
        self._role = role
        self._attrs = attrs
        self._requires_runtime_binding = requires_runtime_binding
        self._source_span = source_span if source_span is not None else ir.Span.unknown()
        self._builder = IRBuilder()
        self._frame: PtoFrame | None = None
        self._frame_started = False
        self._result: ir.Function | None = None
        # The borrowing expansion parent: ``None`` for a root Sema (the
        # default), or the root Sema whose builder and function context an
        # in-trace expansion borrows.
        self._parent: PtoSema | None = None
        # The forward-sticky ``pl.dump_tag`` set (per function, reset at
        # trace): the tagged tensor Vars, merged into every *subsequent*
        # kernel call and non-Runtime scope that consumes the Var. A rebind
        # of the tagged name drops its tag (the dump_tag contract: the new
        # value is not covered).
        self._dump_tagged: list[ir.Var] = []
        # The param-declared extent symbols (annotation-determined
        # dyn-dims): the id of every shape-extent Var declared by the
        # function's own params — the only extents codegen materializes.
        self._param_dim_symbols: set[int] = set()

    # —— the borrowing expansion entrance ——
    @classmethod
    def _for_expansion(cls, parent: PtoSema, owner: str) -> PtoSema:
        """The borrowing expansion Sema for one in-trace expansion.

        Borrows the parent's IRBuilder and current insertion point; the
        frame creates no function, and finalization never ends one. Function
        classification, extent-symbol, and dump-tag state delegate to the
        parent — the active function is the parent's.
        """
        sema = cls(
            name=owner,
            function_type=parent._function_type,
            level=parent._level,
            role=parent._role,
            attrs=parent._attrs,
            strict_ssa=parent._strict_ssa,
            source_span=parent._source_span,
        )
        sema._builder = parent._builder
        sema._parent = parent
        sema._param_dim_symbols = set(parent._param_dim_symbols)
        return sema

    @property
    def expansion_parent(self) -> PtoSema | None:
        """The root Sema whose builder this Sema borrows, or ``None`` for a root Sema."""
        return self._parent

    @property
    def name(self) -> str:
        """The display name of the function this Sema is building (an
        expansion Sema carries the expanded body's name)."""
        return self._name

    # —— the ``pl.dump_tag`` marker (traced) ——
    def record_dump_tag(self, var: ir.Var) -> None:
        """Record ``var`` in the function's forward-sticky dump set."""
        if self._parent is not None:
            # The borrowing expansion: the active function is the parent's.
            return self._parent.record_dump_tag(var)
        if self._function_type not in (ir.FunctionType.Orchestration, ir.FunctionType.Inline):
            raise TypeError(
                "pl.dump_tag() is only valid inside an Orchestration or Inline "
                "function (the traced marker is consumed by the "
                "kernel-call dump_vars merge)"
            )
        if not isinstance(var.type, ir.TensorType):
            raise TypeError(
                f"pl.dump_tag() argument is not a tensor (got {var.type}) — only "
                "tensors can be selectively dumped"
            )
        if not any(v is var for v in self._dump_tagged):
            self._dump_tagged.append(var)

    def is_dump_tagged(self, var: ir.Var) -> bool:
        """Is ``var``'s identity in the function's forward-sticky dump set?"""
        if self._parent is not None:
            return self._parent.is_dump_tagged(var)
        return any(v is var for v in self._dump_tagged)

    def sticky_dump_vars(self) -> list[ir.Var]:
        """The forward-sticky tagged tensor Vars — the scope-entry capture
        (the parser's ``_merge_forward_sticky_dump`` source: every
        non-Runtime scope that becomes a kernel dispatch merges these into
        its ``dump_vars`` attr at entry)."""
        if self._parent is not None:
            return self._parent.sticky_dump_vars()
        return [v for v in self._dump_tagged if isinstance(v.type, ir.TensorType)]

    def clear_dump_tag(self, var: ir.Var) -> None:
        """Drop ``var``'s tag — the rebind invalidation. The dump_tag
        contract ("reassigning q produces a new value that the prior tag
        does not cover"): the one-cell-per-name design assigns through the
        tagged Var on rebind, so the identity would otherwise survive the
        reassignment and the tag would cover the new value."""
        if self._parent is not None:
            return self._parent.clear_dump_tag(var)
        if self._dump_tagged:
            self._dump_tagged = [v for v in self._dump_tagged if v is not var]

    # —— the decorator-adapter entrance ——
    def _prepare_frame(self) -> PtoFrame:
        """Create the one-shot frame without starting it.

        The trace interpreter's first ``enter_frame`` starts the frame it
        finds prepared here (the idempotent handout); a directly driven
        trace prepares the frame to install it as the active Frame before
        execution begins.
        """
        if self._frame is not None or self._frame_started:
            raise RuntimeError(
                "PtoSema is one-shot: this Sema already began a function trace "
                "(acquire a fresh Sema per trace)"
            )
        frame = PtoFrame(self)
        self._frame = frame
        return frame

    @staticmethod
    def _param_extent_symbols(types: Iterable[ir.Type]) -> set[int]:
        """The ids of the shape-extent Vars declared by parameter types —
        the only extents the Orchestration codegen materializes, hence the
        only ones ``pl.tensor.dim`` may fold onto."""
        symbols: set[int] = set()
        for param_type in types:
            if isinstance(param_type, ir.ShapedType):
                for extent in param_type.shape:
                    if isinstance(extent, ir.Var):
                        symbols.add(id(extent))
        return symbols

    def bind_parameters(
        self,
        annotations: dict[str, object],
        directions: dict[str, ir.ParamDirection] | None = None,
        scope: Mapping[str, object] | None = None,
    ) -> list[Any]:
        """Open the function context and create parameter variables.

        Returns typed facades in annotation order for the decorator adapter to
        pass to the instrumented kernel; the interpreter's ``enter_frame``
        then stores them unchanged. A later user binding to the same name
        assigns to the existing parameter variable. ``directions`` carries
        the ``Out``/``InOut`` parameter directions; ``scope`` is the
        decoration-boundary name scope (the caller's visible names — the AST's
        ``closure_vars``), needed for the dynamic-symbol shadowing rule.

        A ``pl.Scalar[pl.INDEX]`` parameter whose name matches a
        ``pl.dynamic`` extent symbol binds that symbol to the parameter's
        Var — later annotations naming the symbol resolve to the parameter
        (annotations resolve in declaration order). A parameter arriving
        *after* an annotation that already read the symbol is rejected: the
        earlier type would keep pointing at a different Var, which later
        reads as an unbound symbol and materializes a second, redundant
        extent argument (the parser frontend's
        ``_maybe_bind_shadowed_symbol`` discipline)."""
        frame = self._prepare_frame()
        # The param-declared extent symbols: collect the shape-extent Vars
        # the annotations declare — the only extents the Orchestration
        # codegen materializes, hence the only ones ``pl.tensor.dim`` may
        # fold onto. The same walk tracks which Vars *earlier*
        # annotations already read, for the shadowing rule below.
        symbols: set[int] = set()
        symbols_read: set[int] = set()
        dynvars: dict[str, DynVar] = {
            name: value for name, value in (scope or {}).items() if isinstance(value, DynVar)
        }
        facades: list[object] = []
        for name, ann in annotations.items():
            if not is_staged(ann):
                facades.append(
                    frame.bind_param(name, ann, direction=(directions or {}).get(name, ir.ParamDirection.In))
                )
                continue
            ann_type = annotation_to_type(ann)
            if isinstance(ann_type, ir.ShapedType):
                extents: list[object] = list(ann_type.shape)
                tensor_view = getattr(ann_type, "tensor_view", None)
                if tensor_view is not None:
                    # A view's valid_shape/stride name the same extent-symbol
                    # world as the shape — the shadowing rule must see them.
                    extents.extend(tensor_view.valid_shape)
                    extents.extend(tensor_view.stride)
                for extent in extents:
                    if isinstance(extent, ir.Var):
                        symbols.add(id(extent))
                        symbols_read.add(id(extent))
            shadowed = dynvars.get(name)
            if (
                shadowed is not None
                and isinstance(ann_type, ir.ScalarType)
                and ann_type.dtype == DataType.INDEX
            ):
                previous = shadowed._ir_var
                if previous is not None and id(previous) in symbols_read:
                    raise ParserTypeError(
                        f"Parameter '{name}' shadows the dynamic symbol '{name}', "
                        f"which an earlier parameter's type annotation already uses",
                        hint=(
                            f"Declare '{name}: pl.Scalar[pl.INDEX]' before the parameter whose "
                            f"pl.TensorView(valid_shape=...) names it, so the annotation resolves to "
                            f"the parameter"
                        ),
                    )
            facades.append(
                frame.bind_param(name, ann, direction=(directions or {}).get(name, ir.ParamDirection.In))
            )
            if (
                shadowed is not None
                and isinstance(ann_type, ir.ScalarType)
                and ann_type.dtype == DataType.INDEX
            ):
                # Bind forward: every later read of this DynVar (annotation
                # or body) unwraps to the parameter's Var.
                param_var = frame._locals[name]
                shadowed._ir_var = param_var
                shadowed.expr = param_var
        self._param_dim_symbols = symbols
        return facades

    def bind_typed_parameters(
        self, params: Sequence[tuple[str, ir.Type, ir.ParamDirection]]
    ) -> list[_UnwrapsToExpr]:
        """Open the function context from typed parameter triples.

        The typed dual of ``bind_parameters``: ``(name, ir.Type, direction)``
        instead of annotation facades — the entrance foundation's
        declaration-driven entry. Returns the matching facades in
        declaration order.
        """
        frame = self._prepare_frame()
        self._param_dim_symbols = self._param_extent_symbols([t for _, t, _ in params])
        return [frame.bind_typed_param(name, t, direction=direction) for name, t, direction in params]

    @property
    def frame(self) -> PtoFrame | None:
        """The frame of this one-shot trace, once begun (the trace driver
        installs it as the active Frame for the invocation)."""
        return self._frame

    # —— the root creation ——
    def begin_frame(self) -> PtoFrame:
        """The root creation — one-shot.

        Idempotent handout under the decorator-adapter entrance: when
        ``bind_parameters`` already opened the function context, the trace
        interpreter's first ``enter_frame`` receives that existing frame. Any later call
        is a rejected reuse.
        """
        if self._frame_started:
            raise RuntimeError(
                "PtoSema is one-shot: begin_frame() already served one function "
                "trace (reuse after success or failure is "
                "rejected, never reset)"
            )
        if self._frame is None:
            self._frame = PtoFrame(self)
        self._frame_started = True
        return self._frame

    def note_span(self, source_file: str, lineno: int, col: int) -> None:
        """Pin the IR span to the instrumented row's original source position.

        Called before each rewritten row executes (``rt._span(...)``). The
        pin wins over frame capture in the op and builder layers; the first
        pin saves the reset token, restored when the frame finishes. A trace
        that aborts on an exception leaves the last pin in place — spans are
        diagnostics-only and the process is on an error path.
        """
        span = ir.Span(source_file, lineno, col)
        if self._span_token is None:
            self._span_token = set_pinned_span(span)
        else:
            set_pinned_span(span)

    def _reset_span_pin(self) -> None:
        if self._span_token is not None:
            reset_pinned_span(self._span_token)
            self._span_token = None

    def abort(self) -> None:
        """Exception-path cleanup: drop this Sema's span pin.

        ``finish`` restores the pin on normal completion only; an aborted
        trace would otherwise leak the last row's pin into the process
        (the contextvar outlives the dead Sema), and a long-lived host
        that catches trace errors — pytest, the parity harnesses — then
        captures the stale span on every later host-API op call.
        """
        self._reset_span_pin()

    def result(self) -> ir.Function:
        """The built function (the frame's ``finish`` finalized it)."""
        if self._result is None:
            raise RuntimeError("PtoSema has no built function: the kernel never ran")
        return self._result

    # —— function-level facts the Frame delegates ——
    def auto_scope_enabled(self) -> bool:
        """Whether the active function owns automatic runtime-scope placement."""
        if self._parent is not None:
            return self._parent.auto_scope_enabled()
        return self._attrs is None or bool(self._attrs.get(AUTO_SCOPE_ATTR, True))

    def is_inside_device_scope(self) -> bool:
        """BAND-AID (``pld.world_size`` host-only guard): is the innermost
        PTO runtime scope a device ``InCore`` / ``Spmd`` region? An ambient
        scope-stack read via the funnel — this guards a **lowerability**
        invariant that belongs in the IR verifier; retire when the C++
        verifier owns the invariant.
        """
        return self._builder.is_inside_scope(ir.ScopeKind.InCore) or self._builder.is_inside_scope(
            ir.ScopeKind.Spmd
        )

    def finish_inline_body(self, body: str) -> None:
        """Emit a verbatim Python body and finish the prepared function frame.

        The inline entry is a function trace: it requires the prepared,
        not-yet-started frame. A trace-driver invocation goes through
        ``begin_frame`` instead. The one-shot state is locked before the
        first builder mutation, so reuse after an inline body, success or
        failure, is rejected like any other trace.
        """
        if self._frame is None:
            raise RuntimeError("PtoSema has no active frame for an inline body")
        if self._frame_started:
            raise RuntimeError(
                "PtoSema is one-shot: the function trace already began "
                "(begin_frame or a previous inline body) — reuse after success "
                "or failure is rejected, never reset"
            )
        self._frame_started = True
        self._builder.inline_stmt(body, ir.InlineLanguage.Python)
        self._frame.finish()


class PtoFrame:
    """One function tracing context implementing ``Frame[Facade]``.

    The value domain is the closed facade union
    (:data:`pypto.language.facade.Facade`). PtoFrame owns the mapping from
    source names to mutable `ir.Var`s, pending comments, pending
    `iter_args`, and stdlib trace-protocol mediation. It borrows the Sema's
    builder while active and has no active-Construct stack, loop stack,
    control-flow-depth counter, or result-binding stack; structural questions
    are queries to the trace interpreter.
    """

    def __init__(self, sema: PtoSema) -> None:
        self._sema = sema
        self._rt: Interpreter[Facade] | None = None
        self._locals: dict[str, ir.Var] = {}
        self._pending_comments: list[str] = []
        self._pending_iter_args: tuple[ir.IterArg, ...] | None = None
        # Declarable scopes whose ``begin_scope`` is deferred (see
        # ``_DeferredDeclarableScope``) — at scopes and spmd carriers, nesting
        # (registration) order. Driven by the builder's pre-emit hook
        # (``_flush_deferred_scopes``) and the block-index carrier trigger
        # (``_maybe_open_spmd_carriers``).
        self._deferred_scopes: list[_DeferredScopeEntry] = []
        self._finished = False
        self._temp_seq = 0
        self._ctx = TraceContext(self)
        if sema._parent is not None:
            # The borrowing expansion frame: no function, declaration, or call
            # — it emits into the parent's builder at the current insertion
            # point, and its ``finish`` ends nothing.
            return
        # The pre-emit hook opens deferred declarable scopes before a
        # statement lands (see ``_flush_deferred_scopes``); installed per
        # root Frame and routed to the *ambient* frame — a borrowing
        # expansion shares this builder while owning its own deferred list.
        # The parser frontend never installs one.
        sema._builder.set_pre_emit_hook(_ambient_pre_emit_flush)
        # The builder function context: opened here (so bind_param/bind can
        # emit), closed by ``finish``.
        sema._builder.begin_function(
            sema._name,
            span=sema._source_span,
            type=sema._function_type,
            level=sema._level,
            role=sema._role,
            attrs=sema._attrs,
            requires_runtime_binding=sema._requires_runtime_binding,
        )
        if sema._return_type is not None:
            sema._builder.return_type(sema._return_type)

    # —— the trace-interpreter association: ambient ops use it internally ——
    def note_interpreter(self, rt: Interpreter[Facade] | None) -> None:
        """Record the active trace interpreter at frame entry.

        A ``None`` interpreter, as used by direct-API probes without a
        collector, is ignored."""
        if rt is not None:
            self._rt = rt

    def active_runtime(self) -> Interpreter[Facade] | None:
        return self._rt

    def _require_rt(self) -> Interpreter[Facade]:
        if self._rt is None:
            raise RuntimeError(
                "the PtoFrame has no active trace interpreter: this operation needs a "
                "trace-interpreter-driven trace (the decorator adapter installs one)"
            )
        return self._rt

    @property
    def sema(self) -> PtoSema:
        """The one-shot Sema owning the builder and the trace result."""
        return self._sema

    @property
    def function_level(self) -> ir.Level | None:
        """The active function's dispatch level (HOST / AIC / AIV / ...),
        from its declaration — shared ops read it through ``active_frame()``
        to decide caller-context rules (e.g. HOST submits may not omit
        Out/InOut arguments)."""
        return self._sema._level

    def trace_context(self) -> TraceContext:
        """The tracing context passed to stdlib trace-protocol hooks."""
        return self._ctx

    # —— parameter binding (the decorator-adapter entrance) ——
    def bind_param(
        self,
        name: str,
        var: Any | None,
        span: object | None = None,
        direction: ir.ParamDirection = ir.ParamDirection.In,
    ) -> Any:
        """Bind a parameter variable from its annotation facade.

        Return the typed facade the decorator injects. A non-facade annotation
        (the ``self`` proxy or a host container) passes through unchanged; no
        IR parameter is created for it."""
        del span
        if var is None:
            return None
        if not is_staged_annotation(var):
            return var
        if direction == ir.ParamDirection.InOut and isinstance(annotation_to_type(var), ir.ScalarType):
            # The AST's resolve_param_type discipline: a Scalar is a
            # pass-by-value value — InOut makes no sense on it.
            raise ParserTypeError(
                f"kernel parameter {name!r}: Scalar parameters cannot have InOut "
                f"direction (only Tensor and Tile parameters support InOut)"
            )
        param_var = self._sema._builder.func_arg(
            name, annotation_to_type(var), span=self._sema._source_span, direction=direction
        )
        self._locals[name] = param_var
        # A subscript annotation is a facade instance (``. __class__`` is its
        # facade class); a parameterless kind's annotation is the bare class
        # itself (``pld.CommCtx``) — already the class to re-wrap as.
        facade_cls: type[Facade] = var if isinstance(var, type) else var.__class__
        return _rewrap_as_base_facade(facade_cls, param_var)

    def bind_typed_param(
        self,
        name: str,
        param_type: ir.Type,
        direction: ir.ParamDirection = ir.ParamDirection.In,
    ) -> _UnwrapsToExpr:
        """Bind a typed parameter (name, ``ir.Type``, direction) to its Var.

        The typed dual of ``bind_param``: the declaration-driven entrance
        already holds the ``ir.Type``, so no annotation facade is consulted;
        the facade kind is selected by the type itself.
        """
        if direction == ir.ParamDirection.InOut and isinstance(param_type, ir.ScalarType):
            # The AST's resolve_param_type discipline: a Scalar is a
            # pass-by-value value — InOut makes no sense on it.
            raise ParserTypeError(
                f"kernel parameter {name!r}: Scalar parameters cannot have InOut "
                f"direction (only Tensor and Tile parameters support InOut)"
            )
        param_var = self._sema._builder.func_arg(name, param_type, direction=direction)
        self._locals[name] = param_var
        return wrap_expr(param_var, param_type)

    # —— preconditions shared by every mutating Frame operation ——
    def _require_no_pending_result(self, operation: str) -> Interpreter[Facade]:
        """Reject ``operation`` while a pending result construct is on top.

        This centralized mutation guard is checked by every Frame method that
        emits through the builder, directly or through the TraceContext. The
        result-binding phase admits only bindings for that construct and
        comment attachment, which is annotation state rather than mutation.
        Queries stay unguarded.
        """
        rt = self._require_rt()
        pending = rt.pending_result()
        if pending is not None:
            raise TypeError(
                f"{operation}: result bindings for the staged {pending.kind} must complete before another operation is traced"
            )
        return rt

    def _reject_pending_carry(self, operation: str) -> None:
        """Reject ``operation`` while pending `iter_args` await a loop opener.

        The group must be consumed by the immediately following loop opener.
        Comments stay exempt because they are annotation state consumed by
        that opener.
        """
        if self._pending_iter_args is not None:
            raise TypeError(
                f"pl.intrin.carry: pending iter_args must be claimed by the "
                f"immediately following loop opener — {operation} intervenes"
            )

    # —— comments (Frame-owned until consumed) ——
    def add_leading_comments(self, comments: Sequence[str]) -> None:
        """Queue source comments for the next logical IR statement."""
        if not comments:
            return
        self._pending_comments.extend(comments)

    def _take_pending_comments(self) -> list[str]:
        comments = self._pending_comments
        self._pending_comments = []
        return comments

    def _restore_pending_comments(self, comments: list[str]) -> None:
        if not comments:
            return
        self._pending_comments = [*comments, *self._pending_comments]

    @contextmanager
    def _statement_comments(self) -> Iterator[None]:
        """Temporarily expose one statement's comments to the builder."""
        comments = self._take_pending_comments()
        if not comments:
            yield
            return
        self._sema._builder.push_pending_leading_comments(comments)
        try:
            yield
        finally:
            self._restore_pending_comments(self._sema._builder.pop_pending_leading_comments())

    def _begin_construct_comments(self) -> bool:
        """Leave one comment entry pending until the construct closes."""
        comments = self._take_pending_comments()
        if not comments:
            return False
        self._sema._builder.push_pending_leading_comments(comments)
        return True

    def _finish_construct_comments(self, pending: bool) -> None:
        if pending:
            self._restore_pending_comments(self._sema._builder.pop_pending_leading_comments())

    # —— the recognition query ——
    def is_staged(self, value: object) -> bool:
        """Whether a facade or trace-protocol object participates in staging.

        Recognition is value-directed, never based on context; the
        bool-coercing resolvers use this query."""
        return is_staged(value)

    # —— binding and trace-protocol dispatch ——
    def bind(
        self, rt: Interpreter[Facade] | None, name: str, value: object, span: object | None = None
    ) -> BindingResult[Facade]:
        self._reject_pending_carry(f"an ordinary bind of {name!r}")
        if isinstance(value, CachePolicyDecl):
            # The declaration is statement-position only; binding a value from
            # it would silently drop the declaration (it is not a value).
            raise TypeError(
                f"pl.set_cache_policy(...) is a standalone declaration — {name!r} cannot bind its result"
            )
        with self._statement_comments():
            # The collector protocol is span-free; the instrumented rows pin
            # their span (rt._span -> set_pinned_span) and this boundary reads
            # the pin — no stack walk during a trace.
            if span is None:
                span = current_pinned_span()
            return self._bind_impl(rt, name, value, span)

    def _bind_impl(
        self, rt: Interpreter[Facade] | None, name: str, value: object, span: object | None
    ) -> BindingResult[Facade]:
        """The binding row and the typed staged-initialization guards.

        Both directions of a staged-ness flip are loud under staged control
        flow. A host prior cannot feed a staged rebind (the join or loop
        back-edge would inline the literal), and a staged local cannot be
        demoted to a host literal (reads after the join or back-edge would
        inline the literal instead of re-reading the mutable `ir.Var`). The
        interpreter supplies the previous host-binding fact."""
        # A host container holding an allreduce result is the one bind the
        # host tuple path cannot lower: the parser emits a tuple construction
        # the synthesize-allreduce-signals pass rejects, while the traced path
        # would drop the call entirely whenever the container never reaches a
        # return. Reject here, loudly, with the pass's wording (a direct
        # assignment / expression statement / return value are the legal
        # positions) so the silent-drop path stays closed.
        if not is_staged(value) and _contains_allreduce(value):
            raise ValueError(
                "pld.tensor.allreduce must be a direct assignment, expression statement, "
                "or return value before allreduce signal synthesis."
            )
        # None is the parser's TaskId sentinel. Reify it before the normal
        # staged binding so loop-carried dependency state gets a real Var.
        if value is None:
            value = Scalar(expr=ir.create_op_call("system.task_invalid", [], {}, ir.Span.unknown()))
        pending = rt.pending_result() if rt is not None else None
        if pending is not None:
            if not isinstance(value, _ResultBindingProxy):
                raise TypeError(
                    f"pl.intrin.leak: result bindings for the staged {pending.kind} "
                    f"must complete before binding {name!r}"
                )
        result = self._bind_dispatch(rt, name, value, span)
        if isinstance(result, StagedBinding) and not isinstance(value, _ResultBindingProxy):
            if rt is not None and rt.staged_control_flow_active() and rt.current_binding(name)[0]:
                raise TypeError(
                    f"typed staged initialization: {name!r} holds a host literal, "
                    f"but staged control flow needs a typed staged value here "
                    "— bind a typed constant first: "
                    f"{name} = pl.const(..., dtype)"
                )
        if isinstance(result, HostBinding) and rt is not None and rt.staged_control_flow_active():
            was_host, prior = rt.current_binding(name)
            if prior is not None and not was_host and self.is_staged(prior):
                raise TypeError(
                    f"typed staged initialization: {name!r} holds a staged value, "
                    f"but staged control flow cannot demote it to a host literal — "
                    "reads after the join or loop back-edge would inline the "
                    "literal. Use a fresh name for the host value, or bind a "
                    f"typed constant: {name} = pl.const(..., dtype)"
                )
        return result

    def _bind_dispatch(
        self, rt: Interpreter[Facade] | None, name: str, value: object, span: object | None
    ) -> BindingResult[Facade]:
        """Dispatch a binding through a value's ``__trace_bind__`` hook.

        A hook may return an ``ir.Var`` registered and wrapped here, an
        :class:`_ExistingVarBinding` registered without ``LetStmt`` or
        ``AssignStmt``, a facade for redispatch, or ``None`` to use
        ``_default_bind``."""
        if isinstance(value, SupportsTraceBind):
            # Open deferred declarable scopes before the hook runs — a binding
            # proxy may read structural state (the for-target activation's
            # ``is_inside_incore``) that the pending scopes would change.
            self._flush_deferred_scopes()
            pending = rt.pending_result() if rt is not None else None
            result = value.__trace_bind__(self._ctx, name)
            if isinstance(result, ir.Var):
                self._locals[name] = result
                facade_cls = self._facade_class(value.__class__)
                if facade_cls is None:
                    # A raw-Var binding proxy outside the facade vocabulary (e.g. the
                    # pre-created task id): infer the kind from the Var's type.
                    facade_cls = Tile if isinstance(result.type, ir.TileType) else Scalar
                return self._structural_result(pending, facade_cls, result)
            if isinstance(result, _ExistingVarBinding):
                var = result.var
                facade_cls = self._facade_class(value.__class__)
                if facade_cls is None:
                    # A binding proxy (iter_args or operation-result binding): infer the facade kind
                    # from the Var's type.
                    if isinstance(var.type, ir.TileType):
                        facade_cls = Tile
                    elif isinstance(var.type, ir.TensorType):
                        facade_cls = Tensor
                    else:
                        facade_cls = Scalar
                self._locals[name] = var
                return self._structural_result(pending, facade_cls, var)
            if result is None:
                return self._default_bind(name, value, span)
            if not is_staged(result):
                raise TypeError(
                    f"{type(value).__name__}.__trace_bind__ returned an "
                    f"unsupported result {result!r} — a bind hook returns an "
                    "ir.Var, a facade, or None (decline); anything else "
                    "would silently downgrade the name to a host value"
                )
            return self._default_bind(name, result, span)
        return self._default_bind(name, value, span)

    @staticmethod
    def _facade_class(owner: type[object]) -> type[Facade] | None:
        """The base facade of ``owner``, or ``None`` outside the vocabulary.

        A ``__trace_bind__`` implementer that is itself a facade subclass
        re-wraps in its base; a binding proxy returns ``None`` so
        the variable's type selects the facade kind."""
        return _base_facade_class(owner)

    def _maybe_open_spmd_carriers(self, expr: ir.Expr) -> None:
        """Open a deferred spmd InCore carrier when ``expr`` reads the block index.

        The plain ``with pl.spmd(...)`` form defers its carrier decision to the
        body — the parser classifies the same fact statically
        (``_spmd_body_reads_block_idx``): a body that reads the per-block index
        does its own per-block work, so its statements must land inside an
        ``InCoreScopeStmt`` for the outliner; a dispatch body stays bare. The
        trigger fires only at the with-body's top level — a read nested in
        staged control flow opened *inside* the body belongs to that region, a
        read inside an explicit ``pl.at`` carrier (the printer's own wrapped
        form) already has its InCore, and a read while an inner deferred scope
        is pending belongs to that scope's body (the innermost pending entry
        must be the carrier itself). A staged construct *outside* the
        with-form (the ``pl.parallel`` loop a spmd scope is nested in, the
        per-expert-tile pattern) does not block the open — the read still
        sits at the with-body's own top level. Opening the carrier flushes
        any outer deferred scopes first — ``begin_scope`` runs the pre-emit
        hook — so the nesting lands outer-scope-first either way."""
        if not self._deferred_scopes:
            return
        if self._rt is not None and self._rt.staged_control_flow_active():
            # The blunt fact alone cannot locate the read: decline only when
            # a staged if/for/while sits between the emission point and the
            # carrier's own construct (the read belongs to that nested
            # region); a staged construct below the with-form does not own
            # the with-body's statements.
            entry = self._deferred_scopes[-1]
            if self._rt.staged_control_flow_above(entry.construct):
                return
        if self._sema._builder.is_inside_scope(ir.ScopeKind.InCore):
            return
        if not _expr_reads_block_index(expr):
            return
        entry = self._deferred_scopes[-1]
        if entry.flush_on_emit or not isinstance(entry.construct, _RuntimeDeferredSpmdCarrier):
            return
        self._deferred_scopes.pop()
        entry.construct.open_pending_carrier(self._sema._builder)

    def _retire_deferred_scope(self, ctxobj: object) -> None:
        """Drop ``ctxobj`` from the deferred-scope list at its construct's exit."""
        self._deferred_scopes = [e for e in self._deferred_scopes if e.construct is not ctxobj]

    # —— the deferred declarable scopes ——
    def register_deferred_scope(self, construct: object, *, flush_on_emit: bool = True) -> None:
        """Register a scope that deferred its ``begin_scope`` (see
        ``_DeferredDeclarableScope``). Nesting order is registration order."""
        assert isinstance(construct, _RuntimeDeferredDeclarableScope)  # the caller checked
        self._deferred_scopes.append(_DeferredScopeEntry(construct, flush_on_emit))

    def _flush_deferred_scopes(self, entering_incore: bool = False) -> None:
        """Open every flushable pending scope, outermost first.

        The builder's pre-emit hook: the first IR-emitting statement inside a
        deferred scope's body opens the scope (with every buffered
        ``cache_policy_vars`` declaration folded into its attrs) before the
        statement lands. Non-flushable entries (a plain spmd carrier) are
        skipped — their open decision belongs to the block-index read — but
        never shadowed: an inner flushable entry still opens, nested directly
        in whatever region is current. ``entering_incore`` says the construct
        about to be entered (or a later pending entry) opens its own InCore
        region — a wrapped spmd carrier is absorbed rather than opened, so
        the body's own region nests directly in the Spmd scope."""
        index = 0
        while index < len(self._deferred_scopes):
            entry = self._deferred_scopes[index]
            if not entry.flush_on_emit:
                index += 1
                continue
            if isinstance(entry.construct, _RuntimeDeferredSpmdCarrier) and (
                entering_incore
                or any(
                    isinstance(other.construct, _RuntimeDeferredDeclarableScope)
                    and getattr(other.construct, "_pending_begin", (None,))[0] is ir.ScopeKind.InCore
                    for other in self._deferred_scopes[index + 1 :]
                )
            ):
                # A wrapped spmd whose body opens its own InCore region (a
                # bare nested ``pl.at``): the carrier is absorbed — the
                # region nests directly in the Spmd scope, and opening the
                # carrier would produce the nested-InCore shape the verifier
                # rejects.
                del self._deferred_scopes[index]
                continue
            del self._deferred_scopes[index]
            entry.construct.open_deferred_scope(self._sema._builder)

    def _attach_cache_policy(self, decl: CachePolicyDecl) -> None:
        """Route one ``pl.set_cache_policy`` declaration to its owning scope.

        The declaration must reach the innermost *pending* declarable scope
        before that scope's first statement opens it — attrs are fixed at
        ``begin_scope`` — so the routing is: pending scope (with the
        control-flow position check against its owning construct), else the
        adjacency rejection (the innermost open scope is InCore/Hierarchy but
        its attrs are already fixed), else the kind rejection (Spmd/Cluster/
        SplitAiv have nothing to attach to), else the position rejection."""
        rt = self._rt
        if self._deferred_scopes:
            target = self._deferred_scopes[-1].construct
            if rt is not None and rt.staged_control_flow_above(target):
                raise ParserSyntaxError(
                    "pl.set_cache_policy() must be a standalone statement directly inside "
                    "a pl.at(...) / pl.spmd(...) scope body",
                    hint=_CACHE_POLICY_POSITION_HINT,
                )
            target.attach_cache_policy(decl.var, decl.policy)
            return
        if rt is not None and rt.staged_control_flow_active():
            raise ParserSyntaxError(
                "pl.set_cache_policy() must be a standalone statement directly inside "
                "a pl.at(...) / pl.spmd(...) scope body",
                hint=_CACHE_POLICY_POSITION_HINT,
            )
        kind = self._sema._builder.innermost_scope_kind()
        if kind in (ir.ScopeKind.InCore, ir.ScopeKind.Hierarchy):
            # The scope is already open: its attrs were fixed at begin. The
            # parser hoists position-independently, so this is the tracing-
            # side adjacency restriction — louder than the parser, and
            # satisfied by every printed kernel (declarations print first).
            raise ParserSyntaxError(
                "pl.set_cache_policy() must precede the scope's first statement — the "
                "declaration joins the scope's attrs, which are fixed when the scope begins",
                hint=_CACHE_POLICY_ADJACENCY_HINT,
            )
        if kind is not None:
            raise ParserSyntaxError(
                f"pl.set_cache_policy() has nothing to attach to on this {kind.name} scope",
                hint=(
                    "This `pl.spmd(...)` body dispatches a pre-defined kernel, so the GM reads "
                    "happen inside that callee, not here. Declare the policy in the callee — "
                    "a `with pl.at(level=pl.Level.CORE_GROUP):` scope in its body, or "
                    "pl.load(..., cache=pl.CachePolicy.BYPASS) on the access itself. An "
                    "*inline* pl.spmd body does accept the marker: it is carried by the "
                    "InCore scope the body is outlined into."
                    if kind is ir.ScopeKind.Spmd
                    else _CACHE_POLICY_POSITION_HINT
                ),
            )
        raise ParserSyntaxError(
            "pl.set_cache_policy() must be a standalone statement directly inside "
            "a pl.at(...) / pl.spmd(...) scope body",
            hint=_CACHE_POLICY_POSITION_HINT,
        )

    def _structural_result(
        self, pending: Construct[Facade] | None, facade_cls: type[Facade], var: ir.Var
    ) -> BindingResult[Facade]:
        """Wrap a binding proxy's variable and report result-binding progress
        or exact completion when a pending result construct is present."""
        facade = _rewrap_as_base_facade(facade_cls, var)
        if pending is None:
            return StagedBinding(facade)
        if not isinstance(pending, ResultBindingTarget):
            raise TypeError("result bindings are unavailable for this staged Construct")
        if pending.completed:
            return FinalResultBinding(pending, facade)
        return PartialResultBinding(pending, facade)

    def _default_bind(self, name: str, value: object, span: object | None = None) -> BindingResult[Facade]:
        """The ordinary staged binding path after trace-protocol hooks decline.

        Host values decline. Staged facades create or reuse one mutable
        `ir.Var` per name (``let`` on first definition and ``assign`` on
        rebinding) and return a fresh facade wrapping that variable; the input
        ``value`` is never mutated in place. ``span`` (captured at the Sema
        boundary) names the user's statement."""
        stmt_span = span if isinstance(span, ir.Span) else None
        if not is_staged(value):
            return HostBinding()
        expr = _unwrap_facade(value)  # a facade (is_staged above)
        self._maybe_open_spmd_carriers(expr)
        local = self._locals.get(name)
        if local is None:
            local = self._sema._builder.let(name, expr, span=stmt_span)
            self._locals[name] = local
        elif self._sema._strict_ssa:
            # round-trip dialect opts out of mutable-local assignment — a
            # rebind of an already-bound name is a loud error (the lint).
            raise SSAViolationError(
                f"variable '{name}' is already bound; strict_ssa forbids rebind "
                f"(round-trip dialect, text-parser)",
                hint="strict_ssa requires one bind per name; use "
                "@pl.function_parser (round-trip dialect) for carried "
                "values, or a distinct name",
            )
        elif isinstance(local, ir.IterArg):
            # IterArgs are loop inputs, not legal AssignStmt targets. A
            # rebind inside the loop starts a regular body-local SSA variable.
            local = self._sema._builder.let(name, expr, span=stmt_span)
            self._locals[name] = local
        elif id(local) in self._sema._param_dim_symbols:
            # The name is bound to an immutable extent symbol; a staged
            # rebind must not assign *through* the alias into the symbol.
            local = self._sema._builder.let(name, expr, span=stmt_span)
            self._locals[name] = local
        elif self._rebind_changes_type(local, expr):
            # The relaxed reassignment rule (parser parity): a type-changing
            # rebind starts a fresh let instead of assigning through the old
            # Var — assign-through would pin every later trace-time type
            # query on the name to the first binding's type (the
            # silently-wrong state #642 rejected).
            local = self._sema._builder.let(name, expr)
            self._locals[name] = local
        else:
            self._sema._builder.assign(local, expr, span=stmt_span)
            # A rebind drops the forward-sticky dump tag; the tag applies
            # only to the original staged local definition.
            self._sema.clear_dump_tag(local)
        # A facade here (is_staged and _unwrap_facade above); the
        # base-class lookup is loud outside the closed vocabulary.
        base = _base_facade_class(value.__class__)
        if base is None:
            raise TypeError(
                f"cannot rewrap {type(value).__name__} — it unwraps to an expr "
                "but has no ancestor in the closed facade vocabulary"
            )
        return StagedBinding(base(expr=local))

    def _rebind_changes_type(self, local: ir.Var, expr: ir.Expr) -> bool:
        """Whether a rebind's value type differs from the name's current Var.

        One Python name maps to one mutable Var under assign-through
        rebinding, so a type-changing rebind must NOT assign through: the
        traced facade would keep the *first* binding's type and every
        trace-time type query on the name would lie (an ``[8,1]``-typed Var
        fed a ``[1,8]`` reshape once broadcast the corpus's softplus to
        ``[8,8]``). Both frontends share the comparison (the typing seam's
        ``_types_match``) so they accept and reject identically; the relaxed
        rule then lets a fresh Var carry the new type. An ``UnknownType`` on
        either side never counts as a change (an unresolved type proves
        nothing — the parser's exemption).
        """
        from pypto.language.typing.type_utils import (  # noqa: PLC0415 — the shared seam (both frontends match identically)
            _tile_types_differ_only_by_unset_memory_space,
            _types_match,
        )

        existing, new = local.type, expr.type
        if isinstance(existing, ir.UnknownType) or isinstance(new, ir.UnknownType):
            return False
        if _types_match(existing, new):
            return False
        # An unset memory_space is a wildcard, not a distinct type: a
        # load-created (unset) tile reassigned from an op-produced (Vec)
        # tile keeps one Var — the loop-carry/SSA derivation relies on
        # assign-through for exactly this shape.
        return not _tile_types_differ_only_by_unset_memory_space(existing, new)

    @property
    def borrowed_parent(self) -> PtoFrame | None:
        """The parent frame whose locals an expansion body may write through.

        An expansion's parameters are substitution: an item-assignment
        targeting one is the caller's own write, so the frame that owns the
        name (and whose interpreter must rebind it) is the parent's, not the
        expansion's fresh lexical frame. ``None`` on a root frame.
        """
        parent_sema = self._sema._parent
        return parent_sema.frame if parent_sema is not None else None

    def resolve_staged_local(self, expr: object) -> tuple[PtoFrame | None, str]:
        """The frame owning ``expr`` as a staged local, and the local's name.

        A ``None`` owner means ``expr`` is no frame's staged local — the loud
        item-assignment error. Under a borrowing expansion a substituted
        caller value resolves to the parent frame (write-through semantics:
        the body behaves as text inlined at the call site), while the
        expansion's own locals shadow first.
        """
        for name, local in self._locals.items():
            if local is expr:
                return self, name
        parent = self.borrowed_parent
        if parent is not None:
            owner, name = parent.resolve_staged_local(expr)
            if owner is not None:
                return owner, name
        return None, ""

    def resolve_local_name(self, expr: object) -> str | None:
        """The name whose mutable `ir.Var` is ``expr``, or ``None`` when
        ``expr`` is not a staged local. The owning frame may be a borrowing
        expansion's parent (see :meth:`resolve_staged_local`)."""
        owner, name = self.resolve_staged_local(expr)
        return name if owner is not None else None

    # —— expression statements ——
    def expression_statement(
        self, rt: Interpreter[Facade] | None, value: object, span: object | None = None
    ) -> None:
        """Handle an expression statement.

        A staged call produces an ``EvalStmt`` after the operation has run; a
        host value is ignored because it produces no IR."""
        self._reject_pending_carry("an expression statement")
        with self._statement_comments():
            if rt is not None:
                pending = rt.pending_result()
                if pending is not None and not isinstance(value, SupportsTraceStmt):
                    # A yield marker stays exempt here so it reaches its own
                    # precise rejection in ``collect_yield``; everything else
                    # is an ordinary operation against the pending result construct.
                    raise TypeError(
                        f"expression statement while the staged {pending.kind} awaits "
                        "result bindings — only result bindings may address it"
                    )
            stmt_span = span if isinstance(span, ir.Span) else current_pinned_span()
            if isinstance(value, SupportsTraceStmt):
                value.__trace_stmt__(self._ctx)
                return
            if isinstance(value, CachePolicyDecl):
                # The scope-level cache-policy declaration: routed to the
                # enclosing deferred scope, no IR of its own (the parser's
                # ``_handle_set_cache_policy`` consumes it the same way).
                self._attach_cache_policy(value)
                return
            if isinstance(value, DumpTagMarker):
                var = value.unwrap()
                # The parser path rejects non-Var bindings loudly (its
                # ``lookup_var`` placeholder check); match it — a non-Var
                # would otherwise never match the read-side Var filter.
                if not isinstance(var, ir.Var):
                    raise TypeError(
                        f"pl.dump_tag() argument is not a bound tensor Var (got "
                        f"{type(var).__name__}) — tag a bound variable"
                    )
                self._sema.record_dump_tag(var)
                return
            if is_staged(value):
                expr = _unwrap_facade(value)
                self._maybe_open_spmd_carriers(expr)
                # A bare non-Call staged expression routes to ``EvalStmt`` —
                # matching the parser frontend.
                self._sema._builder.eval_stmt(expr, span=stmt_span)
            elif isinstance(value, ir.Call):
                # Compatibility with low-level helpers exposing a raw call
                # for a bare side effect.
                self._sema._builder.eval_stmt(value, span=stmt_span)

    # —— returns ——
    def return_(
        self, rt: Interpreter[Facade] | None, values: Sequence[object] | None, span: object | None = None
    ) -> None:
        """Record a staged return in the current branch.

        The per-branch ``ReturnStmt`` values converge through the SSA pass. A
        host return is ignored because the source rewriter's
        ``in_host_mode`` guard suppresses the native ``return`` while a staged
        Construct is open."""
        self._reject_pending_carry("a return")
        if rt is not None and rt.pending_result() is not None:
            raise TypeError(
                "return is not allowed while a result-producing construct awaits its result bindings"
            )
        if self._sema._parent is not None:
            # The borrowing expansion: the return is expansion-local — the
            # native return (the in-host-mode guard) carries the value to the
            # parent, and no ReturnStmt is emitted into the borrowed builder.
            # An early return under staged control flow would need region
            # merge semantics the expansion does not own; reject it loudly.
            if rt is not None and rt.staged_control_flow_active():
                raise TypeError(
                    "an expansion body must end in one trailing return — an early "
                    "return inside staged control flow has no expansion semantics"
                )
            return
        with self._statement_comments():
            if span is None:
                span = current_pinned_span()
            self._return_impl(values, span)

    def _return_impl(self, values: Sequence[object] | None, span: object | None) -> None:
        stmt_span = span if isinstance(span, ir.Span) else None
        if values is None:
            self._sema._builder.return_stmt(span=stmt_span)
            return
        # The source rewriter's ``_return_row`` always does ``rt.return_(__pto_ret0)``
        # with a single temp. For ``return [y, z]`` or ``return y, z`` that temp
        # is a host list/tuple containing facades, so ``values`` is ``[[y,z]]``
        # or ``[(y,z)]``. Expand a single host container of facades into the
        # actual multi-value return. A ``pl.Tuple(y,z)`` facade is *not* a host
        # list/tuple, so it stays as a single staged value. For a declared
        # ``-> pl.Tuple[T1, T2]`` the printer emits ``return [y, z]`` as a
        # single ``MakeTuple``; keep it as a single Tuple return.
        if len(values) == 1 and _is_sequence(values[0]):
            inner = values[0]
            if any(is_staged(v) for v in inner):
                if isinstance(self._sema._return_type, ir.TupleType) and len(inner) == len(
                    self._sema._return_type.types
                ):
                    from pypto.language.typing import Tuple as PlTuple  # noqa: PLC0415

                    tup = PlTuple(*inner)
                    self._sema._builder.return_stmt(tup.unwrap(), span=stmt_span)
                    return
                values = list(inner)
        staged = [_unwrap_facade(v) for v in values if is_staged(v)]
        if not staged:
            return
        if len(staged) == 1:
            self._sema._builder.return_stmt(staged[0], span=stmt_span)
        else:
            self._sema._builder.return_stmt(staged, span=stmt_span)

    # —— staged Construct construction ——
    def create_if_construct(
        self, rt: Interpreter[Facade] | None, predicate: object, span: object | None = None
    ) -> IfConstruct[Facade] | None:
        """Open ``builder.begin_if`` and return the exact staged PtoIfConstruct.

        The Construct owns builder state and explicit-result local mappings. A
        host predicate declines (``None`` and the interpreter installs its
        host IfConstruct); a staged predicate must be BOOL-typed."""
        del span
        self._reject_pending_carry("an if")
        # A deferred declarable scope opens before the if's region begins —
        # the parser nests the IfStmt inside it.
        self._flush_deferred_scopes()
        if not is_staged(predicate):
            return None
        self._require_bool(predicate, "if")
        if rt is not None and rt.pending_result() is not None:
            raise TypeError("cannot open if while a result-producing construct awaits leak()")
        comments_pending = self._begin_construct_comments()
        try:
            from pypto.language.tracer.constructs import PtoIfConstruct  # noqa: PLC0415

            return PtoIfConstruct(
                self,
                _predicate_expr(predicate),
                dict(self._locals),
                rt.host_bound_names() if rt is not None else frozenset(),
                comments_pending,
            )
        except Exception:
            self._finish_construct_comments(comments_pending)
            raise

    def create_for_construct(
        self, rt: Interpreter[Facade] | None, iterable: object, span: object | None = None
    ) -> ForOpen[Facade] | None:
        """Open a staged ``for`` from a trace-iterable protocol object.

        The object opens itself through the tracing context and returns
        ``ForOpen(construct, binding)``; the Construct owns deferred target
        activation, while the binding proxy retains that exact Construct. A
        host iterable declines; a staged value without the trace-iterable
        protocol is rejected loudly."""
        del span
        # A deferred declarable scope opens before the for's descriptor runs
        # — both for its own region construction and for the structural reads
        # a ``begin`` may make (``pl.split_aiv`` deciding in-place vs.
        # wrapped from ``is_inside_incore``).
        self._flush_deferred_scopes()
        if not isinstance(iterable, SupportsTraceIter):
            if is_staged(iterable):
                raise TypeError(
                    f"for {iterable!r}: a staged value is not iterable — "
                    f"a staged for needs a trace iterable (an object "
                    f"implementing __trace_iter__)"
                )
            # A host for cannot claim pending iter_args (a host iterable has
            # no iter-arg channel) — the adjacency invariant rejects.
            self._reject_pending_carry("a host for (a host iterable)")
            return None
        if not isinstance(iterable, SupportsTraceExit):
            raise TypeError(
                f"for {type(iterable).__name__}: a trace iterable implements "
                "__trace_iter__ but not __trace_exit__ — the close hook is "
                "required to finalize the region it opens"
            )
        if rt is not None and rt.pending_result() is not None:
            raise TypeError("cannot open for while a result-producing construct awaits leak()")
        # Pending iter_args must be consumed by a loop opener immediately
        # following the carry bindings.
        pending = self._pending_iter_args
        iter_args: list[ir.IterArg] | None = None
        if pending is not None:
            kind = iterable.kind if isinstance(iterable, _LoopDescriptor) else None
            if kind == "unroll":
                raise ParserSyntaxError("pl.unroll() cannot be combined with init_values")
            if kind not in ("range", "parallel", "pipeline"):
                raise TypeError(
                    f"pl.intrin.carry follower must be pl.range/parallel/pipeline, got {type(iterable).__name__}"
                )
            self._pending_iter_args = None
            iter_args = list(pending)
        comments_pending = self._begin_construct_comments()
        try:
            from pypto.language.tracer.constructs import PtoForConstruct  # noqa: PLC0415

            binding = iterable.__trace_iter__(self._ctx)
            if not isinstance(binding, SupportsTraceBind):
                raise TypeError(
                    f"{type(iterable).__name__}.__trace_iter__ returned a binding proxy "
                    f"({type(binding).__name__}) without __trace_bind__ — the "
                    "loop target activates the region through the proxy's bind "
                    "protocol; a bare proxy would install a staged Construct "
                    "whose region never opens"
                )
            if not isinstance(iterable, _ForDescriptor):
                raise TypeError(
                    f"for {type(iterable).__name__}: a trace iterable must implement "
                    "begin() (the deferred for_loop construction) — the target "
                    "binding would otherwise fail at activation"
                )
            construct = PtoForConstruct(self, iterable, iter_args, comments_pending)
            construct.attach_binding(binding)
            if isinstance(binding, _ConstructAttachingBinding):
                # The binding proxy retains this exact Construct and asks it to activate
                # the bound target (deferred region construction).
                binding.attach_construct(construct)
            return ForOpen[Facade](construct, binding)
        except Exception:
            self._finish_construct_comments(comments_pending)
            raise

    def create_while_construct(
        self,
        rt: Interpreter[Facade] | None,
        predicate_thunk: Callable[[], object],
        span: object | None = None,
    ) -> WhileConstruct[Facade] | None:
        """Return a staging-capable PtoWhileConstruct at lexical entry.

        The Construct starts in host execution; the trace interpreter's
        ``test`` algorithm asks its activation hook for each predicate.
        Pending `iter_args` are claimed by the new Construct immediately, so
        an explicit carried while whose first predicate stays a host value is
        rejected at activation rather than silently consuming the group.
        """
        del rt, predicate_thunk, span
        from pypto.language.tracer.constructs import PtoWhileConstruct  # noqa: PLC0415

        construct = PtoWhileConstruct(self)
        if self._pending_iter_args is not None:
            construct.set_iter_args(self._pending_iter_args)
            self._pending_iter_args = None
        return construct

    def create_with_construct(
        self,
        rt: Interpreter[Facade] | None,
        context: object,
        name: str | None = None,
        span: object | None = None,
        single_call_body: bool = False,
    ) -> WithOpen[Facade] | None:
        """Open a staged ``with`` from a trace context manager.

        The object opens itself through the tracing context and returns
        ``WithOpen(construct, binding)``; the Construct owns the object's active
        lifecycle. ``name`` is the ``with ... as Y`` target. A host context
        manager declines (``None`` and the interpreter wraps it).
        """
        del rt, span
        self._reject_pending_carry("a with")
        # A deferred declarable scope must open before the new construct's
        # enter hook reads structural state (e.g. ``pl.split_aiv`` deciding
        # in-place vs. wrapped from ``is_inside_incore``) or begins its own
        # region — the parser's nesting has the new construct inside it.
        # A construct that will open its own InCore region (a bare nested
        # ``pl.at`` under a wrapped ``pl.spmd``) absorbs the carrier instead:
        # the region nests directly in the Spmd scope, never inside the
        # carrier's InCore (the nested-InCore shape the verifier rejects).
        self._flush_deferred_scopes(entering_incore=_enters_incore_scope(context))
        if not isinstance(context, SupportsTraceEnter):
            return None
        if not isinstance(context, SupportsTraceExit):
            raise TypeError(
                f"with {type(context).__name__}: a trace context manager implements "
                "__trace_enter__ but not __trace_exit__ — the close hook is "
                "required to finalize the region it opens"
            )
        comments = self._take_pending_comments()
        comments_pending = False
        if comments and name is None:
            self._sema._builder.push_pending_leading_comments(comments)
            comments_pending = True
        try:
            binding = context.__trace_enter__(self._ctx, name, single_call_body)
            if name is not None and not isinstance(binding, SupportsTraceBind):
                raise TypeError(
                    f"{type(context).__name__}.__trace_enter__ returned an "
                    f"as-target ({type(binding).__name__}) without __trace_bind__ — "
                    "a 'with ... as' target binds through the binding proxy's bind protocol"
                )
        except Exception:
            if comments_pending:
                self._finish_construct_comments(True)
            else:
                self._restore_pending_comments(comments)
            raise
        from pypto.language.tracer.constructs import PtoWithConstruct  # noqa: PLC0415

        deferred = comments if (name is not None and comments) else None
        construct = PtoWithConstruct(
            self, context, binding, deferred_comments=deferred, comments_pending=comments_pending
        )
        # A with-descriptor entered with its scope deferred — at scopes and
        # spmd carriers alike. The Frame owns the open (the pre-emit flush or
        # the block-index trigger) and the retire (``PtoWithConstruct.close``);
        # the pending flags are plain attributes (getattr member-checks, the
        # sanctioned open-world pattern), and only a plain spmd carrier opts
        # out of the first-emission flush.
        if isinstance(context, _RuntimeDeferredDeclarableScope) and (
            getattr(context, "_carrier_pending", False) or getattr(context, "_scope_pending", False)
        ):
            self.register_deferred_scope(
                context, flush_on_emit=getattr(context, "_carrier_flush_on_emit", True)
            )
        return WithOpen[Facade](construct, binding)

    # —— the value algebra ——
    def logical_and(self, a: object, b: object, span: object | None = None) -> Facade:
        """Build first-class ``ir.And``.

        Operands are normalized at the operand boundary: a host operand lifts
        to a typed ``Const``; a staged Scalar operand must be BOOL-typed."""
        del span
        self._reject_pending_carry("a logical operation")
        return Scalar(
            expr=ir.And(
                self._logical_operand(a, "and", "left"),
                self._logical_operand(b, "and", "right"),
                DataType.BOOL,
                ir.Span.unknown(),
            )
        )

    def logical_or(self, a: object, b: object, span: object | None = None) -> Facade:
        """As ``logical_and``, building ``ir.Or``."""
        del span
        self._reject_pending_carry("a logical operation")
        return Scalar(
            expr=ir.Or(
                self._logical_operand(a, "or", "left"),
                self._logical_operand(b, "or", "right"),
                DataType.BOOL,
                ir.Span.unknown(),
            )
        )

    def logical_not(self, a: object, span: object | None = None) -> Facade:
        """As ``logical_and``, building ``ir.Not`` (a single operand)."""
        del span
        self._reject_pending_carry("a logical operation")
        return Scalar(
            expr=ir.Not(
                self._logical_operand(a, "not", "the"),
                DataType.BOOL,
                ir.Span.unknown(),
            )
        )

    def _logical_operand(self, v: object, op: str, side: str) -> ir.Expr:
        """Normalize a logical-op operand (operand boundary)."""
        if not is_staged(v):
            return coerce_host_scalar(v, op, side)
        expr = _unwrap_facade(v)
        if not (isinstance(expr.type, ir.ScalarType) and expr.type.dtype == DataType.BOOL):
            raise TypeError(
                f"{op}: {side} operand must be BOOL-typed, got "
                f"{expr.type} — a non-bool "
                "Scalar coerced to bool is a verifier error; write the "
                "explicit test instead, e.g. x != 0"
            )
        return expr

    def _require_bool(self, pred: object, op: str) -> None:
        """Validate that a staged predicate is BOOL-typed."""
        expr = _unwrap_facade(pred)
        if not (isinstance(expr.type, ir.ScalarType) and expr.type.dtype == DataType.BOOL):
            raise TypeError(f"a staged {op} predicate must be BOOL-typed, got {expr.type}")

    # —— named stdlib operations ——
    def let_temp(self, node: ir.Expr, hint: str) -> ir.Var:
        """Emit a named temporary for a structural lowering helper."""
        self._reject_pending_carry(f"a temporary emission ({hint})")
        self._require_no_pending_result(f"a temporary emission ({hint})")
        self._temp_seq += 1
        return self._sema._builder.let(f"_{hint}_{self._temp_seq}", node)

    def is_dump_tagged(self, var: ir.Var) -> bool:
        return self._sema.is_dump_tagged(var)

    def is_inside_device_scope(self) -> bool:
        return self._sema.is_inside_device_scope()

    def rt_bind(self, name: str, value: object, via: PtoFrame | None = None) -> object:
        """Rebind a staged local through the trace interpreter.

        A cross-frame write-through (``via`` is a borrowing child of this
        frame, i.e. ``via is not self``) additionally consults the
        requester's own interpreter: its pending result construct must be
        closed, because the owner's bind path cannot see it (ADR-0001's
        guard — the requester's result-binding phase admits only result
        bindings, wherever the write lands).
        """
        if via is not None and via is not self:
            pending = via._require_rt().pending_result()
            if pending is not None:
                raise TypeError(
                    f"item-assignment write-through of {name!r} while the staged "
                    f"{pending.kind} in the expansion body awaits result bindings — "
                    "complete the result bindings first"
                )
        return self._require_rt().bind(name, value)

    def publish_iter_args(self, iter_args: tuple[ir.IterArg, ...]) -> None:
        """Publish a complete group of pending iter_args."""
        if self._pending_iter_args is not None:
            raise TypeError(
                "pl.intrin.carry: a second pending group before the loop opener — one pending group of iter_args at a time"
            )
        self._require_no_pending_result("pl.intrin.carry")
        self._pending_iter_args = iter_args

    def collect_yield(self, operands: list[object]) -> None:
        """Route a yield marker to the nearest permitted Construct.

        The zero-operand form is structural, never a result: it terminates a
        region (the backedge of a staged while with inferred loop-carried
        values, or a staged if's branch).
        The trace interpreter owns ordered selection — an innermost-first walk
        may cross foreign Constructs but never skips a nearer eligible target.
        Each Construct answers the typed eligibility question through
        ``zero_operand_yield``. `Frame` dispatches ACCEPT through ``on_zero_operand_yield``;
        BLOCK routes the marker to that Construct's result-yield path, where
        its arity check reports the mismatch. Ordinary result-producing yields
        use the yield-interface selection.
        """
        rt = self._require_no_pending_result("pl.intrin.yield_")
        self._reject_pending_carry("pl.intrin.yield_")
        if not operands:
            selection = rt.nearest_zero_operand_yield()
            if selection is None:
                raise TypeError("pl.intrin.yield_: no region accepts a zero-operand yield terminator here")
            if selection.action is ZeroOperandYieldAction.ACCEPT:
                selection.construct.on_zero_operand_yield(rt)
            else:
                # BLOCK: a result-taking region is nearer — the zero-operand
                # yield terminator must address that exact Construct, whose
                # arity check rejects.
                construct = selection.construct
                if not isinstance(construct, YieldTarget):
                    raise TypeError(
                        "the Construct accepting a yield terminator does not implement on_yield "
                        "(the interface tag is advertised but the surface is missing)"
                    )
                construct.on_yield(rt, operands)
            return
        construct = rt.nearest_supporting("yield")
        if construct is None:
            raise TypeError("pl.intrin.yield_: no active Construct accepts a yield terminator")
        if not isinstance(construct, YieldTarget):
            raise TypeError(
                "the Construct accepting a yield terminator does not implement on_yield "
                "(the interface tag is advertised but the surface is missing)"
            )
        construct.on_yield(rt, operands)

    def _require_pending_result(self, op: str) -> ResultBindingTarget:
        """The innermost result-producing Construct, narrowed and checked."""
        construct = self._require_rt().pending_result()
        if construct is None:
            raise TypeError(f"pl.intrin.leak: no pending result construct{op}")
        if not isinstance(construct, ResultBindingTarget):
            raise TypeError("result bindings are unavailable for this staged Construct")
        return construct

    def pending_result_construct(self) -> ResultBindingTarget | None:
        """The innermost result-producing Construct awaiting result bindings."""
        construct = self._require_rt().pending_result()
        if construct is None:
            return None
        if not isinstance(construct, ResultBindingTarget):
            raise TypeError("result bindings are unavailable for this staged Construct")
        return construct

    def create_leak_var(self, bundle: _LeakBundle, name: str, index: int) -> ir.Var:
        """Create the result variable for leak position ``index`` on the exact
        pending result Construct.

        The bundle records its originating Construct; the binding must find
        that same Construct still at the top of the active-construct stack. A
        stale binding proxy, from a completed or later region, is rejected instead of
        completing an unrelated region.
        """
        construct = self._require_pending_result("")
        origin = bundle.construct
        if origin is not construct:
            origin_kind = getattr(origin, "kind", None) or "unknown"
            raise TypeError(
                "pl.intrin.leak: a result binding proxy must bind while its own region is "
                f"the innermost active construct — this bundle's {origin_kind} region is "
                "not the top of the active-construct stack"
            )
        return construct.create_leak_var(name, index)

    def complete_leak(self, bundle: _LeakBundle) -> None:
        """Commit a fully filled result-binding group and close its region.

        The trace interpreter removes the Construct when the
        ``FinalResultBinding`` binding completes."""
        # Same interface-typed surface as create_leak_var, and the same
        # exact-region identity: the bundle's Construct must be the innermost
        # active construct.
        construct = self._require_pending_result(" awaiting commit")
        if bundle.construct is not construct:
            raise TypeError(
                "pl.intrin.leak: the bundle does not address the exact pending "
                "result construct — a leak commits its own region's results"
            )
        if bundle.n != (construct.result_arity or 0):
            raise TypeError(
                f"pl.intrin.leak arity mismatch: staged {construct.kind} expects "
                f"{construct.result_arity}, got {bundle.n}"
            )
        slots: list[ir.Var] = []
        for i in range(bundle.n):
            v = bundle.slots[i]
            if v is None:
                raise TypeError(f"result-binding group slot {i} not filled")
            slots.append(v)
        construct.complete(slots)

    def current_split_aiv_mode(self) -> ir.SplitMode | None:
        """The innermost active ``pl.split_aiv`` mode.

        This is an exact PtoConstruct property queried from the trace
        interpreter."""
        construct = self._require_rt().nearest_supporting("split_aiv")
        if construct is None:
            return None
        if not isinstance(construct, SplitAivTarget):
            raise TypeError(
                "the active construct advertising the split_aiv interface tag "
                "does not expose split_mode (the tag is advertised but the "
                "surface is missing)"
            )
        return construct.split_mode

    # —— finalization ——
    def finish(self) -> ir.Function | None:
        """Finalize the function (normal completion only) and
        store the result on the Sema."""
        if self._finished:
            return self._sema._result
        if self._pending_iter_args is not None:
            raise TypeError("pl.intrin.carry: the pending iter_args were never claimed by a loop opener")
        self._finished = True
        self._sema._reset_span_pin()
        if self._sema._parent is not None:
            # The borrowing expansion: the parent owns the function context —
            # there is nothing to end and no result to record.
            return None
        self._sema._result = self._sema._builder.end_function(self._sema._source_span)
        return self._sema._result
