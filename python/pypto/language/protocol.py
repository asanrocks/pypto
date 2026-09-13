# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: E501

"""The PTO trace protocol — the hook contract between the shared language
layer and the tracing frontend.

Shared language objects (constructs, intrinsics, markers) implement these
hooks; the tracing frontend drives them. The module also owns
:func:`is_staged`, the value-recognition query (facades plus trace-protocol
membership)."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pypto.language.diagnostics import ParserSyntaxError, concise_error_message
from pypto.language.facade import FACADE_TYPES, Facade
from pypto.pypto_core import ir

if TYPE_CHECKING:
    from pypto.ir.builder import IRBuilder
    from pypto.language.intrin import _LeakBundle
    from pypto.language.tracer.collector.interpreter import Interpreter
    from pypto.language.tracer.constructs import PtoForConstruct as _PtoForConstruct
    from pypto.language.tracer.semantics import PtoFrame

    PtoForConstruct = _PtoForConstruct

__all__ = [
    "ResultBindingTarget",
    "SplitAivTarget",
    "TraceContext",
    "YieldTarget",
    "_ForDescriptor",
    "_ResultBindingProxy",
    "_LoopDescriptor",
    "_ConstructAttachingBinding",
    "_SplitAivDescriptor",
    "_ExistingVarBinding",
    "SupportsTraceStmt",
    "SupportsTraceBind",
    "_TraceBindReturn",
    "SupportsTraceEnter",
    "SupportsTraceExit",
    "SupportsTraceIter",
    "_WithDescriptor",
    "_DeferredSpmdCarrier",
    "_RuntimeDeferredSpmdCarrier",
    "_DeferredDeclarableScope",
    "_RuntimeDeferredDeclarableScope",
    "is_staged",
]


# Trace-protocol hooks are structural (duck-typed): every trace iterable,
# trace context manager, and binding proxy satisfies its relevant protocol
# without inheritance. ``runtime_checkable`` makes ``isinstance`` a
# ``hasattr`` check. Hooks receive the explicit :class:`TraceContext`, never
# a raw builder.
@runtime_checkable
class SupportsTraceIter(Protocol):
    """A trace iterable (``__trace_iter__`` — create_for_construct)."""

    def __trace_iter__(self, ctx: TraceContext) -> typing.Any: ...


@runtime_checkable
class SupportsTraceEnter(Protocol):
    """A trace context manager (``__trace_enter__`` — create_with_construct)."""

    def __trace_enter__(
        self,
        ctx: TraceContext,
        as_name: str | None = None,
        single_call_body: bool = False,
        reads_block_idx: bool = False,
    ) -> typing.Any: ...


@runtime_checkable
class SupportsTraceBind(Protocol):
    """A trace-bindable object (``__trace_bind__`` — binding).

    The closed return vocabulary (see :data:`_TraceBindReturn`) is an
    ``ir.Var``, a :class:`_ExistingVarBinding` wrapper around an
    ``ir.Var``/``ir.IterArg`` registered without ``LetStmt``/``AssignStmt``,
    a transform-and-defer facade that `Frame` redispatches, or ``None``
    (decline and use the ordinary staged-binding path).
    """

    def __trace_bind__(self, ctx: TraceContext, name: str) -> _TraceBindReturn: ...


@runtime_checkable
class ResultBindingTarget(Protocol):
    """A result-producing Construct awaiting result bindings.

    The Frame-facing half of the result-binding interface: the exact pending
    result Construct (queried through the trace interpreter) answers the
    result-binding progress question and provides result-variable and commit
    operations. The interpreter's ``nearest_supporting``/``pending_result``
    queries return it; narrowing happens here, on the Frame side.
    """

    kind: str

    @property
    def completed(self) -> bool: ...

    @property
    def result_arity(self) -> int | None: ...

    def create_leak_var(self, name: str, index: int) -> ir.Var: ...

    def complete(self, slots: list[ir.Var]) -> None: ...


@runtime_checkable
class YieldTarget(Protocol):
    """A Construct that accepts ``pl.intrin.yield_`` operands.

    The yield interface's typed surface: loops with explicit `iter_args` and
    staged if Constructs advertise the ``"yield"`` interface tag; `Frame`
    dispatches ``on_yield`` to the exact Construct the interpreter selected.
    """

    def on_yield(self, rt: Interpreter[Facade], operands: list[object]) -> None: ...


@runtime_checkable
class SplitAivTarget(Protocol):
    """A loop that recorded a ``pl.split_aiv`` region mode at activation."""

    @property
    def split_mode(self) -> ir.SplitMode | None: ...


@runtime_checkable
class _ResultBindingProxy(Protocol):
    """The result-binding proxy marker that owns a ``_bundle`` slot.

    The Frame recognizes this protocol during the result-binding phase: an
    ordinary binding is rejected while a staged Construct awaits result
    bindings, but a result-binding proxy is permitted. Membership is the
    marker, consistent with the other trace protocols.
    """

    _bundle: _LeakBundle


class _ExistingVarBinding:
    """A binding result holding an already-created ``ir.Var`` or ``ir.IterArg``.

    The variable is registered for the source name without ``LetStmt`` or
    ``AssignStmt``. This is the one non-facade member of the binding return
    union and accompanies the ``SupportsTraceBind`` implementation that
    produces it.
    """

    __slots__ = ("var",)

    var: ir.Var  # an IterArg is a Var (the loop-input form)

    def __init__(self, var: ir.Var) -> None:
        self.var = var


#: The closed ``__trace_bind__`` return union — the Frame's dispatch
#: (``_bind_dispatch``) narrows by isinstance over exactly these.
_TraceBindReturn = ir.Var | _ExistingVarBinding | Facade | None


@runtime_checkable
class SupportsTraceExit(Protocol):
    """A trace-protocol object with a close hook (``__trace_exit__``)."""

    def __trace_exit__(self, ctx: TraceContext) -> object: ...


@runtime_checkable
class _LoopDescriptor(Protocol):
    """A trace iterable that reports its origin kind.

    The kind vocabulary (``range``/``parallel``/``pipeline``/``unroll``)
    drives validation of the loop opener that follows ``pl.intrin.carry``
    and transfer rejection — recorded
    debt (interface tags should eventually replace kind strings); typed
    here as the interim honest surface.
    """

    kind: str


@runtime_checkable
class _ConstructAttachingBinding(Protocol):
    """A loop-target binding proxy retaining its exact PtoForConstruct for
    deferred region construction (``attach_construct``)."""

    def attach_construct(self, construct: typing.Any) -> None: ...


@runtime_checkable
class SupportsTraceStmt(Protocol):
    """An expression-statement marker (``__trace_stmt__``).

    A marker may declare itself **annotation state** with a true
    ``annotation_state`` class attribute: it performs no IR collection of its
    own (a body-prologue directive — ``pl.func_attr``), so the trace
    interpreter's first-statement fact does not count it. Markers that emit
    (``pl.intrin.yield_``) leave the attribute unset. The attribute is a
    structural convention, deliberately not a protocol member: a data member
    on a runtime-checkable protocol would enter every ``isinstance`` check.
    """

    def __trace_stmt__(self, ctx: TraceContext) -> None: ...


@runtime_checkable
class _SplitAivDescriptor(Protocol):
    """A trace iterable carrying a ``pl.split_aiv`` region mode."""

    @property
    def split_aiv_mode(self) -> ir.SplitMode: ...


@runtime_checkable
class _ForDescriptor(Protocol):
    """The trace-iterable surface driven by PtoForConstruct.

    ``begin`` constructs the loop when the loop-target binding activates, and the
    close hook finalizes it. Origin kind and split mode are optional member
    checks (:class:`_LoopDescriptor`, :class:`_SplitAivDescriptor`)."""

    def begin(
        self,
        builder: IRBuilder,
        name: str,
        already_in_incore: bool = False,
        sticky_dump_vars: list[ir.Var] | None = None,
    ) -> ir.Var: ...

    def __trace_exit__(self, ctx: TraceContext) -> object: ...


class _WithDescriptor(Protocol):
    """The trace-context-manager surface PtoWithConstruct closes.

    The close hook runs after the enter hook has produced any binding proxy.
    """

    def __trace_exit__(self, ctx: TraceContext) -> object: ...


class _DeferredSpmdCarrier(Protocol):
    """A with-descriptor whose InCore-carrier decision is deferred.

    The plain ``with pl.spmd(...)`` form opens its inner InCore region only
    when the body proves itself inline (a per-block-index read); a dispatch
    body stays bare. The Frame drives the trigger through this surface and
    the enter-side ``_carrier_pending`` flag (a plain attribute, so it is a
    getattr member-check rather than part of the runtime-checkable surface).
    The body can also supply the region itself (a nested
    ``pl.at(level=pl.Level.CORE_GROUP, ...)`` carrier), which the Frame
    reports through :meth:`cancel_pending_carrier`: the region already exists,
    so the carrier must neither open at the body's first statement nor at the
    construct's close."""

    def open_pending_carrier(self, builder: IRBuilder) -> None: ...

    def cancel_pending_carrier(self) -> None: ...


@runtime_checkable
class _RuntimeDeferredSpmdCarrier(_DeferredSpmdCarrier, Protocol):
    """The runtime-checkable view of ``_DeferredSpmdCarrier``.

    Method-only surface, so the structural isinstance stays sound."""


class _DeferredDeclarableScope(Protocol):
    """A with/for descriptor whose scope begin is deferred so the body's
    leading ``pl.set_cache_policy(...)`` declarations can join its attrs.

    A ``ScopeStmt``'s attrs are fixed at ``begin_scope`` and the body executes
    after it — so a declarable scope registers itself with the Frame at enter
    and begins at the body's first IR-emitting statement (or its own close,
    whichever comes first), with every buffered declaration folded into the
    ``cache_policy_vars`` attr at the parser's attr position. The declarations
    must therefore *precede* the scope's first statement — the loud, stricter
    form of the parser's position-independent hoist (the printer re-emits
    declarations first, so printed output always satisfies it)."""

    def attach_cache_policy(self, var: ir.Var, policy: int) -> None: ...

    def open_deferred_scope(self, builder: IRBuilder) -> None: ...


@runtime_checkable
class _RuntimeDeferredDeclarableScope(_DeferredDeclarableScope, Protocol):
    """The runtime-checkable view of ``_DeferredDeclarableScope``.

    Method-only surface, so the structural isinstance stays sound."""


class TraceContext:
    """The tracing context passed to trace-protocol hooks.

    It is passed to ``__trace_bind__(ctx, name)``, ``__trace_iter__(ctx)``,
    ``__trace_enter__(ctx, as_name)``, ``__trace_exit__(ctx)``, and
    ``__trace_stmt__(ctx)``. It is explicit rather than ambient and is never
    a raw builder, the Sema, or the Frame. It exposes builder access,
    temporary emission, and approved function-property and active-construct
    queries needed by stdlib code.
    """

    def __init__(self, frame: PtoFrame) -> None:
        self._frame = frame

    @property
    def builder(self) -> IRBuilder:
        """The builder exposed to the language plugin during the hook."""
        return self._frame._sema._builder

    # —— emission ——
    def let_temp(self, node: ir.Expr, hint: str) -> ir.Var:
        """Create a temporary name, emit ``LetStmt(name, node)``, and return its variable."""
        return self._frame.let_temp(node, hint)

    def defer_scope(self, construct: _DeferredDeclarableScope) -> None:
        """Register a deferred declarable scope with the Frame (see
        ``_DeferredDeclarableScope``). For-descriptors use this from their
        ``begin`` — their region opens at the first target binding, which the
        with-form's Frame-side registration point never sees."""
        self._frame.register_deferred_scope(construct)

    def eval(self, node: ir.Expr) -> None:
        """Emit an ``EvalStmt`` for a trace-protocol object's structural lowering."""
        frame = self._frame
        frame._reject_pending_carry("a trace-context emission")
        frame._require_no_pending_result("a trace-context emission")
        self.builder.eval_stmt(node)

    # —— function-property and scope-fact queries ——
    def is_inside_scope(self, scope_kind: ir.ScopeKind) -> bool:
        """Whether the current build position is nested inside an open scope
        of the given kind (the builder's Python-side scope-stack mirror)."""
        return self.builder.is_inside_scope(scope_kind)

    def is_inside_incore(self) -> bool:
        """Whether the innermost region is an InCore region."""
        return self.is_inside_scope(ir.ScopeKind.InCore)

    def is_inside_manual_scope(self) -> bool:
        """Whether the build position is nested inside an open MANUAL runtime
        scope — the ambient fact behind the runtime's nesting rules (a MANUAL
        scope may not nest in another MANUAL one, an AUTO scope may not nest in
        a MANUAL one). The parser tracks the same fact as
        ``_manual_scope_depth``."""
        return self.builder.is_inside_manual_scope()

    def function_type(self) -> ir.FunctionType:
        """The active function's ``FunctionType`` (its body's kind)."""
        return self._frame._sema._function_type

    def is_orchestration(self) -> bool:
        """Is the active function an orchestration body (Orchestration or Graph)?

        Both are host task-orchestration code — ``ir.is_orchestration_like``
        — so a caller acting on the function *because it orchestrates tasks*
        must accept either (the ``tensor.dim`` extent-symbol fold's gate:
        a Graph body binds the same signature extents an Orchestration one
        does)."""
        return ir.is_orchestration_like(self.function_type())

    def param_extent_symbol(self, var_id: int) -> bool:
        """Is ``var_id`` a param-declared extent symbol of this function?"""
        return var_id in self._frame._sema._param_dim_symbols

    def auto_scope_enabled(self) -> bool:
        """Does the active function own automatic runtime-scope placement?"""
        return self._frame._sema.auto_scope_enabled()

    def is_dump_tagged(self, var: ir.Var) -> bool:
        return self._frame._sema.is_dump_tagged(var)

    def sticky_dump_vars(self) -> list[ir.Var]:
        """The function's forward-sticky dump-tagged tensor Vars."""
        return self._frame._sema.sticky_dump_vars()

    def is_inside_device_scope(self) -> bool:
        return self._frame._sema.is_inside_device_scope()

    # —— named structural operations (func_attr) ——
    def collect_func_attr(self, attrs: dict[str, object]) -> None:
        """Merge one ``pl.func_attr`` prologue directive into the function's attrs.

        The expression-statement marker's named operation. The prologue rule
        is enforced here: the directive must precede every other statement of
        the body (the trace interpreter's event-aware "first statement"
        query — the parser enforces the same rule at the AST level), because
        an attribute describes the whole function and must not appear to
        start applying partway down a body. The merge itself is the
        builder's ``add_function_attrs`` — the unique-key check there covers
        duplicates across directives and against the decorator's ``attrs=``.
        """
        if self._frame._require_rt().collected_statement():
            raise ParserSyntaxError(
                f"pl.func_attr() must appear before every other statement in {self._frame._sema._name!r}",
                hint=(
                    "Move the pl.func_attr(...) call to the top of the function body. "
                    "A function attribute describes the whole function, so it is "
                    "declared in the prologue; only parameters are referenceable there."
                ),
            )
        try:
            self.builder.add_function_attrs(attrs)
        except ValueError as exc:
            raise ParserSyntaxError(concise_error_message(exc)) from exc

    # —— staged-local binding ——
    def resolve_local_name(self, expr: object) -> str | None:
        return self._frame.resolve_local_name(expr)

    def rebind_local(self, name: str, value: object) -> object:
        """Rebind a staged local through the trace interpreter."""
        return self._frame.rt_bind(name, value)

    # —— named structural operations (carry / yield / leak) ——
    def publish_iter_args(self, iter_args: tuple[ir.IterArg, ...]) -> None:
        self._frame.publish_iter_args(iter_args)

    def collect_yield(self, operands: list[object]) -> None:
        self._frame.collect_yield(operands)

    def pending_result(self) -> ResultBindingTarget | None:
        return self._frame.pending_result_construct()

    def create_leak_var(self, bundle: _LeakBundle, name: str, index: int) -> ir.Var:
        return self._frame.create_leak_var(bundle, name, index)

    def complete_leak(self, bundle: _LeakBundle) -> None:
        self._frame.complete_leak(bundle)


def is_staged(v: object) -> bool:
    """The recognition query: is ``v`` a staged stdlib value?

    Facades participate in staging, and a trace-protocol object participates
    by hook membership (``__trace_iter__`` / ``__trace_enter__``). Host values
    such as literals, raw ``ir.Expr`` objects, and lists do not."""
    if isinstance(v, FACADE_TYPES):
        return True
    return isinstance(v, (SupportsTraceIter, SupportsTraceEnter))


def is_staged_annotation(annotation: object) -> bool:
    """The annotation-tier recognition query for the signature inspection
    contract: is ``annotation`` a staged kind's annotation form?

    The subscript kinds yield facade instances (``pl.Tensor[[s], dt]`` → a
    ``Tensor``), so value-tier recognition covers them. The parameterless
    kinds have no subscript form — the parser's spelling is the bare name
    (``pld.CommCtx``), so the class itself is the annotation and is admitted
    as itself."""
    if isinstance(annotation, FACADE_TYPES):
        return True
    from pypto.language.distributed.typing import CommCtx  # noqa: PLC0415 - cycle

    return annotation is CommCtx
