# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The assembly dialect — compiler-facing inline assembly.

The ``pl.intrin`` namespace marks the unstable dialect:

* ``pl.intrin.carry(*values)`` — declares explicit loop-carried values, yields IterArg facades.
* ``pl.intrin.yield_(*values)`` — emits YieldStmt for current structure.
* ``pl.intrin.leak(*witnesses)`` — binds return_vars and finalizes the staged construct's region.
* ``pl.intrin.free_var(descriptor, *, name=...)`` — raw unresolved Var.
* ``pl.intrin.transmute(producer, type_descriptor)`` — producer-only result-type replacement.
* ``pl.intrin.spmd_direct(...)`` — exact captured ``SpmdScopeStmt(Call)``.
* ``pl.intrin.spmd_direct_call(...)`` — exact uncaptured ``SpmdScopeStmt(Call)``.

The surface is compiler-developer only, not a public stable API.  Exact IR
construction and round-trip fidelity take priority over ergonomics.

Typing: heterogeneous carry via TypeVarTuple, yield marker as
IntrinsicStmt, leak overloads for witnesses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

from pypto.language.typing.type_utils import (
    _UnwrapsToExpr,
    annotation_to_type,
    apply_result_type,
    wrap_expr,
)
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir

if TYPE_CHECKING:
    from typing_extensions import TypeVarTuple

    _Ts = TypeVarTuple("_Ts")
    _T = Any  # placeholder for overloads

__all__ = [
    "carry",
    "free_var",
    "leak",
    "spmd_direct",
    "spmd_direct_call",
    "transmute",
    "yield_",
]


# ---------------------------------------------------------------------------
# Private marker / bundle infrastructure
# ---------------------------------------------------------------------------


# The structural-bind wrapper lives in language/protocol.py beside the
# ``SupportsTraceBind`` whose return union it names; the binding proxies
# here construct it. No cycle: the trace interpreter chain never imports
# this module.
from pypto.language.protocol import _ExistingVarBinding  # noqa: PLC0415 - beside its protocol


class _CarryBundle:
    """Shared bundle for one ``pl.intrin.carry`` call."""

    __slots__ = ("initial", "slots", "filled", "bindings")

    def __init__(self, initial_facades: list[Any]) -> None:
        self.initial: list[Any] = list(initial_facades)
        self.slots: list[_ir.IterArg | None] = [None] * len(initial_facades)
        self.filled: list[bool] = [False] * len(initial_facades)
        self.bindings: list[_IterArgBinding] = []


class _IterArgBinding:
    """Binding proxy returned by ``carry`` — one per channel position."""

    __slots__ = ("_bundle", "_index", "_consumed")

    def __init__(self, bundle: _CarryBundle, index: int) -> None:
        self._bundle = bundle
        self._index = index
        self._consumed = False

    def __trace_bind__(self, ctx: Any, name: str) -> Any:
        if self._consumed:
            return None
        # Validate bundle size still matches
        initial = self._bundle.initial[self._index]
        # None is the parser's no-producer sentinel. Carry it as the typed
        # invalid TaskId value so the loop channel remains staged.
        if initial is None:
            init_expr = _ir.create_op_call("system.task_invalid", [], {}, _ir.Span.unknown())
        else:
            # unwrap facade via Sema's typing: initial is a facade with unwrap()
            try:
                init_expr = initial.unwrap()  # type: ignore[attr-defined]
            except Exception:
                raise TypeError(f"pl.intrin.carry: argument {self._index} is not a staged facade")
        # Determine type from initial's type
        init_type = init_expr.type
        # Create IterArg
        iter_arg = _ir.IterArg(name, init_type, init_expr, _ir.Span.unknown())
        # Register structurally
        self._bundle.slots[self._index] = iter_arg
        self._bundle.filled[self._index] = True
        self._consumed = True
        # If all filled, publish the complete bundle as the Frame's
        # pending iter_args group (the named Frame operation).
        if all(self._bundle.filled):
            ctx.publish_iter_args(tuple(a for a in self._bundle.slots if a is not None))
        # Return an existing-var binding so Sema registers without let
        return _ExistingVarBinding(iter_arg)


class _YieldMarker:
    """Expression-statement marker returned by ``yield_``."""

    __slots__ = ("operands",)

    def __init__(self, operands: list[Any]) -> None:
        self.operands: list[Any] = list(operands)

    def __trace_stmt__(self, ctx: Any) -> None:
        # Delegate to the Frame's yield collection (the named operation).
        ctx.collect_yield(self.operands)


class _LeakBundle:
    """Shared bundle for one ``pl.intrin.leak`` call.

    ``construct`` is the exact pending result Construct the bundle addresses. Every
    bundle is created while its region is the interpreter's exact top, and its
    binding proxies must find that same Construct still at the top of the
    active-construct stack. Result-binding methods address the exact
    Construct, so a stale proxy cannot complete a
    later unrelated region."""

    __slots__ = ("n", "witness_types", "slots", "filled", "bindings", "construct")

    def __init__(self, n: int, witness_types: list[_ir.Type] | None, construct: Any) -> None:
        self.n = n
        self.witness_types: list[_ir.Type] | None = witness_types
        self.slots: list[_ir.Var | None] = [None] * n
        self.filled: list[bool] = [False] * n
        self.bindings: list[_ResultBinding] = []
        self.construct = construct


class _ResultBinding:
    __slots__ = ("_bundle", "_index", "_consumed")

    def __init__(self, bundle: _LeakBundle, index: int) -> None:
        self._bundle = bundle
        self._index = index
        self._consumed = False

    def __trace_bind__(self, ctx: Any, name: str) -> Any:
        if self._consumed:
            return None
        # Determine expected type
        exp_type: _ir.Type | None = None
        if self._bundle.witness_types is not None:
            exp_type = self._bundle.witness_types[self._index]
        # The Frame owns result-variable creation: it asks the exact pending
        # result construct for the Var and rejects a bundle whose construct
        # is no longer the innermost active construct.
        var = ctx.create_leak_var(self._bundle, name, self._index)
        # Validate witness vs actual if witness present
        if exp_type is not None:
            # annotation_to_type already gave exp_type; check structural equal?
            # Sema has already validated witness types against staged interface
            # at bundle creation, but per-binding proxy we can also check var.type vs the
            # result-type witness. Actually var type is derived from staged; compare.
            if not _ir.structural_equal(var.type, exp_type, enable_auto_mapping=True):
                # Use same diagnostic as Sema's witness arity/type mismatch
                raise TypeError(
                    f"pl.intrin.leak result-type witness mismatch at position {self._index}: "
                    f"witness {exp_type} vs staged {var.type}"
                )
        self._bundle.slots[self._index] = var
        self._bundle.filled[self._index] = True
        self._consumed = True
        # If all filled, commit: the Frame's complete_leak performs the
        # final result binding and its deferred region finalization; the
        # interpreter removes the construct at the FinalResultBinding.
        if all(self._bundle.filled):
            ctx.complete_leak(self._bundle)
        return _ExistingVarBinding(var)


class _FreeVarBinding:
    __slots__ = ("_type", "_name")

    def __init__(self, ir_type: _ir.Type, name: str) -> None:
        self._type = ir_type
        self._name = name

    def __trace_bind__(self, ctx: Any, bind_name: str) -> Any:
        # The free_var intrinsic's name= kwarg gives the printed name_hint;
        # the bind target name may differ. Register the pre-created Var under
        # the assignment target's name without LetStmt/AssignStmt (structural
        # registration): e.g. ``esc = pl.intrin.free_var(...,
        # name="esc__FREE_VAR")`` binds the target ``esc`` to the Var whose
        # printed hint is ``esc__FREE_VAR``.
        var = _ir.Var(self._name, self._type, _ir.Span.unknown())
        return _ExistingVarBinding(var)


class _SpmdDirectBundle:
    """One canonical direct Spmd scope, emitted when an operation-result binding proxy binds."""

    __slots__ = (
        "_call",
        "_core_num",
        "_sync_start",
        "_name_hint",
        "_scope_attrs",
        "_output_var",
        "_emitted",
    )

    def __init__(
        self,
        call: _ir.Expr,
        core_num: int | _ir.Expr,
        sync_start: bool,
        name_hint: str,
        scope_attrs: list[tuple[str, Any]],
        output_var: _ir.Var,
    ) -> None:
        self._call = call
        self._core_num = core_num
        self._sync_start = sync_start
        self._name_hint = name_hint
        self._scope_attrs = scope_attrs
        self._output_var = output_var
        self._emitted = False

    def emit(self, ctx: Any, task_id_var: _ir.Var | None) -> None:
        if self._emitted:
            return
        attrs = list(self._scope_attrs)
        if task_id_var is not None:
            # ConvertToSSA requires a real definition before the scope. The
            # scope outliner replaces this placeholder with the Submit task id.
            ctx.builder.assign(
                task_id_var,
                _ir.create_op_call("system.task_invalid", [], {}, _ir.Span.unknown()),
            )
            attrs.append(("task_id_var", task_id_var))
        ctx.builder.begin_scope(
            _ir.ScopeKind.Spmd,
            core_num=self._core_num,
            sync_start=self._sync_start,
            name_hint=self._name_hint,
            attrs=attrs or None,
        )
        ctx.builder.assign(self._output_var, self._call)
        ctx.builder.end_scope()
        self._emitted = True


class _SpmdOutputBinding:
    __slots__ = ("_bundle", "_capture_task_id")

    def __init__(self, bundle: _SpmdDirectBundle, capture_task_id: bool) -> None:
        self._bundle = bundle
        self._capture_task_id = capture_task_id

    def __trace_bind__(self, ctx: Any, name: str) -> Any:
        del name
        if not self._capture_task_id:
            self._bundle.emit(ctx, None)
        # A captured dispatch emits through its TaskId binding (the two-target
        # ``out, tid = pl.intrin.spmd_direct(...)`` — the tid target routes
        # after this one, so emitting here would miss the task_id_var). A
        # single-target bind therefore drops the scope silently; making that
        # loud needs a never-emitted-bundle check in the bind protocol.
        return _ExistingVarBinding(self._bundle._output_var)


class _SpmdTaskIdBinding:
    __slots__ = ("_bundle", "_consumed")

    def __init__(self, bundle: _SpmdDirectBundle) -> None:
        self._bundle = bundle
        self._consumed = False

    def __trace_bind__(self, ctx: Any, name: str) -> Any:
        if self._consumed:
            return None
        task_id_var = ctx.builder.var(name, _ir.ScalarType(DataType.TASK_ID), _ir.Span.unknown())
        self._bundle.emit(ctx, task_id_var)
        self._consumed = True
        return _ExistingVarBinding(task_id_var)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def transmute(producer: Any, type_descriptor: Any) -> Any:
    """Replace a detached producer's stored result type without adding IR.

    ``producer`` must be an expr-wrapping facade around an ``ir.Call`` or
    ``ir.Submit``. The operation copies the producer while preserving its
    operation, operands, kwargs, attrs, and span; ordinary binding remains
    responsible for emitting the resulting statement.

    This is intentionally narrower than expression retyping: Vars, constants,
    and other expression nodes have no producer ownership and are rejected.

    Args:
        producer: An expr-wrapping facade around a detached Call or Submit.
        type_descriptor: Annotation-mode facade describing the replacement Type.

    Returns:
        A same-family facade wrapping the copied producer.

    Raises:
        TypeError: If ``producer`` is not a detached Call/Submit facade or the
            descriptor changes its facade family.
    """
    if not isinstance(producer, _UnwrapsToExpr):
        raise TypeError("pl.intrin.transmute expects a facade wrapping a detached Call or Submit")
    try:
        expr = producer.unwrap()
    except Exception as exc:
        raise TypeError(
            "pl.intrin.transmute expects a staged facade wrapping a detached Call or Submit"
        ) from exc
    if not isinstance(expr, (_ir.Call, _ir.Submit)):
        raise TypeError("pl.intrin.transmute only accepts a facade wrapping a detached Call or Submit")

    new_expr = apply_result_type(expr, type_descriptor)
    # Resolve the descriptor through the normal type-to-facade registry before
    # rebuilding the result. This prevents a Tile facade from silently carrying
    # a Tensor/Scalar result while still allowing distributed facade subclasses.
    expected_facade = wrap_expr(new_expr, new_expr.type)
    if type(expected_facade) is not type(producer):
        raise TypeError(
            "pl.intrin.transmute result type changes the producer facade family: "
            f"got {type(expected_facade).__name__}, expected {type(producer).__name__}"
        )

    # The registry build above already minted the same-class replacement
    # (the family guard enforces it); a producer carrying replace semantics
    # beyond its expr (Tuple's shared emit state) keeps them via its hook.
    replacer = getattr(producer, "__trace_replace_expr__", None)
    if callable(replacer):
        return replacer(new_expr)
    return expected_facade


def _direct_spmd_scope_bundle(  # noqa: PLR0912 - validates each assembly boundary explicitly
    fn: Any,
    args: tuple[Any, ...],
    *,
    core_num: Any,
    sync_start: bool,
    name_hint: str,
    deps: list[Any] | tuple[Any, ...] | None,
    allow_early_resolve: bool,
    predicate: Any,
) -> _SpmdDirectBundle:
    """Validate and stage one assembly-only ``SpmdScopeStmt(body=Call)``."""
    from pypto.language.op.marker import Marker  # noqa: PLC0415 - avoid language init cycle

    if not isinstance(fn, Marker):
        raise TypeError("pl.intrin.spmd_direct requires a marked program kernel as its first argument")
    if fn.function_type not in (_ir.FunctionType.InCore, _ir.FunctionType.AIC, _ir.FunctionType.AIV):
        raise TypeError(
            "pl.intrin.spmd_direct requires an existing InCore/AIC/AIV kernel, "
            f"got {fn.name!r} ({fn.function_type.name})"
        )
    if isinstance(core_num, bool):
        raise TypeError("pl.intrin.spmd_direct core_num must be a positive integer or staged expression")
    if hasattr(core_num, "unwrap"):
        core_num = core_num.unwrap()
    if not isinstance(core_num, (int, _ir.Expr)):
        raise TypeError("pl.intrin.spmd_direct core_num must be a positive integer or staged expression")
    if isinstance(core_num, int) and core_num <= 0:
        raise ValueError("pl.intrin.spmd_direct core_num must be positive")

    output_indices = [
        index
        for index, direction in enumerate(fn.param_directions)
        if direction in (_ir.ParamDirection.Out, _ir.ParamDirection.InOut)
    ]
    if len(output_indices) != 1 or len(fn.declared_return_types) != 1:
        raise TypeError(
            "pl.intrin.spmd_direct currently requires exactly one supplied Out/InOut parameter "
            "and one declared return; use the explicit assembly scope for multi-result IR"
        )
    output_index = output_indices[0]
    if output_index >= len(args) or not hasattr(args[output_index], "unwrap"):
        raise TypeError(
            "pl.intrin.spmd_direct requires its Out/InOut argument to be a supplied staged variable"
        )
    output_var = args[output_index].unwrap()
    if not isinstance(output_var, _ir.Var):
        raise TypeError("pl.intrin.spmd_direct Out/InOut argument must unwrap to an ir.Var")

    # Marker invocation owns the call signature, argument coercion, and result
    # deduction. It returns a detached Call that we install as the Spmd body.
    call_facade = fn(*args)
    call = call_facade.unwrap()
    if not isinstance(call, _ir.Call) or not isinstance(call.op, _ir.GlobalVar):
        raise TypeError("pl.intrin.spmd_direct requires a direct GlobalVar kernel call")

    scope_attrs: list[tuple[str, Any]] = []
    if deps is not None:
        dep_vars: list[_ir.Var] = []
        for dep in deps:
            if dep is None:
                continue
            if not hasattr(dep, "unwrap"):
                raise TypeError("pl.intrin.spmd_direct deps entries must be staged TaskId values")
            dep_expr = dep.unwrap()
            if not isinstance(dep_expr, _ir.Var):
                raise TypeError("pl.intrin.spmd_direct deps entries must unwrap to ir.Var")
            dep_vars.append(dep_expr)
        if dep_vars:
            scope_attrs.append(("manual_dep_edges", dep_vars))
    if allow_early_resolve:
        scope_attrs.append(("allow_early_resolve", True))
    if predicate is not None:
        if not hasattr(predicate, "unwrap"):
            raise TypeError("pl.intrin.spmd_direct predicate must be a staged expression")
        scope_attrs.append(("predicate", predicate.unwrap()))

    return _SpmdDirectBundle(call, core_num, sync_start, name_hint, scope_attrs, output_var)


def spmd_direct(
    fn: Any,
    *args: Any,
    core_num: Any,
    sync_start: bool = False,
    name_hint: str = "",
    deps: list[Any] | tuple[Any, ...] | None = None,
    allow_early_resolve: bool = False,
    predicate: Any = None,
) -> tuple[Any, Any]:
    """Emit a direct ``SpmdScopeStmt`` and capture its producer TaskId.

    This is the assembly dialect's exact form for a scope whose body is one
    direct call to an existing device kernel. Unlike :func:`pl.spmd_submit`,
    it does not create a first-class ``ir.Submit``: ``OutlineClusterScopes``
    receives and outlines ``SpmdScopeStmt(body=Assign(Call(GlobalVar)))``.

    The intrinsic currently supports one supplied ``Out`` or ``InOut``
    parameter and a matching single return. It must be unpacked as
    ``out, tid = pl.intrin.spmd_direct(...)``.
    """
    bundle = _direct_spmd_scope_bundle(
        fn,
        args,
        core_num=core_num,
        sync_start=sync_start,
        name_hint=name_hint,
        deps=deps,
        allow_early_resolve=allow_early_resolve,
        predicate=predicate,
    )
    return _SpmdOutputBinding(bundle, capture_task_id=True), _SpmdTaskIdBinding(bundle)


def spmd_direct_call(
    fn: Any,
    *args: Any,
    core_num: Any,
    sync_start: bool = False,
    name_hint: str = "",
    deps: list[Any] | tuple[Any, ...] | None = None,
) -> Any:
    """Emit an uncaptured direct ``SpmdScopeStmt`` for one existing kernel call.

    The result is bound as ``out = pl.intrin.spmd_direct_call(...)``. A
    producer TaskId of this dispatch cannot be captured here (that is
    :func:`spmd_direct`'s two-target form), but ``deps=`` may still declare
    edges on *other* producers' TaskIds — the consumer needs no capture of
    its own to wait on one.
    """
    bundle = _direct_spmd_scope_bundle(
        fn,
        args,
        core_num=core_num,
        sync_start=sync_start,
        name_hint=name_hint,
        deps=deps,
        allow_early_resolve=False,
        predicate=None,
    )
    return _SpmdOutputBinding(bundle, capture_task_id=False)


_CarryT0 = TypeVar("_CarryT0")
_CarryT1 = TypeVar("_CarryT1")
_CarryT2 = TypeVar("_CarryT2")
_CarryT3 = TypeVar("_CarryT3")


@overload
def carry(v0: _CarryT0) -> tuple[_CarryT0]: ...


@overload
def carry(v0: _CarryT0, v1: _CarryT1) -> tuple[_CarryT0, _CarryT1]: ...


@overload
def carry(v0: _CarryT0, v1: _CarryT1, v2: _CarryT2) -> tuple[_CarryT0, _CarryT1, _CarryT2]: ...


@overload
def carry(
    v0: _CarryT0, v1: _CarryT1, v2: _CarryT2, v3: _CarryT3
) -> tuple[_CarryT0, _CarryT1, _CarryT2, _CarryT3]: ...


def carry(*values: Any) -> tuple[Any, ...]:  # type: ignore[reportInconsistentOverload]
    """Declare loop-carried arguments.

    One call declares the entire ordered group::

        (x_iter,) = pl.intrin.carry(x)
        (a_iter, b_iter) = pl.intrin.carry(a, b)

    Each returned binding proxy implements ``__trace_bind__`` and registers an
    ``IterArg`` without emitting a ``LetStmt``. The final proxy publishes the
    complete group as pending `iter_args` for the immediately following loop
    opener.
    """
    if not values:
        raise TypeError("pl.intrin.carry: at least one value required")
    bundle = _CarryBundle(list(values))
    bindings: list[_IterArgBinding] = []
    for i in range(len(values)):
        bindings.append(_IterArgBinding(bundle, i))
    bundle.bindings = bindings
    return tuple(bindings)  # type: ignore[return-value]


def yield_(*values: Any) -> Any:
    """Emit a yield terminator inside the nearest permitted Construct.

    Use it as an expression statement::

        pl.intrin.yield_(x_next, y_next)

    The marker routes to the nearest Construct accepting yield operands (a
    loop with explicit `iter_args` or a staged if); the operand arity must
    match that Construct's results. The zero-operand form is structural,
    never a result: ``pl.intrin.yield_()`` is the backedge terminator of a
    staged while with inferred loop-carried values, or a staged if's
    zero-operand branch terminator, legal only while such a Construct is active.
    """
    return _YieldMarker(list(values))


@overload
def leak() -> Any: ...  # type: ignore[overload-cannot-match]


@overload
def leak(first: Any, *rest: Any) -> tuple[Any, ...]: ...  # type: ignore[overload-cannot-match]


def leak(*witnesses: Any) -> Any:  # type: ignore[reportInconsistentOverload]
    """Bind operation results after a staged region closes.

    Both forms address the exact staged Construct that is in the result-
    binding phase, and every returned binding proxy must bind against that
    same Construct:

    Zero-argument form reads arity from the pending result Construct::

        (acc_out,) = pl.intrin.leak()

    Witnessed form supplies facade type descriptors for type-checking
    precision; the witness arity must match the staged Construct's result
    arity::

        acc_out, tile_out = pl.intrin.leak(
            pl.Scalar[pl.FP32],
            pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec],
        )
    """
    from pypto.language.ambient import active_frame  # noqa: PLC0415

    frame = active_frame()
    if not witnesses:
        # Unsafe tier: read the arity from the exact pending result
        # Construct (it is the interpreter's exact top).
        if frame is None:
            raise RuntimeError("pl.intrin.leak() outside tracing has no arity — use witnessed form")
        construct = frame.pending_result_construct()
        if construct is None:
            raise RuntimeError("pl.intrin.leak(): no pending result construct awaiting leak")
        n = construct.result_arity
        if n is None:
            raise RuntimeError("pl.intrin.leak(): pending result construct has no result arity")
        bundle = _LeakBundle(n, None, construct)
        bindings = [_ResultBinding(bundle, i) for i in range(n)]
        bundle.bindings = bindings
        if n == 1:
            return (bindings[0],)  # type: ignore[return-value]
        return tuple(bindings)  # type: ignore[return-value]
    else:
        # Witnessed pure form: witnesses are annotation-only facades like
        # pl.Scalar[pl.FP32]. The bundle binds to the exact staged region
        # at creation — a result binding addresses one region's results, and a proxy
        # that outlives its region (or arrives at a later one) is rejected
        # at the bind.
        if frame is None:
            raise RuntimeError(
                "pl.intrin.leak() outside tracing — a leak addresses the staged region it is created against"
            )
        construct = frame.pending_result_construct()
        if construct is None:
            raise RuntimeError("pl.intrin.leak(): no pending result construct awaiting leak")
        types: list[_ir.Type] = []
        for w in witnesses:
            try:
                t = annotation_to_type(w)
            except Exception as exc:
                raise TypeError(
                    f"pl.intrin.leak result-type witness {w!r} is not a valid type descriptor: {exc}"
                ) from exc
            types.append(t)
        n = len(types)
        if construct.result_arity != n:
            raise TypeError(
                f"pl.intrin.leak: witness arity {n} mismatches the staged "
                f"{construct.kind} region's {construct.result_arity} results"
            )
        bundle = _LeakBundle(n, types, construct)
        bindings = [_ResultBinding(bundle, i) for i in range(n)]
        bundle.bindings = bindings
        return tuple(bindings) if n != 1 else (bindings[0],)


def free_var(type_descriptor: Any, *, name: str) -> Any:
    """Create raw unresolved Var.

    For malformed/intermediate IR inspection; structurally distinct from UndefExpr.
    """
    if not name:
        raise ValueError("pl.intrin.free_var: name= must be a non-empty string")
    try:
        ir_type = annotation_to_type(type_descriptor)
    except Exception as exc:
        raise TypeError(f"pl.intrin.free_var type descriptor {type_descriptor!r} invalid: {exc}") from exc
    return _FreeVarBinding(ir_type, name)
