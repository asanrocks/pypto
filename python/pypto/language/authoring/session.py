# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""The authoring session — one recipe invocation's ownership.

The runtime beneath both authoring protocols: the borrowed-or-root
:class:`AuthoringSession`, its LIFO stack of function units (one-shot Semae
with a created -> collecting -> finalized lifecycle), the unit effects
(``FunctionContext.collect_body`` / ``finalize`` / ``discard``), the
semantic control capabilities (:class:`ControlContext`), and the borrowed
collection (``splice_body``). The construct tier's region state lives on
the same session, so :class:`~pypto.language.authoring.construct.ConstructContext`
and :class:`~pypto.language.authoring.recipe.RecipeContext` are two views
over one owner.

This module is the sanctioned client of the tracer's internals — recipe
code sees only the context views. See
``docs/en/dev/language/07-authoring-recipes.md`` for the contract and the
invariants.
"""

from __future__ import annotations

import contextvars
import inspect
import types
from collections.abc import Sequence
from typing import Any, cast

from pypto.language.ambient import reset_active_frame, set_active_frame
from pypto.language.authoring.contracts import (
    CollectedBody,
    FunctionArtifact,
    FunctionContract,
    ParamSpec,
)
from pypto.language.tracing.collector.driver import InstrumentedFunction
from pypto.language.tracing.entry.declaration import Param
from pypto.language.tracing.entry.traced import TracedCallable
from pypto.language.tracing.execution import pin_declared_returns, run_instrumented
from pypto.language.tracing.semantics import PtoFrame, PtoSema

__all__ = ["AuthoringSession", "ControlContext", "FunctionContext"]


# -----------------------------------------------------------------------------------------------------------
# The collection helpers
# -----------------------------------------------------------------------------------------------------------


def _prepared(body: object) -> TracedCallable:
    """Normalize a body argument to its prepared :class:`TracedCallable`.

    Accepts a prepared ``TracedCallable`` or a plain function (prepared on the
    spot — preparation is idempotent by caching code-shaped facts only, but a
    recipe should hand the prepared object to avoid re-instrumenting per
    occurrence)."""
    if isinstance(body, TracedCallable):
        return body
    if isinstance(body, types.FunctionType):
        return TracedCallable.from_callable(body)
    raise TypeError(f"a recipe collects a prepared body — a TracedCallable or a plain function; got {body!r}")


#: The bodies currently being collected, innermost last — the recursion
#: guard. A body that collects itself (directly or through a composed
#: callee) would trace its own body again at every level with no staged
#: condition to stop it, so the runtime refuses it the way the splice
#: refuses nesting: loudly, before any effect.
_COLLECTING: contextvars.ContextVar[tuple[int, ...]] = contextvars.ContextVar(
    "pto_collecting_bodies", default=()
)


def _collecting_guard(prepared: TracedCallable, owner: str) -> contextvars.Token[tuple[int, ...]]:
    """Enter one body's collection — loud on re-entry of the same body.

    Identity is the prepared body's kernel (stable across a build — the
    ``pl.authoring.body`` object holds one preparation); a source-backed body has no
    kernel and falls back to the preparation's identity.
    """
    kernel = getattr(prepared._preparation, "_kernel", None)  # noqa: SLF001 - the sanctioned client
    identity = id(prepared) if kernel is None else id(kernel)
    name = prepared.__name__
    if identity in _COLLECTING.get():
        raise TypeError(
            f"{owner}: the body {name!r} is already being collected — a body "
            "cannot collect itself (directly or through a composed callee); the trace "
            "would recurse with no staged condition to stop it"
        )
    return _COLLECTING.set(_COLLECTING.get() + (identity,))


def _live_function(prepared: TracedCallable) -> types.FunctionType:
    """The prepared body's live instrumented function (its definition environment)."""
    return prepared._preparation.live_function()  # noqa: SLF001 - the sanctioned tracer-internal client


def _prefix(prepared: TracedCallable) -> str:
    """The prepared body's generated event prefix."""
    return prepared._preparation.prefix  # noqa: SLF001 - the sanctioned tracer-internal client


def _static_extent(extent: object) -> bool:
    """Is one shape extent static — an ``int`` or an ``ir.ConstInt``?

    The symbolic markers (``True``/``False`` and extent ``Var``s) are the
    dynamic extents a derived signature cannot spell."""
    if isinstance(extent, bool):
        return False
    if isinstance(extent, int):
        return True
    value = getattr(extent, "value", None)  # ir.ConstInt
    return isinstance(value, int)


def _derive_param_specs(kernel: types.FunctionType, args: Sequence[object]) -> list[ParamSpec]:
    """Derive a unit's parameter specs from staged collection arguments.

    The parser's annotationless-callee rule as the authoring runtime's
    derived-signature lane: every positional argument must be staged and
    statically shaped — the body's parameter names pair with the arguments'
    types, and the values themselves flow to the call site, never into the
    body. A host value or a symbolic extent needs an explicit
    :class:`FunctionContract`.
    """
    signature = inspect.signature(kernel)
    names: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(
                f"derived signatures need plain positional parameters — {kernel.__name__!r} "
                f"declares {parameter.kind.name} {parameter.name!r}; declare the ABI with a "
                "FunctionContract(params=[...]) instead"
            )
        names.append(parameter.name)
    if len(args) != len(names):
        raise TypeError(
            f"collect_body derives {kernel.__name__!r}'s signature from its arguments — "
            f"{len(names)} parameter(s), {len(args)} argument(s)"
        )
    specs: list[ParamSpec] = []
    for name, arg in zip(names, args):
        from pypto.language.protocol import is_staged  # noqa: PLC0415 - the facade/typing cycle

        if not is_staged(arg):
            raise TypeError(
                f"collect_body derives parameter {name!r} of {kernel.__name__!r} from its "
                f"argument — {arg!r} is a host value; a host value crosses only through a "
                "declared FunctionContract (the jit envelope's baked-constant lane)"
            )
        param_type = cast("Any", arg).unwrap().type
        shape = getattr(param_type, "shape", None)
        if shape is not None and not all(_static_extent(e) for e in shape):
            raise TypeError(
                f"collect_body derives parameter {name!r} of {kernel.__name__!r} from its "
                "argument, whose type carries a symbolic extent — a dynamic extent crosses "
                "only through a declared FunctionContract (the specializer's extent rebinding)"
            )
        specs.append(ParamSpec(name, param_type))
    return specs


# -----------------------------------------------------------------------------------------------------------
# The session and its units
# -----------------------------------------------------------------------------------------------------------


class _FunctionUnit:
    """One pushed function: a one-shot Sema, its frame, and a lifecycle.

    The states: ``ACTIVE`` (collectable — in practice exactly one collection),
    then ``FINALIZED`` (the contract step ran; the artifact is minted) or
    ``DISCARDED`` (a never-collected unit a recipe abandoned — its empty
    Sema is dropped, not finalized). A unit never reverses; the Sema
    beneath it is one-shot by design.
    """

    _ACTIVE = "active"
    _FINALIZED = "finalized"
    _DISCARDED = "discarded"

    def __init__(self, session: AuthoringSession, sema: PtoSema, contract: FunctionContract) -> None:
        self.session = session
        self.sema = sema
        self.contract = contract
        self.state = self._ACTIVE
        self.artifact: FunctionArtifact | None = None
        self.body_name: str = contract.name

    @property
    def frame(self) -> PtoFrame:
        """The unit's frame — created at the first parameter binding."""
        frame = self.sema.frame
        if frame is None:
            raise RuntimeError(
                f"the function unit {self.contract.name!r} has no frame — its parameters "
                "were never bound (collect_body binds them)"
            )
        return frame


class FunctionContext:
    """The capability view of one function unit — the ``fn`` a recipe holds.

    Owns the two unit effects: :meth:`collect_body` (execute one prepared
    body inside the unit) and :meth:`finalize` (the contract step). The view
    is alive only inside its recipe invocation; using it after the session
    closed is a loud error."""

    __slots__ = ("_unit",)

    def __init__(self, unit: _FunctionUnit) -> None:
        self._unit = unit

    # —— read-only facts ——

    @property
    def name(self) -> str:
        """The unit's function name."""
        return self._unit.contract.name

    @property
    def finalized(self) -> bool:
        """Whether the unit's contract step already ran."""
        return self._unit.state is _FunctionUnit._FINALIZED

    # —— the effects ——

    def collect_body(
        self,
        body: object,
        *,
        args: Sequence[object] = (),
        kwargs: dict[str, object] | None = None,
    ) -> CollectedBody:
        """Execute one prepared body inside this unit.

        A fresh lexical frame (definition-site globals and closures stay
        live); parameters are the contract's declared ABI or — for a
        contract without ``params`` — derived from the staged arguments'
        types, in which case the argument *values* are the call site's, not
        the body's. Exactly one collection per unit; the builder function
        closes when the body completes (the runner's normal finalization).
        """
        unit = self._unit
        session = unit.session
        session._require_live()
        if unit.state is _FunctionUnit._FINALIZED:
            raise TypeError(
                f"the function unit {unit.contract.name!r} is finalized — one body per "
                "unit; push a fresh function for another collection"
            )
        if unit.sema.frame is not None:
            raise TypeError(
                f"the function unit {unit.contract.name!r} already collected its body — "
                "one collection per unit; push a fresh function for another collection"
            )
        prepared = _prepared(body)
        contract = unit.contract
        if contract.params is not None:
            if args or kwargs:
                raise TypeError(
                    f"collect_body into {contract.name!r}: a declared-ABI unit binds its own "
                    "parameters — the caller's values flow to emit_call, not into the body"
                )
            call_args: list[object] = list(
                unit.sema.bind_typed_parameters(
                    [(spec.name, spec.type, spec.direction) for spec in contract.params]
                )
            )
        else:
            if kwargs:
                raise TypeError(
                    f"collect_body into {contract.name!r}: a derived-signature unit takes "
                    "positional arguments only (pass the values to emit_call positionally)"
                )
            specs = _derive_param_specs(_live_function(prepared), args)
            call_args = list(
                unit.sema.bind_typed_parameters([(spec.name, spec.type, spec.direction) for spec in specs])
            )
        unit.body_name = prepared.__name__
        runner = InstrumentedFunction(_live_function(prepared), _prefix(prepared), unit.sema)
        guard = _collecting_guard(prepared, f"collecting into {unit.contract.name!r}")
        try:
            value = run_instrumented(runner, unit.sema, call_args)
        finally:
            _COLLECTING.reset(guard)
        return CollectedBody(unit, value)

    def finalize(self, collected: CollectedBody | None = None) -> FunctionArtifact:
        """The contract step: check return paths and mint the artifact.

        Exactly once per unit, in LIFO order among the session's units.
        ``collected`` is the unit's own receipt (identity-checked); the
        return paths are checked for disagreement, a declared ``returns``
        contract is compared and pinned, and the immutable artifact is
        minted. Nothing is emitted here — the artifact is inert until
        ``emit_call`` / ``materialize_entry``.
        """
        unit = self._unit
        session = unit.session
        session._require_live()
        if collected is not None and collected._unit is not unit:
            raise TypeError(
                "finalize: the collected body belongs to another unit — each unit "
                "finalizes its own collection"
            )
        if unit.state is _FunctionUnit._FINALIZED:
            raise TypeError(f"the function unit {unit.contract.name!r} is already finalized")
        if unit.state is _FunctionUnit._DISCARDED:
            raise TypeError(f"the function unit {unit.contract.name!r} was discarded")
        unfinalized = session._unfinalized_units()
        if unfinalized and unfinalized[-1] is not unit:
            raise TypeError(
                f"finalize: units finalize in LIFO order — {unfinalized[-1].contract.name!r} is still open"
            )
        if unit.sema.frame is None:
            raise TypeError(
                f"finalize: the function unit {unit.contract.name!r} collected no body — "
                "a unit finalizes exactly one collection (discard it instead)"
            )
        fn = pin_declared_returns(unit.sema.result(), unit.contract.returns)
        unit.state = _FunctionUnit._FINALIZED
        unit.artifact = FunctionArtifact(fn, unit.contract)
        return unit.artifact

    def discard(self) -> None:
        """Abandon a unit that never collected — the failure-recovery path.

        A recipe whose collection was refused (an invalid derived signature,
        a rejected ABI argument) holds an empty, unusable unit on the
        session's stack; discarding drops it so the recipe may complete (or
        push a corrected one). Loud on a unit that collected — a collected
        body is part of the program's history, and abandoning it would hide
        a half-effect.
        """
        unit = self._unit
        session = unit.session
        session._require_live()
        if unit.state is _FunctionUnit._FINALIZED:
            raise TypeError(f"the function unit {unit.contract.name!r} is already finalized")
        if unit.sema.frame is not None:
            raise TypeError(
                f"discard: the function unit {unit.contract.name!r} collected a body — "
                "a collected unit finalizes; only a never-collected unit discards"
            )
        unit.state = _FunctionUnit._DISCARDED


class ControlContext:
    """The semantic control capabilities — terminators and transfers as
    context operations.

    The escape hatch for tricky user-defined hooks, integrated with the
    framework: every knob routes through the Frame's own machinery and its
    guards (pending result bindings, pending carry groups, region
    acceptance), never through a raw builder. The Python recipe code keeps
    running after a knob — a knob emits IR control flow; it does not
    transfer Python control — so the session enforces the dead-code
    discipline at every further tier effect.

    Target selection: ``emit_return`` addresses the active function (an
    splice-local boundary obeys the splice's own return rules);
    ``emit_yield`` addresses the nearest result-producing region (the
    ``pl.intrin.yield_`` channel); ``emit_break`` / ``emit_continue`` address
    the nearest legal *staged* loop — a recipe cannot transfer Python
    control, so a host-loop target is a loud error instead of a silent
    no-op.
    """

    __slots__ = ("_session",)

    def __init__(self, session: AuthoringSession) -> None:
        self._session = session

    def _frame(self, operation: str) -> tuple[PtoFrame, Any]:
        session = self._session
        session._require_live()
        frame = session._require_borrowed(operation)
        rt = frame.active_runtime()
        if rt is None:
            raise TypeError(
                f"{operation}: the active trace has no interpreter — the control "
                "capabilities address the traced control flow, which exists only "
                "in an interpreter-driven trace"
            )
        return frame, rt

    def emit_return(self, *values: object) -> None:
        """Emit the active function's return at the current position.

        The Frame's own guards apply (pending carry groups, pending result
        bindings, the splice's trailing-return rule). A return emitted
        with no staged control flow open terminates the region: the session
        rejects every further tier effect into it. A return inside a staged
        branch is one alternative return path — exactly as the parser treats
        a ``return`` inside an ``if`` — and terminates nothing.
        """
        frame, rt = self._frame("emit_return")
        session = self._session
        session._check_not_terminated("emit_return")
        frame.return_(rt, list(values) if values else None)
        if not rt.staged_control_flow_active():
            _terminate_frame(frame)
            session._terminated = True

    def emit_yield(self, *operands: object) -> None:
        """Emit a yield to the nearest result-producing region.

        The ``pl.intrin.yield_`` channel: the Frame routes the operands to
        the nearest Construct that accepts them, with its arity and type
        checks."""
        frame, _rt = self._frame("emit_yield")
        self._session._check_not_terminated("emit_yield")
        frame.collect_yield(list(operands))

    def emit_break(self) -> None:
        """Emit a break targeting the nearest legal staged loop.

        The interpreter's transfer selection walks the construct stack; a
        region that lowers to a scope is crossed, exactly as the native
        keyword is. A host-loop target — or no loop open at all — is a loud
        error: the recipe cannot run the native ``break`` on the author's
        behalf."""
        _, rt = self._frame("emit_break")
        self._session._check_not_terminated("emit_break")
        if rt.break_():
            raise TypeError(
                "emit_break: the nearest loop target is not a staged loop — it is "
                "a host loop, or no loop is open at all — a recipe cannot transfer "
                "Python control; the staged loops are the emittable targets"
            )

    def emit_continue(self) -> None:
        """Emit a continue targeting the nearest legal staged loop."""
        _, rt = self._frame("emit_continue")
        self._session._check_not_terminated("emit_continue")
        if rt.continue_():
            raise TypeError(
                "emit_continue: the nearest loop target is not a staged loop — it "
                "is a host loop, or no loop is open at all — a recipe cannot "
                "transfer Python control; the staged loops are the emittable targets"
            )


def _terminate_frame(frame: PtoFrame) -> None:
    """Mark the frame's region terminated (the dead-code discipline).

    The flag rides the Frame — not the session — because the terminated
    region is the caller's: a later recipe invocation over the same trace
    must see it too. A plain attribute (the frame class has no ``__slots__``)
    keeps the authoring runtime out of the tracer's state machine."""
    setattr(frame, "_authoring_terminated", True)  # noqa: B010 - the sanctioned tracer-internal client


def _frame_terminated(frame: PtoFrame) -> bool:
    return getattr(frame, "_authoring_terminated", False)


class AuthoringSession:
    """One recipe invocation's ownership: the caller frame and the unit stack.

    Borrowed mode wraps the active trace's frame; root mode opens the entry
    function unit itself (``RecipeCallable.invoke``). The session is
    one-invocation-scoped: every handle it handed out is dead once the
    recipe returns. This module is the sanctioned client of the tracer's
    internals — recipe code sees only the context views."""

    def __init__(self, frame: PtoFrame | None = None, root: _FunctionUnit | None = None) -> None:
        if (frame is None) == (root is None):
            raise RuntimeError("an authoring session is borrowed (a frame) or root (a unit) — exactly one")
        self._borrowed_frame = frame
        self._root = root
        self._units: list[_FunctionUnit] = []
        # The construct tier's region state (composition in LIFO order) —
        # owned here so the construct and recipe contexts share one session.
        self._regions: list[object] = []
        self._primary: object | None = None
        self._primary_exit: object = None
        self._live = True
        self._terminated = False

    # —— the mode ——

    @property
    def is_root(self) -> bool:
        """Whether this session opened the entry function itself."""
        return self._root is not None

    @property
    def caller_frame(self) -> PtoFrame:
        """The frame caller-side effects land in (the borrowed frame, or the
        root unit's — a root session's only frame)."""
        if self._borrowed_frame is not None:
            return self._borrowed_frame
        assert self._root is not None
        return self._root.frame

    # —— the guards ——

    def _require_live(self) -> None:
        if not self._live:
            raise RuntimeError(
                "the authoring session has ended — a context or unit handle must not "
                "be used past its recipe invocation"
            )

    def _require_borrowed(self, operation: str) -> PtoFrame:
        self._require_live()
        frame = self._borrowed_frame
        if frame is None:
            raise TypeError(
                f"{operation} belongs to a borrowed (in-trace) session — a root session owns only its entry"
            )
        return frame

    def _check_splice_clear(self, operation: str) -> None:
        """Refuse an effect while a splice body is executing (nesting)."""
        frame = self.caller_frame
        if frame.sema.splice_parent is not None:
            raise TypeError(
                f"{operation}: a splice is already active — nested or recursive "
                "splicing is rejected (one splice at a time per trace)"
            )

    def _check_not_terminated(self, operation: str) -> None:
        if self._terminated:
            raise TypeError(
                f"{operation}: the session's region was terminated by an emitted return — "
                "dead code after emit_return is rejected at every authoring-tier effect"
            )
        frame = self._borrowed_frame if self._borrowed_frame is not None else None
        if frame is not None and _frame_terminated(frame):
            raise TypeError(
                f"{operation}: the caller's region was terminated by an emitted return — "
                "dead code after emit_return is rejected at every authoring-tier effect"
            )

    def _unfinalized_units(self) -> list[_FunctionUnit]:
        return [unit for unit in self._units if unit.state is _FunctionUnit._ACTIVE]

    def _validate_complete(self) -> None:
        """The recipe completed: every pushed unit must be finalized (or, for
        a never-collected one, discarded), and no composed region may remain
        open — the recipe's regions close before it returns."""
        self._require_live()
        if self._regions:
            raise TypeError(
                f"the recipe left {type(self._regions[-1]).__name__} region(s) open — every "
                "ctx.enter pairs with a ctx.exit (a with ctx.region(...) closes itself)"
            )
        unfinalized = self._unfinalized_units()
        if unfinalized:
            raise TypeError(
                f"the recipe left function unit(s) unfinalized — "
                f"{', '.join(repr(u.contract.name) for u in unfinalized)}; a recipe "
                "cannot complete while it owns an unfinalized unit"
            )

    # —— the construct tier's region stack ——

    def enter_region(self, cm: object) -> None:
        """Record one composed region (the first entered is the primary
        surface unless a later one is named primary)."""
        self._require_live()
        self._regions.append(cm)
        if self._primary is None:
            self._primary = cm

    def register_scope_region(self, region: object, *, primary: bool) -> None:
        """Record one directly-opened scope region (``open_scope`` / ``open_carrier``).

        ``primary=True`` names the recipe's structural surface explicitly —
        the region the Frame's deferred-scope and carrier probes see."""
        self._require_live()
        self._regions.append(region)
        if self._primary is None or primary:
            self._primary = region

    def exit_region(self, cm: object) -> None:
        """Validate one region close (paired, LIFO) and pop it."""
        self._require_live()
        if not any(region is cm for region in self._regions):
            raise TypeError(
                f"ctx.exit: this {type(cm).__name__} region is not open — exit pairs "
                "with a successful ctx.enter / ctx.open_scope of the same region"
            )
        if self._regions[-1] is not cm:
            raise TypeError(
                "ctx.exit: regions close in reverse open order — "
                f"{type(self._regions[-1]).__name__} is still open"
            )
        self._regions.pop()

    def note_primary_exit(self, result: object) -> None:
        """Record the primary region's close result (the closed scope statement)."""
        self._primary_exit = result

    def close(self) -> None:
        """End the session — every handle it handed out dies here."""
        self._live = False

    # —— the effects ——

    def push_function(self, contract: FunctionContract) -> FunctionContext:
        """Open one child function unit from ``contract`` (borrowed sessions).

        The unit is a child function of the caller's program — a one-shot
        Sema with its own builder — registered with the active program
        assembly when its call is emitted. A root session owns only its
        entry: callees are pushed from *inside* the collected body (where
        the caller frame is active), never from the host-side root recipe.
        """
        self._require_live()
        self._check_not_terminated("push_function")
        if self._root is not None:
            raise TypeError(
                "push_function: a root session owns only its entry — push callees "
                "from inside the collected body, not from the root recipe"
            )
        sema = PtoSema(
            name=contract.name,
            function_type=contract.function_type,
            level=contract.level,
            role=contract.role,
            attrs=contract.attrs,
            strict_ssa=contract.strict_ssa,
            requires_runtime_binding=contract.requires_runtime_binding,
        )
        unit = _FunctionUnit(self, sema, contract)
        self._units.append(unit)
        return FunctionContext(unit)

    def splice_body(
        self,
        body: object,
        *,
        args: Sequence[object] = (),
        kwargs: dict[str, object] | None = None,
        auto_scope: bool | None = None,
    ) -> object:
        """Execute one prepared body at the caller's insertion point.

        The borrowed collection — the splice's semantics: a
        borrowing Sema over the caller's, a fresh lexical frame, no
        function, no declaration, no ``ir.Call``; the body's return value
        crosses back to the recipe. One splice at a time per trace (the
        nesting guard), and the body's own return obeys the splice's
        trailing-return rule.
        """
        frame = self._require_borrowed("splice_body")
        self._check_not_terminated("splice_body")
        self._check_splice_clear("splice_body")
        prepared = _prepared(body)
        for value in (*args, *(kwargs or {}).values()):
            if isinstance(value, Param):
                raise TypeError(
                    "a body argument may not be a Param placeholder — pass the values "
                    "(facades or host values) directly"
                )
        sema = PtoSema._for_splice(frame.sema, owner=prepared.__name__, auto_scope=auto_scope)
        splice_frame = sema._prepare_frame()
        runner = InstrumentedFunction(_live_function(prepared), _prefix(prepared), sema)
        guard = _collecting_guard(prepared, "splice_body")
        token = set_active_frame(splice_frame)
        try:
            return runner(*args, **(kwargs or {}))
        finally:
            reset_active_frame(token)
            _COLLECTING.reset(guard)

    def emit_call(
        self, artifact: FunctionArtifact, *args: object, key: Any = None, **kwargs: object
    ) -> object:
        """Emit the call site of a finalized artifact in the caller's trace.

        Pure caller-side construction: the detached ``ir.Call`` facade is
        returned to the recipe (which returns it to its caller's expression
        position). ``key`` is the optional variant-dedupe key the callee
        compositions use (same body, structically identical argument types
        → one function); the program-assembly registration that consumes it
        arrives with the jit envelope.
        """
        self._require_live()
        self._check_not_terminated("emit_call")
        if not isinstance(artifact, FunctionArtifact):
            raise TypeError(f"emit_call takes a FunctionArtifact, got {artifact!r}")
        self._require_borrowed("emit_call")
        fn = artifact.function
        from pypto.language.op.marker import Marker  # noqa: PLC0415 - the marker/op cycle

        marker = Marker._from_ir_function(fn)
        return marker(*args, **kwargs)

    def materialize_entry(self, artifact: FunctionArtifact) -> FunctionArtifact:
        """Mark the root unit's artifact as the entry — the root placement.

        The semantic step the jit envelope wraps (specialization, caching,
        and dispatch are the frontend's concerns); a root recipe's last
        effect."""
        self._require_live()
        if self._root is None:
            raise TypeError(
                "materialize_entry belongs to a root session — a borrowed recipe returns "
                "its artifact or its call to the caller instead"
            )
        if not isinstance(artifact, FunctionArtifact):
            raise TypeError(f"materialize_entry takes a FunctionArtifact, got {artifact!r}")
        root = self._root
        if root.artifact is None or artifact._function is not root.artifact.function:
            raise TypeError(
                "materialize_entry: the artifact is not this session's root function — "
                "the entry is the root unit's own finalized artifact"
            )
        entry = FunctionArtifact(artifact.function, artifact.contract, entry=True)
        root.artifact = entry
        return entry
