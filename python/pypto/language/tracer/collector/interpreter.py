# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Trace interpreter for the language-agnostic collector protocol.

The trace interpreter owns invocation state, instrumented locals, per-branch
locals mappings, and one ordered stack of active Construct objects — the only
logical representation of active structure. Every ``if``/``for``/``while``
installs exactly one Construct: a host-kind Construct, or a language
Frame-created staged Construct that inherits the matching host-kind class.

The trace interpreter routes every region-local callback to the exact selected
Construct, never asks the Frame to rediscover a target, and interprets only
generic facts: host kinds, shared Construct interfaces, the Frame protocol,
and typed binding/close results. It knows no language type, trace-protocol
object role, or builder operation.

A result-producing close leaves its exact Construct on top of the stack as a
*pending result construct*: ordinary operations reject it before mutation, and
only result bindings (``PartialResultBinding``/``FinalResultBinding``) may
address it. The interpreter removes it when a binding reports
``FinalResultBinding``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, runtime_checkable

from .interface import (
    AwaitingResultBindings,
    BranchDisposition,
    Closed,
    CloseResult,
    FinalResultBinding,
    Frame,
    HostBinding,
    PartialResultBinding,
    Sema,
    StagedBinding,
)

V = TypeVar("V")

_IF = "if"
_FOR = "for"
_WHILE = "while"
_WITH = "with"
_LOOP_KINDS = frozenset({_FOR, _WHILE})
_CF_KINDS = frozenset({_IF, _FOR, _WHILE})


@runtime_checkable
class LoopTransfer(Protocol):
    """The staged loop's native-transfer callbacks.

    A staged loop Construct (a Frame-created ForConstruct/WhileConstruct subclass)
    implements this so ``break``/``continue`` can hand it the exact
    callback; the trace interpreter checks membership at selection time so a
    non-conforming Construct is a loud contract error, not an AttributeError
    mid-dispatch.
    """

    def on_break(self, rt: Interpreter[V]) -> None: ...

    def on_continue(self, rt: Interpreter[V]) -> None: ...


class ZeroOperandYieldAction(Enum):
    """A Construct's typed answer to the zero-operand yield question.

    A zero-operand yield terminator is not a result binding: the innermost
    Construct that answers ACCEPT terminates it; a Construct that answers
    BLOCK accepts result yields, so the zero-operand yield must address that
    exact Construct and its arity check reports the mismatch; CROSS means
    the Construct is outside zero-operand termination and the search may
    cross it.
    """

    ACCEPT = "accept"
    BLOCK = "block"
    CROSS = "cross"


class TransferTargetAction(Enum):
    """A Construct's typed answer to the ``break``/``continue`` target question.

    ACCEPT: a transfer here binds to this Construct (every host and staged
    loop). CROSS: this Construct lowers to a *scope*, not a loop, so the
    keyword binds to the nearest *enclosing* loop — the walk crosses it, and
    a transfer that finds no target above reaches the crossed Construct's
    :meth:`Construct.on_unbound_transfer`.
    """

    ACCEPT = "accept"
    CROSS = "cross"


@dataclass(frozen=True)
class ZeroOperandYieldSelection(Generic[V]):
    """The trace interpreter's ordered zero-operand-yield selection: the exact
    Construct plus its typed answer."""

    construct: Construct[V]
    action: ZeroOperandYieldAction


class InterpreterError(RuntimeError):
    """A trace-interpreter structural-contract violation from an invalid caller."""


@runtime_checkable
class _SupportsWith(Protocol):
    """The host context-manager surface an instrumented ``with`` wraps.

    Structural: the user's manager needs no inheritance, and the trace
    interpreter preserves its ``__exit__`` answer (suppression) verbatim.
    """

    def __enter__(self) -> object: ...

    def __exit__(self, *exc: object) -> object: ...


class Construct(Generic[V]):
    """One active construct.

    The trace interpreter owns the stack position and logical lifetime; this
    base class and the host-kind subclasses own host behavior and public
    routing. Language implementations inherit the matching class and
    implement staged hooks: the extension surface is the public host-kind
    methods themselves (``close`` and kind-specific routing, returning the
    same public contract) plus interface-tag, zero-operand-yield, and
    transfer-target hooks.
    The interpreter still owns installation, removal, order, and execution
    kind. The one strict Template Method is :meth:`WhileConstruct.test`:
    the algorithm and staging flag are base-owned, and a subclass overrides
    only the protected ``_activate`` hook.
    """

    kind: ClassVar[str] = "construct"

    def __init__(self) -> None:
        # Interface tags are opaque to the trace interpreter; language Constructs
        # advertise region-local semantics (yield, completion, split mode)
        # through them.
        self._interface_tags: frozenset[str] = frozenset()

    def is_staged(self) -> bool:
        """Whether this Construct collects staged IR — its install-time
        classification, constant for its lifetime except for WhileConstruct's
        one-way lazy staging decision."""
        return False

    def supports_interface(self, interface_tag: str) -> bool:
        """Whether this Construct advertises ``interface_tag``.

        Tags are opaque to the trace interpreter — a language's Construct
        subclasses declare a closed vocabulary at construction, and `Frame`
        queries it through :meth:`Interpreter.nearest_supporting`.
        """
        return interface_tag in self._interface_tags

    def zero_operand_yield(self) -> ZeroOperandYieldAction:
        """This Construct's answer to a zero-operand yield terminator.

        The base classes are host Constructs: CROSS. A language subclass
        answers from its own region semantics (see
        :meth:`Interpreter.nearest_zero_operand_yield` for the selection contract).
        """
        return ZeroOperandYieldAction.CROSS

    def transfer_target_action(self) -> TransferTargetAction:
        """This Construct's answer to a ``break``/``continue`` target question.

        The base classes are loops: ACCEPT. A language subclass answers CROSS
        for a region that lowers to a scope rather than a ``ForStmt`` (see
        :meth:`Interpreter.break_` for the selection contract).
        """
        return TransferTargetAction.ACCEPT

    def on_unbound_transfer(self, rt: Interpreter[V], keyword: str) -> None:
        """This CROSS Construct's answer to a transfer with no target above it.

        Reached when the walk crossed a CROSS Construct and found no ACCEPT
        target: the transfer has no loop to bind to, and *must* be rejected
        here. The base answer is the interpreter's loud generic rejection; a
        language subclass reports its own user-facing rule (its region shapes
        have no loop).
        """
        del rt
        raise InterpreterError(
            f"'{keyword}' has no target loop: the nearest enclosing region lowers to a scope, not a loop"
        )

    def on_zero_operand_yield(self, rt: Interpreter[V]) -> None:
        """The typed terminator callback for an ACCEPT-selected zero-operand yield
        terminator.

        Base classes answer CROSS and are never selected; reaching this
        default is a selection-contract violation.
        """
        del rt
        raise InterpreterError("this construct does not accept a zero-operand yield terminator")

    def close(self) -> CloseResult:
        """Normal completion of the Construct; the trace interpreter removes it
        unless the result is :class:`AwaitingResultBindings`."""
        return Closed()


class IfConstruct(Construct[V]):
    """A host ``if``: the host truth value selects the branch."""

    kind: ClassVar[str] = _IF

    def __init__(self, guard: bool) -> None:
        super().__init__()
        self._guard = guard

    def then_active(self) -> bool:
        return self._guard

    def enter_else(self) -> None:
        return None

    def else_active(self) -> bool:
        return not self._guard


class ForConstruct(Construct[V]):
    """A host ``for``: the host iterable drives the loop."""

    kind: ClassVar[str] = _FOR

    def __init__(self, iterable: Iterable[object]) -> None:
        super().__init__()
        self._iterable = iterable

    def iterations(self) -> Iterable[object]:
        """The object the host ``for`` iterates."""
        return self._iterable

    def activate_target(self, ctx: Any, name: str) -> Any:
        """Deferred loop-target binding — staged loops only."""
        raise InterpreterError("a host `for` construct has no deferred loop-target binding")


class WhileConstruct(Construct[V]):
    """A ``while`` with a lazy staging decision: starts in host execution,
    may activate staging at most once, and never reverses.

    ``test`` is the trace-interpreter-owned transition algorithm: only the
    language subclass's protected ``_activate`` hook decides activation, the
    base class sets the staging flag only after successful activation, and
    after activation the thunk is never reevaluated (the body traces one
    representative iteration).
    """

    kind: ClassVar[str] = _WHILE

    def __init__(self) -> None:
        super().__init__()
        self._staged = False

    def is_staged(self) -> bool:
        return self._staged

    def test(self, thunk: Callable[[], object]) -> bool:
        if self._staged:
            # The representative body iteration already ran; terminate the host loop.
            return False
        pred = thunk()
        if self._activate(pred):
            self._staged = True
            return True  # trace one representative staged body iteration
        return bool(pred)

    def _activate(self, pred: object) -> bool:
        """Protected staging hook: may this predicate activate the loop?

        The base class never activates. A language subclass performs its
        builder construction here and returns True; it must not mutate the
        staging flag itself.
        """
        del pred
        return False


class WithConstruct(Construct[V]):
    """A ``with`` Construct wrapping the host context-manager protocol.

    The generated code has no trailing ``exit_construct`` for ``with``; the
    generated context manager below is the exit hook. ``enter``/``close``
    preserve the user context manager's entry, exit, exception triple, and
    suppression exactly once. A staged subclass supplies staged hooks behind
    the same public contract and stores its trace-protocol object instead of
    a host manager — hence ``object`` here, narrowed at host accessors.
    """

    kind: ClassVar[str] = _WITH

    def __init__(self, cm: object) -> None:
        super().__init__()
        self._cm = cm
        self._closed = False
        self.exit_result: object = False

    def _host_cm(self) -> _SupportsWith:
        """The host context manager this Construct wraps (host execution only —
        a staged with Construct has none; its overrides never call this)."""
        cm = self._cm
        if not isinstance(cm, _SupportsWith):
            raise InterpreterError("a staged with Construct wraps no host context manager")
        return cm

    def enter(self) -> object:
        return self._host_cm().__enter__()

    def close(self, exception_triple: tuple[object, ...] | None = None) -> CloseResult:
        if self._closed:
            return Closed()
        self._closed = True
        exc = exception_triple if exception_triple is not None else (None, None, None)
        self.exit_result = self._host_cm().__exit__(*exc)
        return Closed()

    def out_of_order_exit(self, exception_triple: tuple[object, ...] | None = None) -> None:
        """The generated context manager exited while an inner Construct is
        still open: the host context manager still receives its exactly-one
        ``__exit__``; stack removal is deferred to interpreter unwinding."""
        self.close(exception_triple)


class _WithAdapter(Generic[V]):
    """The generated code's context manager for an instrumented ``with``.

    Both execution kinds route through the exact Construct: host execution
    preserves the user context manager's protocol (including suppression —
    the adapter returns the manager's own ``__exit__`` answer), and staged
    execution closes the language Construct. The trace interpreter removes
    the Construct from its stack when the adapter exits in LIFO order.
    """

    __slots__ = ("_rt", "_construct")

    def __init__(self, rt: Interpreter[V], construct: WithConstruct[V]) -> None:
        self._rt = rt
        self._construct = construct

    def __enter__(self) -> object:
        return self._construct.enter()

    def __exit__(self, *exc: object) -> object:
        self._rt._exit_with_construct(self._construct, exc)
        return self._construct.exit_result


class _InvocationFrame(Generic[V]):
    """One function invocation: the host locals, the Frame, and the
    interpreter's host-binding facts for it."""

    __slots__ = ("ns", "sema_frame", "host_bound")

    def __init__(self, ns: dict[str, object], sema_frame: Frame[V]):
        self.ns = ns
        self.sema_frame = sema_frame
        # Names whose most recent instrumented bind stored a host value. The
        # Frame reads this fact (typed staged initialization) but never mutates it.
        self.host_bound: set[str] = set()


class _Locals(Generic[V]):
    """Namespace proxy used by rewritten reads and assignments.

    Writes pass through the Frame before reaching the current frame's
    instrumented locals or per-branch mapping. Reads search mappings from
    inner to outer and then the frame locals. A missing name raises the same
    ``UnboundLocalError`` category that native local lookup would raise.
    """

    __slots__ = ("_rt",)

    def __init__(self, rt: Interpreter[V]):
        # Bypass the proxy's ``__setattr__`` to stash the interpreter back-reference.
        object.__setattr__(self, "_rt", rt)

    def __setattr__(self, name: str, value: object) -> None:
        rt = object.__getattribute__(self, "_rt")
        rt._route_bind(name, value)

    def __getattr__(self, name: str) -> object:
        rt = object.__getattribute__(self, "_rt")
        return rt._route_read(name)


class _UnwindGuard(Generic[V]):
    """The unwind scope guard around a transfer-affected construct.

    The source rewriter wraps a construct whose body holds a native
    ``break``/``continue`` bound to an enclosing loop. The keyword's own
    unwind then runs this guard's exit at the construct's exact lexical
    boundary — after inner source context managers and ``finally`` bodies,
    before outer ones — so the active Construct closes in Python's unwind
    order instead of eagerly before the keyword.

    The exit closes every Construct opened above the entry marker — the
    construct's own, plus any inner Construct whose trailing close the
    unwinding keyword (a return or the transfer itself) skipped. A Construct
    removed earlier — the loop-``else`` pre-close or a result binding —
    simply ends the sweep early; a pending result construct stops it because
    the result bindings that follow the guard must finish that region. An
    exception triple does nothing because escaping exceptions skip normal
    finalization.
    """

    __slots__ = ("_rt", "_enter_top")

    def __init__(self, rt: Interpreter[V]) -> None:
        self._rt = rt
        self._enter_top: Construct[V] | None = None

    def __enter__(self) -> None:
        # The enclosing Construct at entry: the Construct itself is created
        # inside the guard (eval_if / begin_for / eval_bool run in the wrapped
        # rows), so the exit sweeps down to this marker.
        self._enter_top = self._rt.current_construct()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            return False  # unwind: no normal finalization
        rt = self._rt
        while True:
            top = rt.current_construct()
            if top is None or top is self._enter_top:
                return False  # the boundary is restored (or was never crossed)
            if top is rt.pending_result():
                # Legal only when the pending result construct is this
                # construct's own result-producing close (or a region left by
                # its else suite): it must sit directly above the entry
                # marker, and the result bindings that follow the guard
                # finish it. An inner construct's pending result beneath
                # this construct's own still-open Construct cannot be
                # crossed — the construct's close would be deferred past
                # its boundary and nothing would close it after the result
                # bindings.
                constructs = rt.active_constructs()
                below = constructs[1] if len(constructs) > 1 else None
                if below is not self._enter_top:
                    raise InterpreterError(
                        f"the {top.kind} boundary cannot close while an inner "
                        "staged result construct awaits result bindings — finish "
                        "the result bindings inside the construct"
                    )
                return False
            rt.exit_construct()


class Interpreter(Generic[V]):
    """Driver for frames, instrumented locals, and the ordered Construct stack."""

    def __init__(self, sema: Sema[V], source_file: str | None = None):
        self._sema = sema
        # The instrumented kernel's original source file (the compiled code
        # object carries it); row-span events pair it with the rewriter's
        # original line/column.
        self._source_file = source_file
        self._frames: list[_InvocationFrame[V]] = []
        self._construct_stack: list[Construct[V]] = []
        self._branch_locals: list[dict[str, object]] = []
        # Each staged if owns its per-branch mapping pair.
        self._if_branch_locals: dict[IfConstruct[V], tuple[dict[str, object], dict[str, object] | None]] = {}
        # While sites keyed by the hoisted predicate thunk's identity (the
        # source rewriter hoists one stable thunk per source loop).
        self._while_sites: dict[object, WhileConstruct[V]] = {}
        # The exact top pending result Construct, if a result-producing close
        # left it awaiting result bindings.
        self._pending_result: Construct[V] | None = None
        # Whether any statement-position event (bind, expression statement,
        # construct head, return) has been collected yet. The prologue query:
        # a body-prologue directive is legal only before the first statement,
        # and the interpreter — which sees every such event, host or staged —
        # is the sole owner of that fact. Comments are annotation state and
        # never flip it.
        self._collected_statement = False

    def collected_statement(self) -> bool:
        """Whether any statement-position event has been collected yet.

        The language-side prologue check (``pl.func_attr``): a prologue
        directive must precede every other statement, matching the host
        language's own "first statement" notion — the interpreter is
        event-aware where the language Frame is deliberately not."""
        return self._collected_statement

    # Frame lifecycle
    def enter_frame(self, /, **params: object) -> None:
        """Create a frame and bind invocation parameters into its host locals.

        Parameters arrive by keyword from the generated prologue. ``self`` is
        positional-only so a kernel parameter named ``self`` remains inside
        ``params`` rather than colliding with the interpreter instance.
        """
        sema_frame = self._sema.begin_frame()
        # Installed once at frame entry so the Frame's named stdlib
        # operations can query the trace interpreter.
        sema_frame.note_interpreter(self)
        ns = dict(params)
        self._frames.append(_InvocationFrame(ns=ns, sema_frame=sema_frame))

    def exit_frame(self) -> None:
        """Pop the invocation frame and finalize the Frame.

        This is the direct/test API: the source rewriter never emits it, and
        the trace driver finalizes via :meth:`close` from the success path.
        Prefer :meth:`close` unless driving the trace interpreter directly.
        """
        frame = self._frames.pop()
        frame.sema_frame.finish()

    def close(self) -> None:
        """Close every open Construct and frame in reverse nesting order.

        Called only on the success path: an exception escaping the
        instrumented function skips this entirely — the one-shot Sema (and
        its builder) is discarded by the caller and the original exception
        propagates unchanged.
        """
        if self._pending_result is not None:
            raise InterpreterError(
                f"result bindings for the staged {self._pending_result.kind} "
                "must complete before trace finalization"
            )
        while self._construct_stack:
            construct = self._construct_stack.pop()
            result = construct.close()
            if isinstance(result, AwaitingResultBindings):
                # The registered pending-result case is the head guard above;
                # this is a Construct whose close is result-producing but was
                # never finished by its result bindings — fail loudly rather
                # than discard the typed result.
                raise InterpreterError(
                    f"the staged {construct.kind} produced results during trace "
                    "finalization — its result bindings must complete before "
                    "the trace ends"
                )
            self._forget_while_site(construct)
        while self._frames:
            self._frames.pop().sema_frame.finish()

    @property
    def vars(self) -> _Locals[V]:
        """The instrumented-locals proxy — ``rt.vars.name`` reads and writes
        route through the Frame and the per-branch locals mappings."""
        return _Locals(self)

    def unwind_guard(self) -> _UnwindGuard[V]:
        """The unwind scope guard for a transfer-affected Construct.

        Generated code wraps the Construct's rows in ``with
        rt.unwind_guard():``; the guard closes the Construct at its exact
        lexical boundary on every non-exception exit (normal fallthrough,
        return, and native ``break``/``continue`` unwind), replacing the
        trailing ``exit_construct()`` that a native transfer would skip.
        """
        return _UnwindGuard(self)

    def _span(self, lineno: int, col: int) -> None:
        """Row-head event: the instrumented row's original source position.

        The rewriter emits one per logical statement; the Sema pins it so
        the op and builder layers attach the original location instead of
        walking the Python stack (the instrumented frame's line numbers are
        shifted emitted text, and the Sema's own frames are framework
        plumbing, not user code).
        """
        self._sema.note_span(self._source_file or "", lineno, col)

    def _leading_comments(self, comments: list[str]) -> None:
        """Forward a source-comment event to the active Frame.

        Deliberately exempt from the pending-result rejection: the
        pending-comment queue is Frame annotation state consumed by the
        next permitted operation (typically the result binding itself),
        not a mutation — a natural comment line between a result-producing
        close and its result binding must not reject.
        """
        self._current_frame().sema_frame.add_leading_comments(comments)

    # Instrumented locals
    def _route_bind(self, name: str, value: object) -> object:
        """Store a value through the Frame and the instrumented locals.

        The Frame's typed result selects the stored representation and the
        host-binding bookkeeping; a result binding that reports
        ``FinalResultBinding`` also removes the exact pending result construct.

        The typed result is also the result-binding enforcement point: the
        trace interpreter cannot reject a binding before the Frame sees the
        value (only the Frame can distinguish a result-binding proxy from an
        ordinary value). It enforces the contract on the result — while a
        pending result construct is the exact top Construct, only result
        bindings addressing that exact Construct are accepted, and a result
        binding with no pending Construct is rejected. The check fires before
        the locals mutation.
        """
        self._note_statement()
        frame = self._current_frame()
        result = frame.sema_frame.bind(self, name, value)
        stored: object
        match result:
            case HostBinding():
                self._reject_pending_result("bind")
                stored = value
                frame.host_bound.add(name)
            case StagedBinding():
                self._reject_pending_result("bind")
                stored = result.value
                frame.host_bound.discard(name)
            case PartialResultBinding() | FinalResultBinding():
                if self._pending_result is None:
                    raise InterpreterError(
                        "a result binding reported progress but no staged result "
                        "construct awaits result bindings"
                    )
                if result.construct is not self._pending_result:
                    raise InterpreterError("the result binding did not address the exact top Construct")
                stored = result.value
                frame.host_bound.discard(name)
                if isinstance(result, FinalResultBinding):
                    self._finish_pending_result(result.construct)
            case _:  # pragma: no cover - the protocol is closed
                raise InterpreterError(f"unknown bind result {result!r}")
        self._write(name, stored)
        return stored

    def _write(self, name: str, value: object) -> None:
        target = self._branch_locals[-1] if self._branch_locals else self._current_frame().ns
        target[name] = value

    def _route_read(self, name: str) -> object:
        for mapping in reversed(self._branch_locals):
            if name in mapping:
                return mapping[name]
        frame = self._current_frame()
        if name in frame.ns:
            return frame.ns[name]
        raise UnboundLocalError(f"local variable '{name}' referenced before assignment")

    def _current_frame(self) -> _InvocationFrame[V]:
        if not self._frames:
            raise RuntimeError(
                "the trace interpreter has no active frame: a nested function "
                "returned from the kernel cannot outlive the invocation "
                "(a nested function escaped its invocation)"
            )
        return self._frames[-1]

    def _frame(self) -> Frame[V]:
        return self._current_frame().sema_frame

    # Rewritten event operations
    def _emit(self, value: object) -> object:
        """Notify the Frame about a call whose host evaluation already occurred.

        This is a collection hook, not a second evaluation of the call. The
        returned value is preserved for expression-compatible rewriting.

        The result-binding check lives in the Frame's ``expression_statement``
        rather than here: the value is opaque to the interpreter, and an
        expression-statement marker (the yield) must reach its own precise
        rejection — only the Frame can distinguish it from an ordinary value.
        """
        self._frame().expression_statement(self, value)
        # An annotation-state marker (a body-prologue directive) collects no
        # IR of its own — the same doctrine as comments: annotation state, not
        # a statement. Every other expression statement counts.
        if not getattr(value, "annotation_state", False):
            self._note_statement()
        return value

    def return_(self, *values: object) -> None:
        """Record a return value and forward it to the Frame.

        The host ``return`` runs only during host execution: the generated
        ``if rt.in_host_mode(): return`` guard suppresses it while a staged
        Construct is open. The ``__return__`` frame-locals entry is diagnostic
        state for the direct/test API; the host ``return`` carries the value
        in production, so nothing in the interpreter or the Frame reads it
        back.
        """
        self._reject_pending_result("return")
        self._note_statement()
        frame = self._current_frame()
        if len(values) == 0:
            frame.ns["__return__"] = None
        elif len(values) == 1:
            frame.ns["__return__"] = values[0]
        else:
            frame.ns["__return__"] = values
        self._frame().return_(self, list(values) if values else None)

    def begin_for(self, iterable: object) -> Iterable[object]:
        """Prepare a ``for`` iterable and install its Construct.

        Host execution receives the original iterable. Staged execution
        receives the Frame's one-element iteration surface. The Construct is
        installed before iteration so even a zero-iteration loop remains
        balanced when the generated exit operation runs. The parameter is
        ``object`` — a staged trace-iterable descriptor is not iterable — but
        the return is what the generated ``for`` iterates in both kinds.
        """
        self._reject_pending_result("begin_for")
        self._note_statement()
        opened = self._frame().create_for_construct(self, iterable)
        if opened is not None:
            self._construct_stack.append(opened.construct)
            return opened.construct.iterations()
        # The Frame's decline is the proof the value is a host iterable (the
        # generated ``for`` iterates the original when nothing is staged).
        construct = ForConstruct[V](cast("Iterable[object]", iterable))
        self._construct_stack.append(construct)
        return construct.iterations()

    def eval_bool(self, thunk: Callable[[], object]) -> bool:
        """Resolve a ``while`` predicate through its lazy-staging Construct.

        The source rewriter hoists one stable thunk per source loop, so thunk
        identity names the site; the Construct's own ``test`` algorithm owns
        the host-to-staged transition and one-representative-iteration
        behavior.
        """
        self._reject_pending_result("eval_bool")
        self._note_statement()
        construct = self._while_sites.get(thunk)
        if construct is None or not self._is_open(construct):
            created = self._frame().create_while_construct(self, thunk)
            construct = created if created is not None else WhileConstruct[V]()
            self._construct_stack.append(construct)
            self._while_sites[thunk] = construct
        return construct.test(thunk)

    def _is_open(self, construct: Construct[V]) -> bool:
        return any(entry is construct for entry in self._construct_stack)

    def _forget_while_site(self, construct: Construct[V]) -> None:
        """Drop a removed while construct's site entry.

        The thunk→construct map tracks *open* sites only: a block re-entry
        re-hoists a fresh thunk, so a dead entry is unreachable state that
        would otherwise accumulate one entry per block execution.
        """
        for thunk, site in list(self._while_sites.items()):
            if site is construct:
                del self._while_sites[thunk]

    def eval_if(self, pred: object) -> None:
        """Evaluate an ``if`` predicate once and select its execution kind.

        Host execution stores truthiness in the Construct guard so generated
        guards skip the inactive branch. Staged execution installs the
        Frame's exact IfConstruct and records each branch in a separate
        per-branch locals mapping so branch-local writes do not overwrite the
        function locals.
        """
        self._reject_pending_result("eval_if")
        self._note_statement()
        construct = self._frame().create_if_construct(self, pred)
        if construct is None:
            construct = IfConstruct[V](bool(pred))
        self._construct_stack.append(construct)
        if construct.is_staged():
            then: dict[str, object] = {}
            self._branch_locals.append(then)
            self._if_branch_locals[construct] = (then, None)

    def eval_then(self) -> bool:
        """The then-guard: host execution uses the Construct's truthiness;
        staged execution returns True for all paths."""
        self._reject_pending_result("eval_then")
        return self._top_if_construct().then_active()

    def eval_else(self) -> None:
        """Switch from the then branch to the else branch.

        The exact Construct's ``enter_else`` runs its staged branch switch;
        the trace interpreter swaps the then mapping for a fresh else mapping
        so the two staged paths stay independent until commit. The pending
        result guard must fire before ``enter_else``: the pending construct
        itself is an IfConstruct at the stack top, and without the guard the
        branch switch would mutate the staged Construct.
        """
        self._reject_pending_result("eval_else")
        top = self._top_if_construct()
        top.enter_else()
        if top.is_staged():
            pair = self._if_branch_locals.get(top)
            stashed = self._branch_locals[-1] if self._branch_locals else None
            if pair is None or pair[0] is not stashed:
                raise InterpreterError("the staged if's then mapping must be the top of the mapping chain")
            self._branch_locals.pop()
            else_mapping: dict[str, object] = {}
            self._branch_locals.append(else_mapping)
            self._if_branch_locals[top] = (pair[0], else_mapping)

    def eval_else_active(self) -> bool:
        """The else-guard: host execution uses the negated Construct guard;
        staged execution returns True for all paths."""
        self._reject_pending_result("eval_else_active")
        return self._top_if_construct().else_active()

    def _top_if_construct(self) -> IfConstruct[V]:
        """The top Construct, proven by its class to be the open ``if``."""
        if not self._construct_stack:
            raise InterpreterError("no open if construct")
        top = self._construct_stack[-1]
        if not isinstance(top, IfConstruct):
            raise InterpreterError(f"expected an open if construct, found {top.kind}")
        return top

    def _merge_branch_locals(self, then: dict[str, object], else_: dict[str, object] | None) -> None:
        """Commit per-branch locals into the enclosing locals mapping.

        A nested staged if closes while an enclosing staged branch is still
        open: the merge must stay branch-local (the innermost active mapping
        — exactly where a plain binding at that point would land), or the
        outer else path would observe values bound only on the outer then
        path.

        The merge rejects a staged-ness disagreement between branches: a
        name staged on one path and host on the other has no sound value
        after the join (the host side would either inline a literal or
        become a use-before-def). Rule A catches the same flip when both
        binds are visible in one chain; this catches the cross-branch case
        where the then mapping was swapped out at ``enter_else``.

        The merged values themselves are otherwise unchecked: a name bound
        on only some paths has no value on the missing path, and that
        unsoundness is rejected by the IR verifier's use-after-def check at
        compile time, not here.
        """
        if else_ is not None:
            staged_query = self._frame().is_staged
            for name, else_value in else_.items():
                then_value = then.get(name)
                if then_value is not None and staged_query(then_value) != staged_query(else_value):
                    raise TypeError(
                        f"typed staged initialization: {name!r} is staged on one "
                        f"branch of this if and a host value on the other — a "
                        f"name read after the join must be staged on every path. "
                        f"Bind a typed constant instead: {name} = pl.const(..., dtype)"
                    )
        branch_locals = [then] if else_ is None else [then, else_]
        union: dict[str, object] = {}
        for mapping in branch_locals:
            union.update(mapping)
        if not union:
            return
        target = self._branch_locals[-1] if self._branch_locals else self._current_frame().ns
        target.update(union)

    def eval_and(self, a: object, thunk: Callable[[], object]) -> object:
        """Evaluate ``and`` with a lazy right-hand thunk.

        Rejected while a pending result construct is on top before the thunk
        runs: no legal result binding can consume the result, so evaluating
        the right-hand side during the result-binding phase is rejected up
        front.
        """
        self._reject_pending_result("and")
        if self._frame().is_staged(a):
            return self._frame().logical_and(a, thunk())
        return a if not a else thunk()

    def eval_or(self, a: object, thunk: Callable[[], object]) -> object:
        """Evaluate ``or`` with a lazy right-hand thunk (as ``eval_and``)."""
        self._reject_pending_result("or")
        if self._frame().is_staged(a):
            return self._frame().logical_or(a, thunk())
        return a if a else thunk()

    def eval_not(self, a: object) -> object:
        """Evaluate host negation or the Frame's staged negation."""
        self._reject_pending_result("not")
        if self._frame().is_staged(a):
            return self._frame().logical_not(a)
        return not a

    def on_begin_with(
        self,
        ctxobj: object,
        name: str | None = None,
        single_call_body: bool = False,
        reads_block_idx: bool = False,
    ) -> _WithAdapter[V]:
        """Prepare a context manager for host or staged execution.

        Both execution kinds install one WithConstruct and return a generated
        context manager: host execution wraps the user's context manager
        (preserving its protocol exactly), and staged execution follows the
        same public host contract through staged hooks. ``name`` identifies a
        single ``as`` target. ``single_call_body`` and ``reads_block_idx`` are
        the rewriter's static facts about the with-suite (exactly one
        call-valued statement; a ``get_block_idx()`` call anywhere in the
        subtree) — consumed by constructs whose IR shape depends on the body
        (``pl.spmd``'s direct-dispatch vs InCore-wrapped form).
        """
        self._reject_pending_result("on_begin_with")
        self._note_statement()
        opened = self._frame().create_with_construct(
            self, ctxobj, name, single_call_body=single_call_body, reads_block_idx=reads_block_idx
        )
        if opened is not None:
            self._construct_stack.append(opened.construct)
            return _WithAdapter(self, opened.construct)
        # The Frame declined: the value is a host context manager. The
        # decline is the proof that it is not a trace context manager, so the structural
        # ``_SupportsWith`` check is the loud form of the remaining claim —
        # a non-manager would otherwise surface as an AttributeError at
        # ``__enter__`` instead of a contract error at the with entry.
        if not isinstance(ctxobj, _SupportsWith):
            raise InterpreterError(
                f"with {ctxobj!r}: the value is neither a staged trace context manager nor a "
                "context manager (no __enter__/__exit__ surface)"
            )
        construct = WithConstruct[V](ctxobj)
        self._construct_stack.append(construct)
        return _WithAdapter(self, construct)

    def _exit_with_construct(self, construct: WithConstruct[V], exc: tuple[object, ...] | None) -> None:
        """Close a ``with`` Construct when its generated context manager exits.

        In LIFO order the Construct closes now and leaves the stack. When an
        inner Construct's generated exit was skipped (a raise or a host
        return), the Construct cannot close out of order: a host Construct
        still owes its context manager exactly one ``__exit__`` now
        (``out_of_order_exit``), while a staged Construct defers its builder
        close to :meth:`close` on the success path or is discarded with the
        one-shot Sema on the exception path.

        An exception unwinding *through* a staged with skips its staged close
        entirely (no staged finalization during exceptional exit): the
        Construct stays on the stack for the success-path :meth:`close` — a
        suppressing outer context manager owns the IR consequences of the
        resumed trace — or is discarded with the Sema when the exception
        escapes. A host with still receives its exception triple and
        suppression exactly once.

        A normal exit cannot cross an inner pending result construct: a
        result-producing region opened inside the with and not yet finished
        by its result bindings sits above this Construct, and closing the
        with now would collect everything after it inside the still-open
        region. The result bindings must finish inside the with; an
        unwinding exit takes the discard path instead — the pending result
        construct dies with the one-shot Sema.
        """
        if self._construct_stack and self._construct_stack[-1] is construct:
            unwinding = exc is not None and any(e is not None for e in exc)
            if unwinding and construct.is_staged():
                return
            construct.close(exc)
            self._construct_stack.pop()
        else:
            unwinding = exc is not None and any(e is not None for e in exc)
            if not unwinding and self._pending_result is not None:
                raise InterpreterError(
                    f"the with cannot close while the staged "
                    f"{self._pending_result.kind} awaits its result "
                    "bindings — finish the result bindings inside the with"
                )
            construct.out_of_order_exit(exc)

    # —— transfers ——
    def break_(self) -> bool:
        """Handle ``break``; return whether the native ``break`` should run.

        The nearest loop — host or staged — is the target; the trace
        interpreter never skips it for a more convenient outer loop. The one
        crossing is a Construct that declares itself no transfer target at all
        (:meth:`Construct.transfer_target_action`): a region that lowers to a
        scope has no loop for the keyword to bind to, so the walk looks
        outward. A staged target receives the exact callback and suppresses
        the native keyword; a host target permits the native keyword only when
        no staged Construct intervenes, because the alternate staged path was
        not collected.
        """
        self._reject_pending_result("break")
        return self._transfer("break")

    def continue_(self) -> bool:
        """Handle ``continue``; as :meth:`break_` with the same contract."""
        self._reject_pending_result("continue")
        return self._transfer("continue")

    def _transfer(self, keyword: str) -> bool:
        target: Construct[V] | None = None
        crossed: Construct[V] | None = None
        staged_seen = False
        for construct in reversed(self._construct_stack):
            if construct.kind in _LOOP_KINDS:
                if construct.transfer_target_action() is TransferTargetAction.CROSS:
                    # A region that lowers to a scope, not a loop: the keyword
                    # binds to the nearest *enclosing* loop, or to nothing.
                    crossed = crossed if crossed is not None else construct
                    staged_seen = staged_seen or construct.is_staged()
                    continue
                target = construct
                break
            staged_seen = staged_seen or construct.is_staged()
        if target is None:
            if crossed is not None:
                # No enclosing loop: the crossed Construct owns the rejection
                # (its region shapes have no loop to bind to). A hook that
                # returns is a contract violation — never a silent native
                # keyword.
                crossed.on_unbound_transfer(self, keyword)
                raise InterpreterError(
                    f"the crossed {crossed.kind} Construct did not reject a targetless '{keyword}'"
                )
            return True  # no loop construct: the host keyword is harmless
        if target.is_staged():
            if not isinstance(target, LoopTransfer):
                raise InterpreterError(
                    f"the staged {target.kind} Construct does not implement the "
                    "transfer callbacks (on_break/on_continue)"
                )
            if keyword == "break":
                target.on_break(self)
            else:
                target.on_continue(self)
            return False
        if staged_seen:
            raise InterpreterError(
                f"native {keyword} targets a host loop across an open staged "
                "Construct — the alternate staged path was not collected; "
                "restructure the loop or make the target staged"
            )
        return True

    def exit_construct(self, else_follows: bool = False) -> None:
        """Pop and close the top Construct, committing staged branches.

        A result-producing close leaves the exact Construct on the stack as
        a pending result construct instead of removing it; its per-branch
        mappings, if any, are dropped rather than committed — only result
        bindings may export an explicit region's results.

        ``else_follows`` is the source rewriter's static ``has_else`` fact:
        the close runs from inside the loop's ``else`` suite. A staged loop
        rejects the close there — its host loop exits through the predicate
        after one representative iteration, so the else would be collected
        unconditionally rather than skipped by ``break``; a host loop keeps
        its native else.
        """
        if not self._construct_stack:
            return
        top = self._construct_stack[-1]
        if top is self._pending_result:
            raise InterpreterError(
                "exit_construct cannot close a pending result construct — "
                "result bindings must finish it first"
            )
        staged = top.is_staged()
        if else_follows and staged:
            raise InterpreterError(
                f"a staged {top.kind} loop cannot have an 'else' suite — the "
                "alternate break path is not collected; restructure "
                "the loop or keep the predicate a host value"
            )
        self._construct_stack.pop()
        result = top.close()
        if isinstance(result, AwaitingResultBindings):
            self._construct_stack.append(top)
            self._pending_result = top
        else:
            self._forget_while_site(top)
        if isinstance(top, IfConstruct) and top.is_staged():
            pair = self._if_branch_locals.pop(top, None)
            if pair is not None:
                active = self._branch_locals.pop()
                if pair[1] is None:
                    then, else_ = active, None
                else:
                    then_mapping = pair[0]
                    then, else_ = then_mapping, active
                if isinstance(result, Closed) and result.branches is BranchDisposition.COMMIT:
                    self._merge_branch_locals(then, else_)

    def in_host_mode(self) -> bool:
        """Return whether no staged Construct is active."""
        return not any(construct.is_staged() for construct in self._construct_stack)

    def bind(self, name: str, value: object) -> object:
        """Bind a walrus target once and return the stored representation."""
        return self._route_bind(name, value)

    def _note_statement(self) -> None:
        """Flip the prologue fact: a statement-position event was collected."""
        self._collected_statement = True

    # —— result binding phase ——
    def _reject_pending_result(self, op: str) -> None:
        if self._pending_result is not None:
            raise InterpreterError(
                f"{op}: result bindings for the staged "
                f"{self._pending_result.kind} must complete before another "
                "operation is traced"
            )

    def _finish_pending_result(self, construct: Construct[V]) -> None:
        if self._pending_result is not construct:
            raise InterpreterError(
                "the final result binding must address the innermost pending result construct"
            )
        if not self._construct_stack or self._construct_stack[-1] is not construct:
            raise InterpreterError("the completed construct is not the innermost active construct")
        self._construct_stack.pop()
        self._pending_result = None
        self._forget_while_site(construct)

    # —— trusted queries ——
    # Frame and Construct receive the trace interpreter and may query active structure
    # (the contracts are on the methods below).

    def current_construct(self) -> Construct[V] | None:
        """The innermost open Construct (the exact object)."""
        return self._construct_stack[-1] if self._construct_stack else None

    def active_constructs(self, innermost_first: bool = True) -> tuple[Construct[V], ...]:
        """A snapshot of the open Constructs, innermost first by default.

        A copy, never a live view of the stack; callers must not cache the
        order across interpreter mutations.
        """
        constructs = tuple(self._construct_stack)
        return constructs[::-1] if innermost_first else constructs

    def nearest_kind(self, *kinds: str) -> Construct[V] | None:
        """The innermost open Construct of one of ``kinds``, or ``None``.

        Never skips a nearer construct of another kind to reach an outer one.
        """
        for construct in reversed(self._construct_stack):
            if construct.kind in kinds:
                return construct
        return None

    def nearest_supporting(self, interface_tag: str) -> Construct[V] | None:
        """The innermost open Construct advertising ``interface_tag``.

        Tags are opaque to the trace interpreter; language Construct
        subclasses declare the vocabulary (see
        :meth:`Construct.supports_interface`).
        """
        for construct in reversed(self._construct_stack):
            if construct.supports_interface(interface_tag):
                return construct
        return None

    def nearest_zero_operand_yield(self) -> ZeroOperandYieldSelection[V] | None:
        """The innermost Construct that answers a zero-operand yield (ACCEPT or BLOCK).

        The trace interpreter owns ordered target selection: the walk visits
        active Constructs innermost-first, may cross CROSS Constructs, and
        never skips a nearer eligible target. The Construct owns language
        eligibility through its typed ``zero_operand_yield`` answer; the caller
        dispatches on the selection's exact Construct and action.
        """
        for construct in reversed(self._construct_stack):
            action = construct.zero_operand_yield()
            if action is not ZeroOperandYieldAction.CROSS:
                return ZeroOperandYieldSelection(construct, action)
        return None

    def staged_control_flow_active(self) -> bool:
        """Whether staged control flow (if/for/while) is active — the
        typed-staged-initialization join fact."""
        return any(
            construct.is_staged() and construct.kind in _CF_KINDS for construct in self._construct_stack
        )

    def staged_control_flow_above(self, owner: object) -> bool:
        """Whether a staged if/for/while is active above the construct owning ``owner``.

        ``owner`` is matched by identity against each active construct's
        ``_descriptor`` (the trace-protocol object whose region the construct
        wraps) — a generic attribute read, so the collector stays frontend-
        agnostic. Answers "is this point inside a nested control-flow *body*
        of the region that owns ``owner``" — the scope-level declaration
        position question (a declaration is legal at the owner's body top
        level, where only the owning construct intervenes). Conservative when
        no active construct owns ``owner``: a position that cannot be located
        is rejected rather than guessed at."""
        found_owner = False
        for construct in reversed(self._construct_stack):
            if getattr(construct, "_descriptor", None) is owner:
                found_owner = True
                break
            if construct.is_staged() and construct.kind in _CF_KINDS:
                return True
        return not found_owner

    def pending_result(self) -> Construct[V] | None:
        """The exact top pending result Construct, if result bindings await it."""
        return self._pending_result

    def current_binding(self, name: str) -> tuple[bool, object | None]:
        """The current host-binding facts for ``name``.

        Returns ``(was_host_bind, value)``: whether the name's most recent
        instrumented binding stored a host value, and the currently visible
        value of the instrumented local (or ``None`` when unbound). Per-branch
        mappings take precedence, matching instrumented reads.
        """
        frame = self._current_frame()
        for mapping in reversed(self._branch_locals):
            if name in mapping:
                return (name in frame.host_bound, mapping[name])
        if name in frame.ns:
            return (name in frame.host_bound, frame.ns[name])
        return (False, None)

    def host_bound_names(self) -> frozenset[str]:
        """A snapshot of the current frame's host-bound names."""
        return frozenset(self._current_frame().host_bound)

    def restore_host_bound(self, names: Iterable[str]) -> None:
        """Restore a host-binding snapshot (explicit-if branch reset)."""
        frame = self._current_frame()
        frame.host_bound = set(names)


__all__ = [
    "ForConstruct",
    "IfConstruct",
    "LoopTransfer",
    "Interpreter",
    "Construct",
    "InterpreterError",
    "ZeroOperandYieldAction",
    "ZeroOperandYieldSelection",
    "TransferTargetAction",
    "WhileConstruct",
    "WithConstruct",
]
