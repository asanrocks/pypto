# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PtoConstructs — the language Construct subclasses.

Each class inherits the matching trace-interpreter host-kind Construct and
implements only the staged hooks behind its public routing: builder region
construction, yield tracking, transfers, and result binding. The trace
interpreter owns stack order, per-branch locals mappings, and execution-kind
transitions; these Constructs own staged region data — explicit-if local
mappings, deferred `for` region construction, lazy `while` staging decision,
trace-protocol-object lifecycle, and pending result-binding state.

A result-producing close returns :class:`AwaitingResultBindings` and the
Construct stays on the interpreter stack as the exact pending result
construct until result bindings finish it (:meth:`complete` performs the
deferred region finalization; the interpreter then removes it).

Interface tags are the closed vocabulary `Frame` queries through the
interpreter's ``nearest_supporting``:

- ``"yield"`` — the region accepts ``pl.intrin.yield_`` operands: every
  PtoIfConstruct and loops with declared `iter_args`. Queried by
  yield.
- ``"split_aiv"`` — the loop recorded a ``pl.split_aiv`` region mode at
  target activation. Queried by the mode lookup.

Pending `iter_args` adjacency is Frame-enforced: the group must be consumed
by the immediately following loop opener, so no separate interface tag is
needed for it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pypto.language.diagnostics import InvalidOperationError
from pypto.language.facade import Facade
from pypto.language.protocol import (
    TraceContext,
    _ForDescriptor,
    _LoopDescriptor,
    _RuntimeDeferredDeclarableScope,
    _SplitAivDescriptor,
    _WithDescriptor,
    is_staged,
)
from pypto.language.tracer.collector.interface import AwaitingResultBindings, BranchDisposition, Closed
from pypto.language.tracer.collector.interpreter import (
    ForConstruct,
    IfConstruct,
    WhileConstruct,
    WithConstruct,
)
from pypto.language.tracer.collector.interpreter import ZeroOperandYieldAction as _ZeroOperandYieldAction
from pypto.language.tracer.semantics import _predicate_expr, _unwrap_facade
from pypto.pypto_core import ir

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypto.language.tracer.collector.interface import CloseResult
    from pypto.language.tracer.collector.interpreter import Interpreter
    from pypto.language.tracer.semantics import PtoFrame

__all__ = ["PtoForConstruct", "PtoIfConstruct", "PtoWhileConstruct", "PtoWithConstruct"]

_W = TypeVar("_W")

_UNKNOWN = ir.Span.unknown()


def _yield_operand_types(operands: list[object]) -> list[ir.Type]:
    """Unwrap yield operands to their IR types (loud on a non-staged operand)."""
    types: list[ir.Type] = []
    for op in operands:
        if not is_staged(op):
            raise TypeError(f"pl.intrin.yield_ operand {op!r} is not a staged value")
        types.append(_unwrap_facade(op).type)
    return types


def _validate_yield_operand_types(operands: list[object], expected: list[ir.Type] | None) -> None:
    """Check a repeat yield matches the recorded arity/types (loud on drift)."""
    for i, op in enumerate(operands):
        if not is_staged(op):
            raise TypeError(f"pl.intrin.yield_ operand {i} not a staged value")
        expr = _unwrap_facade(op)
        if expected is not None and not ir.structural_equal(expr.type, expected[i], enable_auto_mapping=True):
            raise TypeError(f"pl.intrin.yield_ type mismatch at position {i}: {expr.type} vs {expected[i]}")


class _YieldingConstruct(Generic[_W]):
    """Yield and result-binding state shared by result-producing active constructs.

    The ``"yield"`` interface tag is advertised in ``_interface_tags`` only
    while the region accepts yield terminators (a loop with `iter_args` or any if);
    `Frame` asks the trace interpreter for the nearest Construct with that
    tag and dispatches ``on_yield`` to the exact object.
    """

    kind: ClassVar[str]  # supplied by the host-kind base class

    # Whether a repeated yield must match the recorded operand types. A loop
    # channel is one var written every iteration — drift is a silent
    # miscompile, so the check is strict. An if's yields are one producer per
    # branch, merged by ConvertToSSA; the parser records the first branch's
    # type (``setdefault``) and leaves cross-branch divergence to the IR
    # verifier, so the if opts out and only the arity check remains.
    _strict_repeat_yield_types: ClassVar[bool] = True

    _frame: PtoFrame  # supplied by the staged region subclass

    _yield_arity: int | None
    _yield_types: list[ir.Type] | None
    _yield_count: int
    _completed: bool

    def _init_yield_state(self) -> None:
        self._yield_arity = None
        self._yield_types = None
        self._yield_count = 0
        self._completed = False

    @property
    def completed(self) -> bool:
        """Whether the final result binding already closed this region."""
        return self._completed

    # —— hooks each region supplies ——
    @property
    def result_arity(self) -> int | None:
        raise NotImplementedError

    def zero_operand_yield(self) -> _ZeroOperandYieldAction:
        """The shared yield-region answer: ACCEPT while the region takes no
        results (a zero-operand yield terminates it), BLOCK once it produces
        results (the yield must address this exact Construct)."""
        return (
            _ZeroOperandYieldAction.BLOCK if self._yield_arity is not None else _ZeroOperandYieldAction.ACCEPT
        )

    def on_zero_operand_yield(self, rt: Interpreter[_W]) -> None:
        """The typed terminator callback for an ACCEPT-selected zero-operand
        yield; emit the terminating backedge or branch at the current position."""
        del rt
        self._frame.sema._builder.emit(ir.YieldStmt([], ir.Span.unknown()))

    def _emit_return_vars(self, vars_: list[ir.Var]) -> None:
        raise NotImplementedError

    def _end_region(self) -> None:
        raise NotImplementedError

    # —— the shared yield/completion behavior ——
    def on_yield(self, rt: Interpreter[_W], operands: list[object]) -> None:
        del rt
        if self._yield_arity is None:
            self._yield_arity = len(operands)
            self._yield_types = _yield_operand_types(operands)
            self._validate_first_yield_arity()
        elif len(operands) != self._yield_arity:
            raise TypeError(
                f"pl.intrin.yield_ arity mismatch: expected {self._yield_arity}, got {len(operands)}"
            )
        else:
            expected = self._yield_types if self._strict_repeat_yield_types else None
            _validate_yield_operand_types(operands, expected)
        self._yield_count += 1
        self._frame_yield_emit([_unwrap_facade(op) for op in operands])

    def _frame_yield_emit(self, exprs: list[ir.Expr]) -> None:
        raise NotImplementedError

    def _validate_first_yield_arity(self) -> None:
        return None

    def create_leak_var(self, name: str, index: int) -> ir.Var:
        exp_type = self._leak_var_type(index)
        if exp_type is None:
            raise TypeError(
                f"pl.intrin.leak: cannot determine type for position {index} (region={self.kind})"
            )
        return ir.Var(name, exp_type, _UNKNOWN)

    def complete(self, slots: list[ir.Var]) -> None:
        """The final result binding: perform the deferred region finalization."""
        if self._completed:
            raise RuntimeError(
                "pl.intrin.leak: the final result binding already completed this "
                "construct; every result position binds exactly once"
            )
        self._emit_return_vars(slots)
        self._end_region()
        self._completed = True

    def _leak_var_type(self, index: int) -> ir.Type | None:
        if self._yield_types is not None and index < len(self._yield_types):
            return self._yield_types[index]
        return None


class PtoIfConstruct(_YieldingConstruct[Facade], IfConstruct[Facade]):
    """A staged ``if``: owns builder state and explicit-result local mappings.
    Else and close target this exact Construct."""

    # Branch yields are one producer per branch; the phi's type agreement is
    # the IR verifier's report, not a collection-time rejection (parser parity:
    # its ``setdefault`` keeps the first branch's type for the result var).
    _strict_repeat_yield_types = False

    def __init__(
        self,
        frame: PtoFrame,
        pred: ir.Expr,
        local_snapshot: dict[str, ir.Var],
        host_bound_snapshot: frozenset[str],
        comments_pending: bool,
    ) -> None:
        super().__init__(guard=True)
        self._frame = frame
        self._local_snapshot = local_snapshot
        self._host_bound_snapshot = host_bound_snapshot
        self._comments_pending = comments_pending
        self._interface_tags = frozenset({"yield"})
        self._init_yield_state()
        frame.sema._builder.begin_if(pred)

    # —— staged hooks behind the public routing ——
    def is_staged(self) -> bool:
        return True

    def then_active(self) -> bool:
        return True

    def else_active(self) -> bool:
        return True

    def enter_else(self) -> None:
        self._frame.sema._builder.begin_else()
        self._restore_explicit_locals()

    def close(self) -> CloseResult:
        if self._yield_arity is not None:
            # A result-producing if: region finalization waits for the result
            # bindings; per-branch mappings are discarded (only result bindings export).
            self._restore_explicit_locals()
            return AwaitingResultBindings(BranchDisposition.DISCARD)
        self._frame.sema._builder.end_if()
        self._frame._finish_construct_comments(self._comments_pending)
        self._comments_pending = False
        return Closed(BranchDisposition.COMMIT)

    def _restore_explicit_locals(self) -> None:
        """Drop per-branch bindings for an explicit result-producing if."""
        if self._yield_arity is None:
            return
        self._frame._locals = dict(self._local_snapshot)
        rt = self._frame.active_runtime()
        if rt is not None:
            rt.restore_host_bound(self._host_bound_snapshot)

    # —— yield/completion ——
    @property
    def result_arity(self) -> int | None:
        return self._yield_arity

    def _emit_return_vars(self, vars_: list[ir.Var]) -> None:
        builder = self._frame.sema._builder
        for v in vars_:
            builder.add_if_return_var(v)

    def _end_region(self) -> None:
        self._frame.sema._builder.end_if(_UNKNOWN)
        self._frame._finish_construct_comments(self._comments_pending)
        self._comments_pending = False

    def _frame_yield_emit(self, exprs: list[ir.Expr]) -> None:
        self._frame.sema._builder.emit(ir.YieldStmt(exprs, _UNKNOWN))


class PtoForConstruct(_YieldingConstruct[Facade], ForConstruct[Facade]):
    """A staged ``for`` over a trace iterable: its descriptor is construction
    input; this Construct owns the deferred loop-target binding and explicit
    loop-carried state."""

    def __init__(
        self,
        frame: PtoFrame,
        descriptor: _ForDescriptor,
        iter_args: list[ir.IterArg] | None,
        comments_pending: bool,
    ) -> None:
        # The staged Construct has no host iterable — the descriptor is
        # construction input (``begin`` at target activation), never iterated;
        # the ``iterations()`` override below supplies the staged surface.
        super().__init__(())
        self._frame = frame
        self._descriptor = descriptor
        self._iter_args: list[ir.IterArg] = list(iter_args) if iter_args else []
        self._explicit = bool(self._iter_args)
        self._comments_pending = comments_pending
        self._binding: Any = None
        self._split_mode: ir.SplitMode | None = None
        self._interface_tags = frozenset({"yield"} if self._explicit else frozenset())
        self._init_yield_state()

    def is_staged(self) -> bool:
        return True

    @property
    def split_mode(self) -> ir.SplitMode | None:
        """The ``pl.split_aiv`` region mode recorded at target activation
        (None for ordinary trace iterables)."""
        return self._split_mode

    def attach_binding(self, binding: object) -> None:
        self._binding = binding

    def iterations(self) -> Iterable[object]:
        """The staged iteration surface: expose one loop-target binding proxy."""
        return [self._binding]

    # —— deferred region construction ——
    def activate_target(self, ctx: TraceContext, name: str) -> Any:
        """The first target binding activates this exact Construct: the trace
        iterable owns the ``for_loop`` construction, and the Construct records
        its dynamic open state (iteration arguments and split mode). The
        representative iteration exposes one binding proxy, so this fires
        exactly once per install and the iterable's ``__trace_exit__`` at close
        is always paired with a ``begin``."""
        # Parser parity (its Arm 1): a bare top-level ``pl.split_aiv`` emits in
        # place not only inside an open InCore scope but also in an InCore-typed
        # function — the function body *is* the InCore region. Only the plain
        # Opaque/Orchestration form synthesizes the wrapper; this is also what
        # keeps print -> parse a fixed point for the outlined ``*_incore_0``
        # shape, whose regions print bare at the top of the body.
        inside_incore = ctx.is_inside_incore() or ctx.function_type() == ir.FunctionType.InCore
        loop_var = self._descriptor.begin(
            ctx.builder, name, inside_incore, sticky_dump_vars=ctx.sticky_dump_vars()
        )
        # A for-descriptor whose scope began deferred (the spmd for-form's
        # InCore carrier) registers here — its region opens at the first
        # target binding, a point the with-form's Frame-side registration
        # never sees. The pending flag is a plain attribute (the sanctioned
        # open-world getattr pattern).
        if isinstance(self._descriptor, _RuntimeDeferredDeclarableScope) and getattr(
            self._descriptor, "_carrier_pending", False
        ):
            ctx.defer_scope(self._descriptor)
        for ia in self._iter_args:
            ctx.builder.add_iter_arg(ia)
        split_mode = (
            self._descriptor.split_aiv_mode if isinstance(self._descriptor, _SplitAivDescriptor) else None
        )
        if split_mode is not None:
            self._split_mode = split_mode
            self._interface_tags = self._interface_tags | {"split_aiv"}
        return loop_var

    # —— transfers ——
    def on_break(self, rt: Interpreter[Facade]) -> None:
        del rt
        self._validate_transfer("break")

    def on_continue(self, rt: Interpreter[Facade]) -> None:
        del rt
        self._validate_transfer("continue")

    def zero_operand_yield(self) -> _ZeroOperandYieldAction:
        """A staged for's answer: a loop with explicit `iter_args` takes
        result yields (BLOCK — the zero-operand form reports the arity
        mismatch on this exact Construct); a loop with inferred
        loop-carried values has no yield channel (CROSS — selection crosses it)."""
        return _ZeroOperandYieldAction.BLOCK if self._explicit else _ZeroOperandYieldAction.CROSS

    def _validate_transfer(self, keyword: str) -> None:
        # Sequential-only transfer: parallel/unroll loops reject break and
        # continue (matching the parser's rule). A trace iterable without the
        # ``kind`` surface is a sequential range.
        kind = self._descriptor.kind if isinstance(self._descriptor, _LoopDescriptor) else "range"
        kind = kind or "range"
        if kind == "parallel":
            raise InvalidOperationError(
                f"'{keyword}' not supported in parallel loops",
                hint=f"'{keyword}' can only be used in sequential (pl.range) or while loops",
            )
        if kind == "unroll":
            raise InvalidOperationError(
                f"'{keyword}' not supported in unrolled loops",
                hint=f"'{keyword}' can only be used in sequential (pl.range) or while loops",
            )
        builder = self._frame.sema._builder
        if keyword == "break":
            builder.break_stmt()
        else:
            builder.continue_stmt()

    # —— close / completion ——
    def close(self) -> CloseResult:
        if self._explicit and not self._completed:
            # Result-producing: region finalization waits for the result bindings.
            return AwaitingResultBindings(BranchDisposition.NOT_APPLICABLE)
        self._close_descriptor()
        return Closed()

    def _close_descriptor(self) -> Any:
        closed_stmt = self._descriptor.__trace_exit__(self._frame.trace_context())
        # A deferred scope that never flushed (an empty body) or already
        # opened retires here — the Frame's pending list must not outlive
        # the construct.
        self._frame._retire_deferred_scope(self._descriptor)
        self._frame._finish_construct_comments(self._comments_pending)
        self._comments_pending = False
        return closed_stmt

    @property
    def result_arity(self) -> int | None:
        return len(self._iter_args) if self._iter_args else None

    def _validate_first_yield_arity(self) -> None:
        assert self._yield_arity is not None
        if self._yield_arity != len(self._iter_args):
            raise TypeError(
                f"pl.intrin.yield_ arity {self._yield_arity} mismatches for iter_args {len(self._iter_args)}"
            )

    def _leak_var_type(self, index: int) -> ir.Type | None:
        if index < len(self._iter_args):
            return self._iter_args[index].type
        return None

    def _emit_return_vars(self, vars_: list[ir.Var]) -> None:
        builder = self._frame.sema._builder
        for v in vars_:
            builder.add_return_var(v)

    def _end_region(self) -> None:
        self._frame.sema._builder.end_for_loop(_UNKNOWN)
        self._frame._finish_construct_comments(self._comments_pending)
        self._comments_pending = False

    def _frame_yield_emit(self, exprs: list[ir.Expr]) -> None:
        self._frame.sema._builder.emit(ir.YieldStmt(exprs, _UNKNOWN))


class _DeferredStagedExit:
    """A staged Construct whose out-of-order guard exit defers builder close.

    The success path's interpreter ``close()`` runs it in LIFO order, and the
    exception path discards it with the one-shot Sema."""

    def out_of_order_exit(self, exception_triple: tuple[object, ...] | None = None) -> None:
        del exception_triple


class PtoWhileConstruct(_YieldingConstruct[Facade], _DeferredStagedExit, WhileConstruct[Facade]):
    """A staging-capable ``while``: exists from lexical entry, starts in host
    execution, and activates staging at most once."""

    def __init__(self, frame: PtoFrame) -> None:
        super().__init__()
        self._frame = frame
        self._iter_args: list[ir.IterArg] = []
        self._explicit = False
        self._comments_pending = False
        self._interface_tags = frozenset()
        self._init_yield_state()

    def set_iter_args(self, iter_args: tuple[ir.IterArg, ...]) -> None:
        """Claim pending `iter_args` at creation (explicit-while state lives on
        this Construct from the start, never on the Frame)."""
        self._iter_args = list(iter_args)
        self._explicit = True
        self._interface_tags = self._interface_tags | {"yield"}

    def _activate(self, pred: object) -> bool:
        """The protected activation hook (the base class owns the flag).

        The Construct owns its explicit-while state (claimed `iter_args`);
        the hook reads no mutable Frame state. An explicit carried while must
        activate on its first predicate — the predicate references the carried
        values by construction, so a host first predicate means the group was
        never meant for this loop and is rejected rather than drifting to a
        later opener.
        """
        if not is_staged(pred):
            if self._explicit:
                raise TypeError(
                    "pl.intrin.carry: an explicit (carried) while must activate "
                    "on its first predicate — the predicate stayed host, so "
                    "the carried values are never used; claim the carry with the "
                    "loop that consumes it"
                )
            return False
        frame = self._frame
        frame._require_bool(pred, "while")
        comments_pending = frame._begin_construct_comments()
        try:
            frame.sema._builder.begin_while_loop(_predicate_expr(pred))
            for ia in self._iter_args:
                frame.sema._builder.add_while_iter_arg(ia)
        except Exception:
            frame._finish_construct_comments(comments_pending)
            raise
        self._comments_pending = comments_pending
        return True

    # —— transfers ——
    def on_break(self, rt: Interpreter[Facade]) -> None:
        del rt
        self._frame.sema._builder.break_stmt()

    def on_continue(self, rt: Interpreter[Facade]) -> None:
        del rt
        self._frame.sema._builder.continue_stmt()

    def zero_operand_yield(self) -> _ZeroOperandYieldAction:
        """A staged while: ACCEPT with inferred loop-carried values (no
        `iter_args` — the zero-operand yield is the backedge terminator),
        BLOCK once explicit; a host while is a host Construct (CROSS)."""
        if not self.is_staged():
            return _ZeroOperandYieldAction.CROSS
        return _ZeroOperandYieldAction.BLOCK if self._explicit else _ZeroOperandYieldAction.ACCEPT

    # —— close / completion ——
    def close(self) -> CloseResult:
        if not self.is_staged():
            # A host close invokes no staged finalizer.
            return Closed()
        if self._explicit and not self._completed:
            return AwaitingResultBindings(BranchDisposition.NOT_APPLICABLE)
        self._frame.sema._builder.end_while_loop()
        self._frame._finish_construct_comments(self._comments_pending)
        self._comments_pending = False
        return Closed()

    @property
    def result_arity(self) -> int | None:
        return len(self._iter_args) if self._iter_args else None

    def _validate_first_yield_arity(self) -> None:
        assert self._yield_arity is not None
        if self._yield_arity != len(self._iter_args):
            raise TypeError(
                f"pl.intrin.yield_ arity {self._yield_arity} mismatches "
                f"while iter_args {len(self._iter_args)}"
            )

    def _leak_var_type(self, index: int) -> ir.Type | None:
        if index < len(self._iter_args):
            return self._iter_args[index].type
        return None

    def _emit_return_vars(self, vars_: list[ir.Var]) -> None:
        builder = self._frame.sema._builder
        for v in vars_:
            builder.add_while_return_var(v)

    def _end_region(self) -> None:
        self._frame.sema._builder.end_while_loop(_UNKNOWN)
        self._frame._finish_construct_comments(self._comments_pending)
        self._comments_pending = False

    def _frame_yield_emit(self, exprs: list[ir.Expr]) -> None:
        self._frame.sema._builder.emit(ir.YieldStmt(exprs, _UNKNOWN))


class PtoWithConstruct(_DeferredStagedExit, WithConstruct[Facade]):
    """A staged ``with`` over a trace context manager: the object enters
    through ``__trace_enter__`` at construction and self-closes through
    ``__trace_exit__``; this Construct owns its active lifecycle and deferred
    comment attachment."""

    def __init__(
        self,
        frame: PtoFrame,
        descriptor: _WithDescriptor,
        binding: object,
        deferred_comments: Sequence[str] | None,
        comments_pending: bool,
    ) -> None:
        super().__init__(descriptor)
        self._frame = frame
        self._descriptor = descriptor
        self._binding = binding
        self._deferred_comments = list(deferred_comments) if deferred_comments else None
        self._comments_pending = comments_pending
        self._staged_closed = False
        self._interface_tags = frozenset()

    def is_staged(self) -> bool:
        return True

    def enter(self) -> object:
        return self._binding

    def close(self, exception_triple: tuple[object, ...] | None = None) -> CloseResult:
        del exception_triple  # the descriptors' exits are not exception-sensitive
        if self._staged_closed:
            return Closed()
        self._staged_closed = True
        closed_stmt = self._descriptor.__trace_exit__(self._frame.trace_context())
        # A deferred spmd carrier that never triggered (a dispatch body) or
        # already opened retires here — the Frame's pending list must not
        # outlive the with.
        self._frame._retire_deferred_scope(self._descriptor)
        if self._deferred_comments:
            if not isinstance(closed_stmt, ir.Stmt):
                raise RuntimeError(
                    "deferred construct comments require __trace_exit__ to "
                    "return the closed region's statement"
                )
            ir.attach_leading_comments(closed_stmt, self._deferred_comments)
        self._frame._finish_construct_comments(self._comments_pending)
        self._comments_pending = False
        return Closed()
