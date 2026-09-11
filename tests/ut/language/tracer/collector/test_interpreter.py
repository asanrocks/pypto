# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Interpreter unit tests: the namespace chain, the frame lifecycle,
the event rows, and the Construct-stack contract — one ordered stack of Construct
objects, exact callback identity, the while's lazy staging decision, one
representative loop iteration, transfers, and pending result constructs."""

import pytest
from collector.interface import (
    AwaitingResultBindings,
    BindingResult,
    BranchDisposition,
    Closed,
    CloseResult,
    FinalResultBinding,
    ForOpen,
    HostBinding,
    NullFrame,
    NullSema,
    PartialResultBinding,
    StagedBinding,
    WithOpen,
)
from collector.interpreter import (
    ForConstruct,
    IfConstruct,
    Interpreter,
    InterpreterError,
    WhileConstruct,
    WithConstruct,
    ZeroOperandYieldAction,
)


def make_interpreter():
    return Interpreter(NullSema())


def test_enter_exit_frame_binds_parameters():
    rt = make_interpreter()
    rt.enter_frame(a=1, b=2)
    assert rt.vars.a == 1
    assert rt.vars.b == 2
    rt.exit_frame()
    assert rt._frames == []


def test_vars_write_read_frame_ns():
    rt = make_interpreter()
    rt.enter_frame()
    rt.vars.x = 41
    assert rt.vars.x == 41
    rt.vars.x = rt.vars.x + 1
    assert rt.vars.x == 42


def test_unbound_read_raises():
    rt = make_interpreter()
    rt.enter_frame()
    with pytest.raises(UnboundLocalError):
        rt.vars.never_bound


def test_branch_local_chain_walk():
    """reads chain-walk the per-branch locals mappings innermost-first; the chain is
    empty in simulation mode but the walk is structural."""
    rt = make_interpreter()
    rt.enter_frame()
    rt.vars.x = "frame"
    rt._branch_locals.append({"x": "then"})
    assert rt.vars.x == "then"
    rt._branch_locals.pop()
    assert rt.vars.x == "frame"


def test_emit_passthrough():
    rt = make_interpreter()
    rt.enter_frame()
    assert rt._emit(42) == 42
    assert rt._emit(None) is None


def test_return_records_single_tuple_none():
    rt = make_interpreter()
    rt.enter_frame()
    rt.return_(7)
    assert rt._frames[-1].ns["__return__"] == 7
    rt.return_(1, 2)
    assert rt._frames[-1].ns["__return__"] == (1, 2)
    rt.return_()
    assert rt._frames[-1].ns["__return__"] is None


def test_bind_returns_value_and_routes():
    rt = make_interpreter()
    rt.enter_frame()
    assert rt.bind("n", 5) == 5
    assert rt.vars.n == 5


def test_in_host_mode_true_in_simulation():
    rt = make_interpreter()
    rt.enter_frame()
    assert rt.in_host_mode() is True
    # even with host constructs pushed
    rt._construct_stack.append(IfConstruct(True))
    assert rt.in_host_mode() is True


# ------------------------------------------------------------------------


def test_eval_if_then_else_truth_table():
    rt = make_interpreter()
    rt.enter_frame()
    for truthy, falsy in [(True, False), (1, 0), ("x", "")]:
        rt.eval_if(truthy)
        assert rt.eval_then() is True
        rt.eval_else()
        assert rt.eval_else_active() is False
        rt.exit_construct()
        rt.eval_if(falsy)
        assert rt.eval_then() is False
        rt.eval_else()
        assert rt.eval_else_active() is True
        rt.exit_construct()
    assert rt._construct_stack == []


def test_if_host_construct_balance():
    """eval_if installs a host IfConstruct, exit_construct removes it — balanced
    by construction; writes in the taken branch land in the frame ns."""
    rt = make_interpreter()
    rt.enter_frame()
    rt.eval_if(True)
    assert len(rt._construct_stack) == 1
    assert isinstance(rt._construct_stack[0], IfConstruct)
    assert rt._construct_stack[0].kind == "if"
    if rt.eval_then():
        rt.vars.x = 1
    rt.eval_else()
    if rt.eval_else_active():
        rt.vars.x = 2
    rt.exit_construct()
    assert rt._construct_stack == []
    assert rt.vars.x == 1


def test_exit_construct_idempotent_on_empty():
    """exit_construct on an empty stack no-ops (defensive)."""
    rt = make_interpreter()
    rt.enter_frame()
    rt.exit_construct()
    rt.exit_construct()
    assert rt._construct_stack == []


def test_predicate_single_eval():
    evals = []
    rt = make_interpreter()
    rt.enter_frame()
    rt.eval_if(evals.append(1) or True)
    assert evals == [1]  # the predicate expression itself evaluated once


# ------------------------------------------------------------------------


def test_begin_for_host_returns_iterable():
    rt = make_interpreter()
    rt.enter_frame()
    it = [1, 2, 3]
    assert rt.begin_for(it) is it
    assert len(rt._construct_stack) == 1
    assert isinstance(rt._construct_stack[0], ForConstruct)
    assert rt._construct_stack[0].kind == "for"
    rt.exit_construct()
    assert rt._construct_stack == []


def test_begin_for_zero_iteration_balanced():
    rt = make_interpreter()
    rt.enter_frame()
    for x in rt.begin_for([]):
        rt.vars.x = x
    rt.exit_construct()
    assert rt._construct_stack == []


def test_eval_bool_pushes_once_per_site():
    """eval_bool installs the site's WhileConstruct on the thunk's first call
    only (keyed by the thunk's identity) — repeated head evaluations do not
    grow the stack."""
    rt = make_interpreter()
    rt.enter_frame()
    thunk = lambda: False  # noqa: E731 - a fresh thunk per site is the subject
    for _ in range(3):
        assert rt.eval_bool(thunk) is False
    assert len(rt._construct_stack) == 1
    rt.exit_construct()
    assert rt._construct_stack == []


def test_two_while_sites_tracked_independently():
    rt = make_interpreter()
    rt.enter_frame()
    t1 = lambda: False  # noqa: E731 - distinct thunks for distinct sites
    t2 = lambda: False  # noqa: E731 - distinct thunks for distinct sites
    rt.eval_bool(t1)
    rt.eval_bool(t2)
    assert len(rt._construct_stack) == 2
    rt.exit_construct()
    rt.exit_construct()
    assert rt._construct_stack == []


def test_eval_bool_unstable_thunk_leaks_without_hoist():
    """``eval_bool`` dedups the construct install by the thunk's *object
    identity* — a stable (hoisted) thunk is O(1); a fresh thunk per call
    (the pre-hoist inline-lambda shape) is not. This pins the interpreter
    contract that motivates the source rewriter's while-thunk hoist."""
    rt = make_interpreter()
    rt.enter_frame()
    for _ in range(5):
        rt.eval_bool(lambda: False)  # a fresh thunk each call → distinct objects
    assert len(rt._construct_stack) == 5  # unstable → one install per call
    rt.close()
    # the hoisted shape (one thunk, re-driven by the host while) is O(1):
    rt2 = make_interpreter()
    rt2.enter_frame()
    thunk = lambda: False  # noqa: E731 - a stable thunk is the emitted shape
    for _ in range(5):
        rt2.eval_bool(thunk)
    assert len(rt2._construct_stack) == 1  # stable → one install
    rt2.close()


def test_host_while_site_redrive_stays_balanced():
    """A host loop re-driving a closed host while site installs a fresh
    construct for the site — the stack stays balanced across outer iterations."""
    rt = make_interpreter()
    rt.enter_frame()
    thunk = lambda: False  # noqa: E731 - the hoisted per-site thunk
    for _ in range(3):  # an outer host loop re-driving the site
        while rt.eval_bool(thunk):  # noqa: B023 - the thunk is stable
            pass
        rt.exit_construct()  # the generated trailing exit
    assert rt._construct_stack == []
    rt.close()


# ------------------------------------------------------------------------


def test_eval_and_or_short_circuit():
    rt = make_interpreter()
    rt.enter_frame()
    fired = []

    def probe():
        fired.append(1)
        return "rhs"

    assert rt.eval_and(0, probe) == 0
    assert fired == []
    assert rt.eval_and(1, probe) == "rhs"
    assert fired == [1]
    assert rt.eval_or(1, probe) == 1
    assert fired == [1]
    assert rt.eval_or(0, probe) == "rhs"
    assert fired == [1, 1]


def test_eval_not():
    rt = make_interpreter()
    rt.enter_frame()
    assert rt.eval_not(0) is True
    assert rt.eval_not(1) is False
    assert rt.eval_not("") is True


def test_break_continue_host_return_true():
    rt = make_interpreter()
    rt.enter_frame()
    assert rt.break_() is True
    assert rt.continue_() is True


def test_on_begin_with_host_preserves_cm_protocol():
    """A host ``with`` wraps the user's context manager: entry, exit,
    exception triple, and suppression fire exactly once through the adapter;
    the interpreter installs one host WithConstruct for it."""
    rt = make_interpreter()
    rt.enter_frame()
    log = []

    class Cm:
        def __enter__(self):
            log.append("enter")
            return "value"

        def __exit__(self, *exc):
            log.append(("exit", exc[0] is not None))
            return False

    adapter = rt.on_begin_with(Cm())
    assert isinstance(rt._construct_stack[-1], WithConstruct)
    assert rt._construct_stack[-1].kind == "with"
    with adapter as v:
        assert v == "value"
    assert log == ["enter", ("exit", False)]
    assert rt._construct_stack == []  # the adapter exit removed the construct


def test_on_begin_with_host_suppression_preserved():
    rt = make_interpreter()
    rt.enter_frame()

    class SuppressCm:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return True  # suppress

    adapter = rt.on_begin_with(SuppressCm())
    with adapter:
        raise ValueError("suppressed")
    assert rt._construct_stack == []


# ------------------------------------------------------------------------
# The staged wiring against a recording Frame (the language-plugin
# extension pattern): every instrumented construct installs the exact Construct the
# Frame created; the trace interpreter drives callbacks to that exact object.


class _Marker:
    pass


class _StagedIf(IfConstruct):
    def __init__(self, frame):
        super().__init__(guard=True)
        self.frame = frame
        self.elsed = 0
        self.closed = 0
        self.snapshot_taken = False

    def is_staged(self):
        return True

    def then_active(self):
        return True

    def else_active(self):
        return True

    def enter_else(self):
        self.elsed += 1
        self.frame.if_else_swaps.append(self)

    def close(self):
        self.closed += 1
        return Closed(BranchDisposition.COMMIT)


class _StagedLoop(ForConstruct):
    def __init__(self, frame):
        super().__init__(())
        self.frame = frame
        self.breaks = 0
        self.continues = 0
        self.closed = 0
        self.activations = 0

    def is_staged(self):
        return True

    def iterations(self):
        return ["target"]

    def activate_target(self, ctx, name):
        self.activations += 1
        return "loop-var"

    def on_break(self, rt):
        self.breaks += 1

    def on_continue(self, rt):
        self.continues += 1

    def close(self) -> CloseResult:
        self.closed += 1
        return Closed()


class _StagedWhile(WhileConstruct):
    """Makes a lazy staging decision: activates on a Marker predicate,
    records the thunk evaluations it observes."""

    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        self.breaks = 0
        self.continues = 0
        self.closed = 0

    def _activate(self, pred):
        if isinstance(pred, _Marker):
            self.frame.while_activations.append(self)
            return True
        return False

    def on_break(self, rt):
        self.breaks += 1

    def on_continue(self, rt):
        self.continues += 1

    def close(self):
        if self.is_staged():
            self.closed += 1
        return Closed()


class _StagedWith(WithConstruct):
    def __init__(self, frame, target):
        super().__init__(cm=None)
        self.frame = frame
        self._target = target
        self.closed = 0

    def is_staged(self):
        return True

    def enter(self):
        return self._target

    def close(self, exception_triple=None):
        self.closed += 1
        return Closed()


class _StagedFrame(NullFrame):
    """Every ``_Marker`` is staged; binds record and echo the value."""

    def __init__(self):
        self.bound = []
        self.expression_statements = []
        self.rets = []
        self.ifs = []
        self.fors = []
        self.whiles = []
        self.withs = []
        self.if_else_swaps = []
        self.while_activations = []

    def bind(
        self,
        interpreter: Interpreter[object],
        /,
        name: str,
        value: object,
        span: object | None = None,
    ) -> BindingResult[object]:
        if isinstance(value, _Marker):
            self.bound.append((name, value))
            return StagedBinding(value)
        return HostBinding()

    def expression_statement(self, interpreter, value, span=None):
        self.expression_statements.append(value)

    def return_(self, interpreter, values, span=None):
        self.rets.append(values)

    def create_if_construct(self, interpreter, predicate, span=None):
        if isinstance(predicate, _Marker):
            construct = _StagedIf(self)
            self.ifs.append(construct)
            return construct
        return None

    def create_for_construct(self, interpreter, iterable, span=None):
        if isinstance(iterable, _Marker):
            construct = _StagedLoop(self)
            self.fors.append(construct)
            return ForOpen(construct, "target")
        return None

    def create_while_construct(self, interpreter, predicate_thunk, span=None):
        construct = _StagedWhile(self)
        self.whiles.append(construct)
        return construct

    def create_with_construct(self, interpreter, context, name=None, span=None, single_call_body=False):
        if isinstance(context, _Marker):
            construct = _StagedWith(self, "target")
            self.withs.append(construct)
            return WithOpen(construct, "target")
        return None


class _StagedSema(NullSema):
    def __init__(self):
        self.frames = []

    def begin_frame(self):
        frame = _StagedFrame()
        self.frames.append(frame)
        return frame


def _staged_interpreter() -> tuple[Interpreter[object], _StagedFrame]:
    sema = _StagedSema()
    rt = Interpreter(sema)
    rt.enter_frame()
    return rt, sema.frames[-1]


def test_staged_bind_reaches_frame():
    """The bind filter: a staged value reaches the Frame's
    ``bind`` (which returns the representation to store); a host value is
    declined and stored in the namespace."""
    rt, frame = _staged_interpreter()
    marker = _Marker()
    rt.vars.sym = marker
    rt.vars.host = 3
    assert rt.vars.host == 3
    assert rt.vars.sym is marker  # the representation the Frame returned
    assert frame.bound == [("sym", marker)]
    # the host-binding fact (the typed staged initialization check's input):
    assert rt.current_binding("host") == (True, 3)
    assert rt.current_binding("sym") == (False, marker)


def test_staged_if_exact_construct_callbacks():
    """``eval_if`` installs the Frame's exact IfConstruct; ``eval_else`` drives
    that exact object; ``exit_construct`` closes it and commits the per-branch
    locals."""
    rt, frame = _staged_interpreter()
    rt.eval_if(_Marker())
    construct = rt._construct_stack[-1]
    assert isinstance(construct, _StagedIf)
    assert construct is frame.ifs[0]
    assert rt.eval_then() is True  # all-paths
    rt.vars.inside = _Marker()
    rt.eval_else()
    assert construct.elsed == 1
    assert rt.eval_else_active() is True  # all-paths
    rt.exit_construct()
    assert rt._construct_stack == []
    assert construct.closed == 1
    assert frame.bound == [("inside", frame.bound[0][1])] if frame.bound else True
    # the then-branch locals mapping was committed into the frame namespace
    assert "inside" in rt._frames[-1].ns


def test_staged_if_branch_locals_isolation():
    """Branch-local writes stay in their own per-branch locals mapping until
    the close commits the union."""
    rt, frame = _staged_interpreter()
    rt.eval_if(_Marker())
    rt.vars.x = "then"
    assert rt.vars.x == "then"
    rt.eval_else()
    # the then-branch mapping left the chain: ``x`` has no value on the else path
    # (the unsoundness the IR verifier's use-after-def check rejects).
    with pytest.raises(UnboundLocalError):
        rt._route_read("x")
    rt.exit_construct()
    assert rt._frames[-1].ns["x"] == "then"


def test_nested_staged_if_merge_stays_in_enclosing_branch_locals():
    """A nested staged if closes while the enclosing staged branch is
    still open: the merge lands in the innermost active per-branch locals
    mapping (exactly
    where a plain bind at that point would land), never in the frame
    namespace — the outer else path must not observe values bound only on
    the outer then path."""
    rt, frame = _staged_interpreter()
    rt.eval_if(_Marker())  # outer staged if
    rt.eval_if(_Marker())  # nested inside the outer then branch
    rt.vars.y = "inner-then"
    rt.exit_construct()  # nested close: merge, but stay branch-local
    assert len(rt._branch_locals) == 1  # only the outer then-branch mapping remains
    assert "y" in rt._branch_locals[0]
    assert "y" not in rt._frames[-1].ns
    # the outer else path must not see the then-only value
    rt.eval_else()
    with pytest.raises(UnboundLocalError):
        rt._route_read("y")
    rt.exit_construct()  # outer close: both branch mappings commit into the namespace
    assert rt._frames[-1].ns["y"] == "inner-then"


def test_staged_expression_statement_and_return_reach_frame():
    rt, frame = _staged_interpreter()
    marker = _Marker()
    rt._emit(marker)
    rt.return_(marker)
    assert frame.expression_statements == [marker]
    assert frame.rets == [[marker]]


def test_staged_begin_for_returns_binding_proxy_surface():
    rt, frame = _staged_interpreter()
    surface = rt.begin_for(_Marker())
    assert surface == ["target"]
    construct = rt._construct_stack[-1]
    assert isinstance(construct, _StagedLoop)
    assert construct is frame.fors[0]
    rt.exit_construct()
    assert construct.closed == 1


def test_staged_eval_bool_lazy_activation_and_stable_identity():
    """The while making a lazy staging decision: the site installs one PtoWhile-capable
    construct at lexical entry; the first staged predicate activates it in
    place (identity stable); the body traces one representative iteration;
    the thunk is never reevaluated after activation."""
    rt, frame = _staged_interpreter()
    pred = _Marker()
    evaluations = []

    def thunk():
        evaluations.append(1)
        return pred

    assert rt.eval_bool(thunk) is True  # activation → trace the body once
    construct = rt._construct_stack[-1]
    assert isinstance(construct, _StagedWhile)
    assert construct is frame.whiles[0]
    assert construct.is_staged() is True
    assert frame.while_activations == [construct]
    assert rt.eval_bool(thunk) is False  # the representative iteration ends the host loop
    assert rt.eval_bool(thunk) is False
    assert len(evaluations) == 1  # the thunk is not reevaluated after activation
    assert rt._construct_stack[-1] is construct  # identity never changed
    rt.exit_construct()
    assert construct.closed == 1
    assert rt._construct_stack == []


def test_host_while_never_activates_and_closes_clean():
    """A while whose predicates stay host: the site construct exists but
    never activates; the host close invokes no staged finalizer."""
    rt, frame = _staged_interpreter()
    count = [0]

    def thunk():
        count[0] += 1
        return count[0] < 3

    results = []
    while rt.eval_bool(thunk):
        results.append(1)
    assert results == [1, 1]  # two truthy evaluations, then the false one
    construct = rt._construct_stack[-1]
    assert isinstance(construct, _StagedWhile)
    assert construct is frame.whiles[0]
    assert construct.is_staged() is False
    assert frame.while_activations == []
    rt.exit_construct()
    assert construct.closed == 0  # no staged finalizer on a host close
    assert rt._construct_stack == []


def test_staged_while_records_break_on_exact_construct():
    rt, frame = _staged_interpreter()
    pred = _Marker()
    rt.eval_bool(lambda: pred)
    construct = rt._construct_stack[-1]
    assert isinstance(construct, _StagedWhile)
    assert rt.break_() is False  # record-and-continue, not a host break
    assert construct.breaks == 1
    rt.exit_construct()


def test_loop_else_rejects_staged_close_allows_host():
    """The ``has_else`` static fact: closing a staged
    loop into its else suite is a loud rejection (the alternate break path
    was not collected); a host loop keeps its native else."""
    rt, frame = _staged_interpreter()
    pred = _Marker()
    rt.eval_bool(lambda: pred)
    with pytest.raises(InterpreterError, match="cannot have an 'else' suite"):
        rt.exit_construct(else_follows=True)

    rt2, _ = _staged_interpreter()
    count = [0]

    def thunk():
        count[0] += 1
        return count[0] < 2

    while rt2.eval_bool(thunk):
        pass
    rt2.exit_construct(else_follows=True)  # host loop: the native else runs
    assert rt2._construct_stack == []


def test_staged_with_adapter_closes_exact_construct():
    rt, frame = _staged_interpreter()
    adapter = rt.on_begin_with(_Marker())
    construct = rt._construct_stack[-1]
    assert isinstance(construct, _StagedWith)
    assert construct is frame.withs[0]
    with adapter as target:
        assert target == "target"
    assert rt._construct_stack == []
    assert construct.closed == 1


def test_trusted_queries():
    rt, frame = _staged_interpreter()
    assert rt.current_construct() is None
    assert rt.active_constructs() == ()
    assert rt.nearest_kind("if", "for", "while") is None
    assert rt.nearest_supporting("yield") is None
    assert rt.staged_control_flow_active() is False
    assert rt.pending_result() is None

    rt.begin_for(_Marker())
    loop = rt._construct_stack[-1]
    assert rt.current_construct() is loop
    assert rt.nearest_kind("for") is loop
    assert rt.staged_control_flow_active() is True
    rt.eval_if(_Marker())
    top = rt._construct_stack[-1]
    assert rt.active_constructs() == (top, loop)
    assert rt.nearest_kind("for") is loop
    assert rt.in_host_mode() is False
    rt.exit_construct()
    rt.exit_construct()
    assert rt.in_host_mode() is True
    assert rt.staged_control_flow_active() is False


# ------------------------------------------------- pending result constructs ----


class _PendingResultLoop(_StagedLoop):
    """A result-producing loop: close leaves it as the pending result
    construct; a ``FinalResultBinding`` bind removes it."""

    def __init__(self, frame):
        super().__init__(frame)
        self.completed = False

    def iterations(self):
        return ["target"]

    def close(self):
        if not self.completed:
            return AwaitingResultBindings(BranchDisposition.NOT_APPLICABLE)
        return Closed()


class _FinalResultValue:
    """An operation-result binding-proxy stand-in: its bind completes the
    named pending result construct."""

    def __init__(self, construct):
        self._construct = construct

    def __trace_bind__(self, ctx, name):
        self._construct.completed = True
        return self._construct


class _PartialResultValue:
    """An operation-result binding-proxy stand-in that only advances the
    result binding (partial)."""

    def __init__(self, construct):
        self._construct = construct

    def __trace_bind__(self, ctx, name):
        return self._construct


class _PendingResultFrame(_StagedFrame):
    """A frame whose for-constructs produce results, and whose bind routes
    the result-binding stand-ins through the typed results."""

    def create_for_construct(self, interpreter, iterable, span=None):
        if isinstance(iterable, _Marker):
            construct = _PendingResultLoop(self)
            self.fors.append(construct)
            return ForOpen(construct, "target")
        return None

    def bind(self, interpreter, name, value, span=None):
        if isinstance(value, (_FinalResultValue, _PartialResultValue)):
            construct = value._construct
            value.__trace_bind__(None, name)
            facade = _Marker()
            if construct.completed:
                return FinalResultBinding[object](construct, facade)
            return PartialResultBinding[object](construct, facade)
        return super().bind(interpreter, name, value, span)


class _PendingResultSema(NullSema):
    def __init__(self):
        self.frames = []

    def begin_frame(self):
        frame = _PendingResultFrame()
        self.frames.append(frame)
        return frame


def _pending_result_interpreter() -> tuple[Interpreter[object], _PendingResultFrame]:
    sema = _PendingResultSema()
    rt = Interpreter(sema)
    rt.enter_frame()
    return rt, sema.frames[-1]


def test_result_bearing_close_is_exact_top_pending_result():
    rt, frame = _pending_result_interpreter()
    rt.begin_for(_Marker())
    construct = rt._construct_stack[-1]
    rt.exit_construct()
    # the construct stayed as the exact top pending result construct
    assert rt._construct_stack == [construct]
    assert rt.pending_result() is construct

    # ordinary operations reject the pending result construct before mutation
    for op in (
        lambda: rt.eval_if(_Marker()),
        lambda: rt.begin_for(_Marker()),
        lambda: rt.eval_bool(lambda: _Marker()),
        lambda: rt.on_begin_with(_Marker()),
        lambda: rt.eval_then(),
        lambda: rt.eval_else(),
        lambda: rt.eval_else_active(),
        lambda: rt.exit_construct(),
        lambda: rt.break_(),
    ):
        with pytest.raises(InterpreterError):
            op()
    assert rt._construct_stack == [construct]  # unchanged

    # the final result binding removes the exact pending result construct
    rt._route_bind("out", _FinalResultValue(construct))
    assert rt._construct_stack == []
    assert rt.pending_result() is None


def test_partial_then_final_result_binds():
    """A partial result binding keeps the pending result construct; the
    final result binding removes it and stores the value."""
    rt, _frame = _pending_result_interpreter()
    rt.begin_for(_Marker())
    construct = rt._construct_stack[-1]
    rt.exit_construct()
    assert rt.pending_result() is construct

    # partial result binding: the pending result construct stays
    partial = _PartialResultValue(construct)
    stored = rt._route_bind("acc", partial)
    assert isinstance(stored, _Marker)
    assert rt.pending_result() is construct

    done = _FinalResultValue(construct)
    stored = rt._route_bind("out", done)
    assert rt._construct_stack == []
    assert rt.pending_result() is None
    assert rt._frames[-1].ns["out"] is stored


def test_close_rejects_unfinished_pending_result():
    rt, _frame = _pending_result_interpreter()
    rt.begin_for(_Marker())
    rt.exit_construct()
    assert rt.pending_result() is not None
    with pytest.raises(InterpreterError, match="result bindings for the staged for"):
        rt.close()


def test_eval_else_rejects_pending_result_before_mutation():
    """The branch-switch operations are ordinary operations: the guard
    must fire before the kind dispatch — without it, an If-kind pending
    result construct would take a branch-switch mutation on the staged
    construct (and a
    loop-kind pending result construct would surface as a mismatched-kind
    error) instead
    of the pending-result rejection."""
    rt, _frame = _pending_result_interpreter()
    rt.begin_for(_Marker())
    rt.exit_construct()  # enters the result-binding phase
    with pytest.raises(InterpreterError, match="eval_else: result bindings for the staged for"):
        rt.eval_else()
    assert rt._construct_stack == [rt.pending_result()]  # no mutation


def test_while_sites_pruned_on_construct_removal():
    """The thunk→construct site map tracks open sites only: a closed while
    construct forgets its entry, so a re-driven site or an unstable thunk
    does not accumulate dead entries across block executions."""
    rt = make_interpreter()
    rt.enter_frame()
    thunk = lambda: False  # noqa: E731 - the hoisted per-site thunk
    rt.eval_bool(thunk)
    assert len(rt._while_sites) == 1
    rt.exit_construct()
    assert rt._while_sites == {}
    # a redrive of the same thunk installs a fresh site and prunes it again
    rt.eval_bool(thunk)
    assert len(rt._while_sites) == 1
    rt.close()
    assert rt._while_sites == {}

    # fresh thunks (one per block entry) each install a site; close forgets all
    rt2 = make_interpreter()
    rt2.enter_frame()
    for _ in range(5):
        rt2.eval_bool(lambda: False)  # noqa: B023 - distinct sites are the subject
    assert len(rt2._while_sites) == 5
    rt2.close()
    assert rt2._while_sites == {}


def test_ordinary_binds_rejected_while_pending_result_present():
    """The typed-result contract at the Interpreter boundary: while the
    exact top pending result construct is in its result-binding phase, only
    result bindings addressing
    that exact construct are accepted — a declined or ordinary bind result is
    rejected before the namespace mutation."""
    rt, _frame = _pending_result_interpreter()
    rt.begin_for(_Marker())
    construct = rt._construct_stack[-1]
    rt.exit_construct()
    with pytest.raises(InterpreterError, match="bind: result bindings for the staged for"):
        rt._route_bind("host", 3)  # HostBinding
    with pytest.raises(InterpreterError, match="bind: result bindings for the staged for"):
        rt._route_bind("sym", _Marker())  # StagedBinding
    assert rt._construct_stack == [construct]  # the pending result construct is untouched
    assert "host" not in rt._frames[-1].ns  # no mutation happened


def test_final_result_binding_without_pending_result_rejected():
    """A result-binding outcome (partial or final) with no pending result
    construct registered is
    a closed-protocol violation, not a silent store."""
    rt, _frame = _pending_result_interpreter()
    rt.begin_for(_Marker())
    construct = rt._construct_stack[-1]
    rt.exit_construct()
    rt._route_bind("out", _FinalResultValue(construct))  # final → construct removed
    with pytest.raises(InterpreterError, match="no staged result construct"):
        rt._route_bind("late", _PartialResultValue(construct))
    with pytest.raises(InterpreterError, match="no staged result construct"):
        rt._route_bind("late", _FinalResultValue(construct))


def test_partial_result_binding_must_name_the_exact_pending_result():
    rt, _frame = _pending_result_interpreter()
    rt.begin_for(_Marker())
    outer = rt._construct_stack[-1]
    rt.begin_for(_Marker())
    rt.exit_construct()  # the inner loop is the pending result construct
    with pytest.raises(InterpreterError, match="exact top Construct"):
        rt._route_bind("x", _PartialResultValue(outer))


def test_pending_result_rejects_return_and_logical_ops_but_permits_comments():
    """Return and the lazy logical operations are ordinary operations
    against the pending result construct — rejected before mutation, and
    the ``and``/``or``
    right-hand thunks are never evaluated. Comment attachment is the one
    deliberate exemption: the pending-comment queue is Frame annotation
    state consumed by the next permitted operation (a natural comment
    between a result-producing close and its result binding must not
    reject)."""
    rt, _frame = _pending_result_interpreter()
    rt.begin_for(_Marker())
    rt.exit_construct()
    with pytest.raises(InterpreterError, match="return"):
        rt.return_(_Marker())
    rt._leading_comments(["# collected after the region close"])
    thunk_evals: list[int] = []

    def thunk() -> object:
        thunk_evals.append(1)
        return _Marker()

    with pytest.raises(InterpreterError, match="and"):
        rt.eval_and(_Marker(), thunk)
    with pytest.raises(InterpreterError, match="or"):
        rt.eval_or(_Marker(), thunk)
    with pytest.raises(InterpreterError, match="not"):
        rt.eval_not(_Marker())
    assert thunk_evals == []  # the RHS thunk never ran
    assert rt._construct_stack == [rt.pending_result()]  # no mutation


def test_close_rejects_result_bearing_close_during_unwind():
    """A construct whose close produces results but was never registered as
    the pending result construct (no exit_construct): the finalization
    unwind fails loudly
    instead of silently discarding the typed result. The registered case
    is the close head guard (test_close_rejects_unfinished_pending_result)."""
    rt, _frame = _pending_result_interpreter()
    rt.begin_for(_Marker())
    assert rt.pending_result() is None  # never staged
    with pytest.raises(InterpreterError, match="produced results during trace finalization"):
        rt.close()


def test_staged_transfer_callbacks_are_a_checked_protocol():
    """A staged loop without the transfer callbacks is a loud contract
    error at selection time, not an AttributeError mid-dispatch."""

    class _TransferlessFor(ForConstruct):
        def is_staged(self):
            return True

    rt, _frame = _staged_interpreter()
    rt._construct_stack.append(_TransferlessFor(()))
    with pytest.raises(InterpreterError, match="transfer callbacks"):
        rt.break_()
    with pytest.raises(InterpreterError, match="transfer callbacks"):
        rt.continue_()


def test_nearest_zero_operand_yield_is_ordered_typed_selection():
    """The zero-operand yield selection is interpreter-owned and typed: the walk visits
    the active constructs innermost-first, crosses CROSS constructs, stops at the
    first ACCEPT/BLOCK answer, and never skips a nearer eligible target.
    The construct itself answers the eligibility question — the interpreter reads
    no language fact while selecting."""

    class _ZeroOperandYieldConstruct(IfConstruct):
        def __init__(self, action):
            super().__init__(guard=True)
            self._action = action

        def zero_operand_yield(self):
            return self._action

    rt, _frame = _staged_interpreter()
    assert rt.nearest_zero_operand_yield() is None  # no constructs: nothing to select

    outer = _ZeroOperandYieldConstruct(ZeroOperandYieldAction.ACCEPT)
    blocker = _ZeroOperandYieldConstruct(ZeroOperandYieldAction.BLOCK)
    inner = _ZeroOperandYieldConstruct(ZeroOperandYieldAction.ACCEPT)
    rt._construct_stack.extend([outer, blocker, inner])  # outermost → innermost
    sel = rt.nearest_zero_operand_yield()
    assert sel is not None and sel.construct is inner and sel.action is ZeroOperandYieldAction.ACCEPT

    rt._construct_stack.pop()  # the inner ACCEPT leaves
    sel = rt.nearest_zero_operand_yield()
    assert sel is not None and sel.construct is blocker  # BLOCK stops the walk
    assert sel.action is ZeroOperandYieldAction.BLOCK

    rt._construct_stack.pop()  # the BLOCK leaves
    sel = rt.nearest_zero_operand_yield()
    assert sel is not None and sel.construct is outer

    rt._construct_stack.pop()
    assert rt.nearest_zero_operand_yield() is None
