# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Coverage completion: the expression/construct corners the
main suites do not reach — set/dict/generator comprehensions, literal
collections, ternaries, nested-def compound bodies, and the staged-path
wiring (the trace hooks, exercised against recording Semas)."""

import collector
import pytest
from collector.interface import (
    Closed,
    CloseResult,
    ForOpen,
    HostBinding,
    NullFrame,
    NullSema,
    WithOpen,
)
from collector.interpreter import ForConstruct, IfConstruct, Interpreter, WhileConstruct, WithConstruct
from collector.rewriter import instrument_function_source


def test_set_and_generator_comprehensions():
    def kernel(data):
        s = {x * 2 for x in data}
        g = sum(x for x in data if x > 0)
        return s, g

    assert collector.instrument(kernel)([1, -2, 3]) == kernel([1, -2, 3])


def test_dict_comp_and_literals():
    def kernel(data):
        d = {k: k * k for k in data}
        lst = [1, 2, *data]
        st = {3, 4}
        tup = (5, *data)
        return d, lst, st, tup

    assert collector.instrument(kernel)([6, 7]) == kernel([6, 7])


def test_dict_starred_and_ternary():
    def kernel(a, b):
        d = {**a, "k": b}
        r = b if a["flag"] else -b
        return d, r

    assert collector.instrument(kernel)({"flag": True, "x": 1}, 5) == kernel({"flag": True, "x": 1}, 5)


def test_nested_comp_outer_iterable_reads_enclosing():
    def kernel(limit):
        return [y for y in range(limit) for x in (y, y * 2)]

    assert collector.instrument(kernel)(3) == kernel(3)


def test_nested_def_with_compound_bodies():
    """nested defs pass through with reads-only bodies: if/for/while/with/
    try-finally inside a helper stay host-language, outer reads route."""

    def kernel(data, scale):
        def helper(v):
            if v > 0:
                for i in range(v):
                    v = v + 1
            while v > 5:
                v = v - 1
            return v * scale

        return [helper(x) for x in data]

    assert collector.instrument(kernel)([1, 4], 2) == kernel([1, 4], 2)


def test_nested_def_try_finally_and_with():
    class Cm:
        def __enter__(self):
            return 1

        def __exit__(self, *exc):
            pass

    def kernel(cm):
        def helper(v):
            with cm:
                try:
                    v = v + 1
                finally:
                    v = v * 2
            return v

        return helper(3)

    assert collector.instrument(kernel)(Cm()) == kernel(Cm())


def test_nested_def_elif_chain():
    def kernel(n):
        def helper(x):
            if x < 0:
                return "neg"
            elif x == 0:
                return "zero"
            else:
                return "pos"

        return helper(n)

    for n in (-1, 0, 1):
        assert collector.instrument(kernel)(n) == kernel(n)


def test_import_and_class_binding_syncs():
    def kernel(x):
        import math  # noqa: PLC0415 - the test kernel is the subject

        class Box:
            pass

        b = Box()
        return math.floor(x), b.__class__.__name__

    assert collector.instrument(kernel)(3.7) == kernel(3.7)


def test_import_from_sync():
    def kernel():
        from math import sqrt  # noqa: PLC0415 - the test kernel is the subject

        return sqrt(9)

    assert collector.instrument(kernel)() == kernel()


def test_emission_literals_route_reads():
    out = instrument_function_source("def k(a, b):\n    return {a: b, **d}, [a, *b], (a,), {a, b}\n", "k")[0]
    assert "rt.vars.a" in out
    assert "rt.vars.b" in out


def test_emission_keeps_return_annotation():
    """the return annotation survives the rebuild; param
    annotations ride along inside params."""
    out = instrument_function_source("def k(x: int, *, y: str = 'a') -> float:\n    return float(x)\n", "k")[
        0
    ]
    assert "-> float" in out
    assert "x: int" in out
    assert "y: str" in out


# ------------------------------------------------ nested class bodies ---


def test_nested_class_methods_route_kernel_locals():
    """nested class methods close over kernel locals natively (they are
    freevars of the class body via its methods); reads-only redirection must
    route them through ``rt.vars`` — pre-fix this
    raised NameError."""

    def kernel():
        log = []

        class Cm:
            def __enter__(self):
                log.append("enter")
                return 1

            def __exit__(self, *a):
                log.append("exit")
                return False

        with Cm() as _b:
            pass
        return log

    assert collector.instrument(kernel)() == kernel()


def test_nested_class_body_reads_kernel_locals():
    """a class-body-level read of a kernel local resolves natively (the name
    is a freevar of the class body); the body's instrumented reads must preserve it."""

    def kernel(mod):
        class C:
            y = mod * 2

            def m(self):
                return self.y + mod

        return C().m()

    assert collector.instrument(kernel)(3) == kernel(3)


def test_nested_class_in_nested_def_reads_kernel_locals():
    """a class inside a helper: reads of kernel instrumented names redirect,
    the helper's own locals stay native."""

    def kernel(data, scale):
        def helper(v):
            class C:
                y = v * scale

            return C.y

        return [helper(x) for x in data]

    assert collector.instrument(kernel)([1, 2], 3) == kernel([1, 2], 3)


def test_nested_class_with_exception_runs_exit():
    """a kernel-defined context manager's ``__exit__`` fires on exception
    (host cm protocol preserved) — the class-body redirection must not break
    the exception path."""
    events = []

    def kernel():
        class Cm:
            def __enter__(self):
                events.append("enter")
                return 1

            def __exit__(self, *a):
                events.append("exit")
                return False

        with Cm() as _b:
            raise ValueError("boom")

    with pytest.raises(ValueError):
        collector.instrument(kernel)()
    assert events == ["enter", "exit"]


def test_one_line_suite_bodies_do_not_glue():
    """one-line suites inside nested defs/classes expand to proper blocks
    — the reads-only walker used to emit the raw small
    statements of a SimpleStatementSuite directly into an IndentedBlock,
    gluing the next statement onto the same line."""

    def kernel():
        log = []

        class Cm:
            def __enter__(self):
                log.append("enter")
                return 1  # noqa: E702 - one-line suite is the subject

            def __exit__(self, *a):
                log.append("exit")
                return False  # noqa: E702 - one-line suite is the subject

        with Cm() as _b:
            pass
        return log

    assert collector.instrument(kernel)() == kernel()

    def kernel2(scale):
        def helper(v):
            return v * scale

        return helper(2)

    assert collector.instrument(kernel2)(3) == kernel2(3)


# ------------------------------------------------------ staged wiring ----
# The trace interpreter drives a language Frame, and every instrumented
# construct installs a Construct object. The recording Frame below is the language-plugin
# extension pattern in miniature — host-kind Construct subclasses carrying
# recording hooks behind the interpreter-owned public routing.


class _RecMarker:
    pass


class _RecIfConstruct(IfConstruct):
    """A staged if construct: records enter_else/close."""

    def __init__(self) -> None:
        super().__init__(guard=True)
        self.elsed = 0
        self.closed = 0

    def is_staged(self) -> bool:
        return True

    def then_active(self) -> bool:
        return True

    def else_active(self) -> bool:
        return True

    def enter_else(self) -> None:
        self.elsed += 1

    def close(self) -> CloseResult:
        self.closed += 1
        return Closed()


class _RecLoopConstruct(ForConstruct):
    """A staged loop construct: records transfers and close."""

    def __init__(self) -> None:
        super().__init__(())
        self.breaks = 0
        self.continues = 0
        self.closed = 0

    def is_staged(self) -> bool:
        return True

    def iterations(self):
        return ["item"]

    def on_break(self, rt) -> None:
        self.breaks += 1

    def on_continue(self, rt) -> None:
        self.continues += 1

    def close(self) -> CloseResult:
        self.closed += 1
        return Closed()


class _RecWhileConstruct(WhileConstruct):
    """A while construct making a lazy staging decision: activates on a Marker predicate."""

    def __init__(self) -> None:
        super().__init__()
        self.activations = 0
        self.closed = 0

    def _activate(self, pred) -> bool:
        if isinstance(pred, _RecMarker):
            self.activations += 1
            return True
        return False

    def close(self) -> CloseResult:
        if self.is_staged():
            self.closed += 1
        return Closed()


class _RecWithConstruct(WithConstruct):
    """A staged with construct: the adapter drives enter/close."""

    def __init__(self, target) -> None:
        super().__init__(cm=None)
        self._target = target
        self.closed = 0

    def is_staged(self) -> bool:
        return True

    def enter(self):
        return self._target

    def close(self, exception_triple=None) -> CloseResult:
        self.closed += 1
        return Closed()


class _RecordingFrame(NullFrame):
    """Every ``_RecMarker`` is staged; the Frame hands out recording
    Constructs the interpreter drives back."""

    def __init__(self) -> None:
        self.for_: list = []
        self.whiles: list = []
        self.withs: list = []
        self.ifs: list = []

    def create_if_construct(self, interpreter, predicate, span=None):
        if isinstance(predicate, _RecMarker):
            construct = _RecIfConstruct()
            self.ifs.append((predicate, construct))
            return construct
        return None

    def create_for_construct(self, interpreter, iterable, span=None):
        if isinstance(iterable, _RecMarker):
            construct = _RecLoopConstruct()
            self.for_.append((iterable, construct))
            return ForOpen(construct, "target")
        return None

    def create_while_construct(self, interpreter, predicate_thunk, span=None):
        construct = _RecWhileConstruct()
        self.whiles.append(construct)
        return construct

    def create_with_construct(self, interpreter, context, name=None, span=None, single_call_body=False):
        if isinstance(context, _RecMarker):
            construct = _RecWithConstruct("as-target")
            self.withs.append((context, construct))
            return WithOpen(construct, "as-target")
        return None


class _RecordingSema(NullSema):
    def __init__(self) -> None:
        self.frames: list[_RecordingFrame] = []

    def begin_frame(self) -> _RecordingFrame:
        frame = _RecordingFrame()
        self.frames.append(frame)
        return frame


def test_staged_begin_for_one_iteration_surface():
    rt = Interpreter(_RecordingSema())
    rt.enter_frame()
    it = _RecMarker()
    surface = rt.begin_for(it)
    assert surface == ["item"]  # the representative iteration surface
    construct = rt._construct_stack[-1]
    assert isinstance(construct, _RecLoopConstruct)
    assert rt._construct_stack[-1] is rt._construct_stack[-1]  # exact identity retained
    rt.exit_construct()
    assert construct.closed == 1
    assert rt._construct_stack == []


def test_staged_open_with_returns_target():
    rt = Interpreter(_RecordingSema())
    rt.enter_frame()
    adapter = rt.on_begin_with(_RecMarker())
    construct = rt._construct_stack[-1]
    assert isinstance(construct, _RecWithConstruct)
    # the adapter: the host ``with`` drives it — __enter__ yields
    # the as-value, __exit__ closes the exact construct and the interpreter
    # removes it (the cm protocol is the exit hook).
    assert adapter.__enter__() == "as-target"
    assert rt._construct_stack[-1] is construct
    adapter.__exit__(None, None, None)
    assert rt._construct_stack == []
    assert construct.closed == 1


def test_staged_break_continue_record():
    rt = Interpreter(_RecordingSema())
    rt.enter_frame()
    opened = rt.begin_for(_RecMarker())
    assert opened == ["item"]
    loop = rt._construct_stack[-1]
    assert isinstance(loop, _RecLoopConstruct)
    assert rt.break_() is False  # record-and-continue, not a host break
    assert rt.continue_() is False
    assert loop.breaks == 1
    assert loop.continues == 1
    rt.exit_construct()


def test_staged_break_continue_walks_past_if_construct():
    """``break_``/``continue_`` select the nearest loop construct, walking past
    a staged if construct above it — the case ``break`` inside a staged
    ``if`` inside a staged ``for``."""
    rt = Interpreter(_RecordingSema())
    rt.enter_frame()
    loop = _RecLoopConstruct()
    rt._construct_stack.append(loop)
    rt._construct_stack.append(_RecIfConstruct())  # staged if directly atop the loop
    assert rt.break_() is False  # record on the loop, not a host break
    assert loop.breaks == 1
    assert rt.continue_() is False
    assert loop.continues == 1
    rt.close()


def test_close_auto_unwinds():
    """close() LIFO-closes open scopes and frames — the success-path
    finalizer; each exact construct's ``close`` drives it."""
    rt = Interpreter(_RecordingSema())
    rt.enter_frame()
    rt.begin_for(_RecMarker())
    rt.close()
    assert rt._construct_stack == []
    assert rt._frames == []


def test_native_transfer_rejected_across_staged_construct():
    """A native break targeting a host loop across an open staged
    construct is rejected: the alternate staged path was not
    collected."""
    rt = Interpreter(_RecordingSema())
    rt.enter_frame()
    rt._construct_stack.append(ForConstruct([1, 2]))  # a host loop
    rt._construct_stack.append(_RecIfConstruct())  # a staged if inside it
    with pytest.raises(RuntimeError, match="across an open staged Construct"):
        rt.break_()
    rt.close()


def test_null_frame_surface_is_inert():
    """The host-path surface: every create_* declines, bind declines,
    the logical ops keep host semantics (direct calls)."""
    rt = Interpreter(NullSema())
    frame = NullFrame()
    assert isinstance(frame.bind(rt, "x", 5), HostBinding)
    frame.expression_statement(rt, 5)
    frame.return_(rt, [5])
    assert frame.create_if_construct(rt, _RecMarker()) is None
    assert frame.create_for_construct(rt, _RecMarker()) is None
    assert frame.create_while_construct(rt, lambda: _RecMarker()) is None
    assert frame.create_with_construct(rt, _RecMarker()) is None
    assert frame.logical_and(1, 0) == 0
    assert frame.logical_or(0, 2) == 2
    assert frame.logical_not(1) is False
    assert frame.finish() is None
