# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared range-based loop constructs for the PyPTO language."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union, overload

if TYPE_CHECKING:
    from pypto.ir.builder import IRBuilder
    from pypto.language.protocol import PtoForConstruct, TraceContext, _ForDescriptor
    from pypto.language.typing import Scalar

from pypto.language.typing import Scalar
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir

# Range argument type: int literal or Scalar variable
RangeArg = Union[int, "Scalar"]

# ForStmt attrs ride through to begin_for_loop; dict value types are the
# machine-only attr surface (an open mapping).
_Attrs = dict[str, typing.Any] | None


T = TypeVar("T")

# Loop-carried init_value channels (the printer/parser-owned round-trip
# spelling; the tracing frontend binds each channel before the loop and
# rebinds it at the end of the body).
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")

# The while-iterator's carried tuple (the parser lane's ``pl.while_``).
W = TypeVar("W")


def infer_loop_var_dtype(*bounds: int | _ir.Expr | None) -> DataType:
    """Infer the loop-var dtype from ``pl.range`` bounds (shared by both lanes).

    The default is INDEX (bare Python ints stay INDEX); explicitly typed INT64
    ConstInt bounds rebuild an INT64 loop var so a printed loop re-parses with
    its original loop-var dtype (print→parse roundtrip fidelity), and any
    other typed ConstInt bound wins outright. A mixed INDEX/INT64 set stays
    INDEX. Non-constant bounds (runtime scalar reads) do not constrain the
    loop var — the parser lane and the tracing descriptor apply the same
    rule so identical bounds produce the identical loop-var dtype per lane.
    """
    dtype = DataType.INDEX
    saw_index_bound = False
    saw_int64_bound = False
    for bound in bounds:
        if isinstance(bound, _ir.ConstInt):
            if bound.dtype == DataType.INDEX:
                saw_index_bound = True
            elif bound.dtype == DataType.INT64:
                saw_int64_bound = True
            else:
                dtype = bound.dtype
                break
    if dtype == DataType.INDEX and saw_int64_bound and not saw_index_bound:
        dtype = DataType.INT64
    return dtype


def _unwrap_bound(bound: RangeArg) -> int | _ir.Expr:
    """A ``pl.range`` bound: a Scalar facade unwraps to its Expr; a host
    int passes through (``builder.begin_for_loop`` materializes it as
    ConstInt(INDEX)). The trace iterable unwraps its own fields (the
    structural analogue of an operation unwrapping its operands)."""
    if isinstance(bound, Scalar):
        return bound.unwrap()
    return bound


class _LoopTargetBinding(Scalar):
    """A loop-target binding proxy carrying the ``__trace_bind__`` hook.

    The proxy receives the source target name and asks the exact
    PtoForConstruct attached by `Frame` to activate the loop. It retains that
    Construct rather than rediscovering active state through Sema. Before the
    target binding it wraps a placeholder Scalar; after the binding the
    returned loop variable is the staged value stored under the source name.

    The hook is one-shot: a later binding of that value (``y = i``) follows the
    ordinary staged-binding path rather than constructing the loop again. The
    proxy owns its consumed state; Sema never inspects a value flag.
    """

    def __init__(self, descriptor: _ForDescriptor) -> None:
        super().__init__(expr=_ir.ConstInt(0, DataType.INDEX, _ir.Span.unknown()))
        self._descriptor = descriptor
        # Attached by the Frame at create_for_construct: the exact PtoForConstruct
        # that owns this loop's deferred region construction.
        self._construct: PtoForConstruct | None = None
        self._consumed: bool = False

    def attach_construct(self, construct: PtoForConstruct) -> None:
        """The Frame attaches the exact PtoForConstruct owning this loop."""
        self._construct = construct

    def __trace_bind__(self, ctx: TraceContext, name: str) -> _ir.Var | None:
        """Bind the loop target and activate the exact PtoForConstruct.

        The first target binding (``rt.vars.i = __x``) calls
        ``activate_target`` with the tracing context; the Construct records
        its dynamic open state. A consumed proxy declines (returns ``None``),
        so the Frame's ``_default_bind`` takes the ordinary staged-binding
        path.
        """
        if self._consumed:
            return None  # one-shot — decline → _default_bind (ordinary staged binding)
        if self._construct is None:
            raise RuntimeError(
                "the loop-target binding proxy has no attached PtoForConstruct — "
                "the loop was not opened through the Frame"
            )
        loop_var = self._construct.activate_target(ctx, name)
        self._consumed = True  # value-owned
        return loop_var


class RangeIterator(Generic[T]):
    """Trace iterable for ``pl.range()`` and related loop operations.

    Builder access occurs only through trace-protocol hooks; the iterable has
    no ambient builder read and remains inert under NullSema outside a trace.
    """

    def __init__(
        self,
        stop: RangeArg,
        start: RangeArg = 0,
        step: RangeArg = 1,
        pipeline_stages: int | None = None,
        kind: str = "range",
        attrs: _Attrs = None,
        init_values: tuple[typing.Any, ...] | None = None,
    ):
        """Initialize range iterator.

        Args:
            stop: Stop value (int or Scalar)
            kind: The trace iterable's origin — ``range`` (sequential),
                ``parallel``, ``unroll``, or ``pipeline``. The tracing path
                tags it so Sema can select the ForKind variant; the AST
                frontend reads the same origin from source.
            start: Start value (default 0, int or Scalar)
            step: Step value (default 1, int or Scalar)
            pipeline_stages: Software-pipelining depth — replicates the body this
                many times per outer iteration (None = no pipelining). Only set by
                pl.pipeline(); validated by the parser.
            attrs: Machine-only ForStmt attrs round-tripped through the printer
                (e.g. ``pipeline_membership``) — carried to ``begin_for_loop``.
            init_values: Loop-carried initial values — the printer/parser-owned
                round-trip spelling of what the tracing frontend states by
                binding each channel before the loop and rebinding it at the
                end of the body. Host iteration yields
                ``(loop_var, init_values)``; the trace path rejects it (see
                ``begin``) rather than silently dropping the channels.
        """
        self.start = start
        self.stop = stop
        self.step = step
        self.pipeline_stages = pipeline_stages
        self.kind = kind
        self.attrs = attrs
        self.init_values = init_values
        self.current = start
        self._begun: bool = False  # whether begin fired (never-begun no-op for auto-unwind)

    def __trace_iter__(self, ctx: TraceContext) -> Scalar:
        """Return a loop-target binding proxy through the trace-iter hook.

        `Frame` recognizes this object by protocol membership and calls the
        hook with the tracing context. The returned ``_LoopTargetBinding``
        defers ``for_loop`` construction until its target binding reaches the
        exact PtoForConstruct.

        Loop-carried channels are stated by binding each channel before the
        loop and rebinding it at the end of the body; the round-trip
        ``init_values=`` spelling is printer/parser-owned text (its host
        execution parity is kept for the parser-era suites — inside a trace
        it is rejected here, before the tuple-target binding proxy can
        mislead with an unpack error).
        """
        del ctx  # the context is consumed by the binding proxy's __trace_bind__, not here
        if self.init_values:
            raise TypeError(
                f"pl.{self.kind}(init_values=...) is the printer/parser-owned round-trip "
                "spelling of loop-carried channels — the tracing frontend binds each "
                "channel before the loop and rebinds it at the end of the body"
            )
        return _LoopTargetBinding(self)

    def begin(
        self,
        builder: IRBuilder,
        name: str,
        already_in_incore: bool = False,
        sticky_dump_vars: list[_ir.Var] | None = None,
    ) -> _ir.Var:
        """Construct the ``for_loop`` at the first target binding.

        The trace iterable owns construction through the builder exposed by
        the tracing context and unwraps its own ``int | Scalar`` bounds, the
        structural analogue of an operation unwrapping its operands. Return
        the loop-variable `ir.Var` that the Frame registers. A plain loop
        never merges ``sticky_dump_vars`` (no scope entry — protocol
        uniformity with the scope-carrying trace iterables)."""
        del sticky_dump_vars  # a plain for_loop has no dump_vars attr
        if self.init_values:
            raise TypeError(
                f"pl.{self.kind}(init_values=...) is the printer/parser-owned round-trip "
                "spelling of loop-carried channels — the tracing frontend binds each "
                "channel before the loop and rebinds it at the end of the body"
            )
        if self.kind == "pipeline":
            # ForKind.Pipeline's replication depth rides in the ForStmt attrs
            # (lower_pipeline_loops: kind == Pipeline ⇔ pipeline_stages attr
            # present). The pl.pipeline() entry validates stage >= 1
            # (``_pin_stage``), so pipeline_stages is always set here.
            attrs = dict(self.attrs) if self.attrs else {}
            attrs["pipeline_stages"] = self.pipeline_stages
        else:
            attrs = self.attrs
        start = _unwrap_bound(self.start)
        stop = _unwrap_bound(self.stop)
        step = _unwrap_bound(self.step)
        loop_var = builder.var(name, _ir.ScalarType(infer_loop_var_dtype(start, stop, step)))
        kind = {
            "parallel": _ir.ForKind.Parallel,
            "unroll": _ir.ForKind.Unroll,
            "pipeline": _ir.ForKind.Pipeline,
        }.get(self.kind, _ir.ForKind.Sequential)
        # pl.hardware_loop() is a Sequential loop that additionally *requires*
        # hardware qualification; the parser spells the same requirement, so the
        # tracing lane must carry it too.
        hardware_loop_policy = (
            _ir.HardwareLoopPolicy.Require if self.kind == "hardware_loop" else _ir.HardwareLoopPolicy.Auto
        )
        builder.begin_for_loop(
            loop_var,
            start,
            stop,
            step,
            kind=kind,
            attrs=attrs,
            hardware_loop_policy=hardware_loop_policy,
        )
        self._begun = True
        return loop_var

    def __trace_exit__(self, ctx: TraceContext) -> None:
        """Finalize the ``for_loop`` through the tracing context.

        This is the exact PtoForConstruct close. It is a no-op when the loop
        was never begun: target binding defers construction, so a bind-time
        validation error cannot leave a builder context open; the builder
        still rejects any malformed unbalanced trace at ``end_function``.
        """
        if self._begun:
            ctx.builder.end_for_loop()
            self._begun = False

    def __iter__(self) -> RangeIterator[T]:
        """Return iterator."""
        return self

    def _host_next(self) -> object:
        """Advance the host-mode counter (the fallback iteration surface).

        The tracing interpreter never drives this: the staged ``for`` binds
        the Scalar loop variable through ``__trace_iter__``'s seed. Host
        iteration yields the raw counter (``int`` for literal bounds).
        """
        if self.current >= self.stop:
            raise StopIteration

        value = self.current
        self.current += self.step
        # Host parity with the AST frontend's Pattern A: with init_values the
        # iteration yields ``(loop_var, (iter_args...))``; without, the bare
        # counter (``int`` for literal bounds).
        if self.init_values:
            return (value, self.init_values)
        return value

    def __next__(self) -> T:
        """Yield one iteration value typed as the traced loop variable.

        The annotation speaks for the *traced* contract — a ``for`` over this
        iterable binds a ``T`` (``Scalar`` from the factories) — while the
        host counter underneath produces ``RangeArg``. The cast bridges the
        two; tracing never executes it.
        """
        return typing.cast(T, self._host_next())


def _pin_stage(stage: object) -> int:
    """Pin ``pl.pipeline(stage=...)`` to a positive int (bool rejected)."""
    if isinstance(stage, bool) or not isinstance(stage, int) or stage < 1:
        raise ValueError(f"pipeline() stage must be a positive integer, got {stage!r}")
    return stage


def _make_range_iterator(
    *args: RangeArg,
    pipeline_stages: int | None = None,
    func_name: str = "range",
    attrs: _Attrs = None,
    init_values: tuple[typing.Any, ...] | None = None,
) -> RangeIterator[typing.Any]:
    """Shared implementation for range(), parallel(), unroll(), and pipeline().

    ``pipeline_stages`` arrives pre-pinned from the public entry
    (``pipeline()`` validates the user's ``stage=``).
    """
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        start, stop, step = args[0], args[1], 1
    elif len(args) == 3:
        start, stop, step = args[0], args[1], args[2]
    else:
        raise ValueError(f"{func_name}() takes 1 to 3 positional arguments")
    return RangeIterator(
        stop,
        start,
        step,
        pipeline_stages=pipeline_stages,
        kind=func_name,
        attrs=attrs,
        init_values=init_values,
    )


@overload
def range(*args: RangeArg, init_values: None = None, attrs: _Attrs = None) -> RangeIterator[Scalar]: ...


@overload
def range(
    *args: RangeArg, init_values: tuple[T1], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1]]]: ...


@overload
def range(
    *args: RangeArg, init_values: tuple[T1, T2], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2]]]: ...


@overload
def range(
    *args: RangeArg, init_values: tuple[T1, T2, T3], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3]]]: ...


@overload
def range(
    *args: RangeArg, init_values: tuple[T1, T2, T3, T4], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4]]]: ...


@overload
def range(
    *args: RangeArg, init_values: tuple[T1, T2, T3, T4, T5], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4, T5]]]: ...


def range(
    *args: RangeArg,
    init_values: tuple[typing.Any, ...] | None = None,
    attrs: _Attrs = None,
) -> RangeIterator[typing.Any]:
    """Create a range iterator for for loops.

    For software pipelining (body replication for ping-pong buffering), use
    ``pl.pipeline(N, stage=F)`` instead — it is a sibling loop iterator.
    Loop-carried channels are stated by binding each channel before the
    loop and rebinding it at the end of the body; the ``init_values=``
    spelling exists only in printer/parser-owned round-trip text (and mirrors
    those channels in host execution, yielding ``(loop_var, init_values)``).

    Args can be int literals or Scalar variables:
        for i in pl.range(n):  # n is pl.Scalar[pl.INT64]
        for i in pl.range(0, n, 1):
        for i in pl.range(n * 2 + 1):

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument can be an int literal or a pl.Scalar value.
        init_values: Initial values for iteration arguments — the round-trip
            spelling of the loop-carried channels (see above).
        attrs: Machine-only ForStmt attrs round-tripped through the printer
            (e.g. ``pipeline_membership``) — carried to ``begin_for_loop``.
    """
    return _make_range_iterator(
        *args,
        func_name="range",
        attrs=attrs,
        init_values=init_values,
    )


@overload
def parallel(*args: RangeArg, init_values: None = None, attrs: _Attrs = None) -> RangeIterator[Scalar]: ...


@overload
def parallel(
    *args: RangeArg, init_values: tuple[T1], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1]]]: ...


@overload
def parallel(
    *args: RangeArg, init_values: tuple[T1, T2], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2]]]: ...


@overload
def parallel(
    *args: RangeArg, init_values: tuple[T1, T2, T3], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3]]]: ...


@overload
def parallel(
    *args: RangeArg, init_values: tuple[T1, T2, T3, T4], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4]]]: ...


@overload
def parallel(
    *args: RangeArg, init_values: tuple[T1, T2, T3, T4, T5], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4, T5]]]: ...


def parallel(
    *args: RangeArg,
    init_values: tuple[typing.Any, ...] | None = None,
    attrs: _Attrs = None,
) -> RangeIterator[typing.Any]:
    """Create a parallel range iterator for parallel for loops.

    Host execution behaves identically to range(). The trace iterable emits a
    ``ForStmt`` with ``ForKind.Parallel`` (vs. ``Sequential`` for
    ``pl.range``); the scheduler assigns cores/blocks at lowering.

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument can be an int literal or a pl.Scalar value.
        init_values: Initial values for iteration arguments — the round-trip
            spelling of the loop-carried channels (see ``pl.range``).
        attrs: Machine-only ForStmt attrs round-tripped through the printer.
    """
    return _make_range_iterator(*args, func_name="parallel", attrs=attrs, init_values=init_values)


@overload
def hardware_loop(stop: RangeArg, *, init_values: None = None) -> RangeIterator[Scalar]: ...


@overload
def hardware_loop(stop: RangeArg, *, init_values: tuple[T1]) -> RangeIterator[tuple[Scalar, tuple[T1]]]: ...


@overload
def hardware_loop(
    stop: RangeArg, *, init_values: tuple[T1, T2]
) -> RangeIterator[tuple[Scalar, tuple[T1, T2]]]: ...


@overload
def hardware_loop(
    stop: RangeArg, *, init_values: tuple[T1, T2, T3]
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3]]]: ...


@overload
def hardware_loop(
    stop: RangeArg, *, init_values: tuple[T1, T2, T3, T4]
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4]]]: ...


@overload
def hardware_loop(
    stop: RangeArg, *, init_values: tuple[T1, T2, T3, T4, T5]
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4, T5]]]: ...


def hardware_loop(
    stop: RangeArg,
    *,
    init_values: tuple[Any, ...] | None = None,
) -> RangeIterator[Scalar] | RangeIterator[tuple[Scalar, tuple[Any, ...]]]:
    """Require a Sequential hardware loop over [0, stop) with step one.

    Only valid inside an enclosing ``pl.reg_scope()``, including empty loops.
    Integer scalar stops may be dynamic without a proven range; static stops
    must fit the unsigned counter limit. Floating-point stops are rejected.
    Static qualification is distinct from final artifact evidence. Unsupported
    targets fail compilation rather than silently falling back to ordinary loops.
    Static empty loops may be removed with their initial carry values preserved.
    """
    return _make_range_iterator(stop, func_name="hardware_loop", init_values=init_values)


@overload
def unroll(*args: RangeArg, init_values: None = None, attrs: _Attrs = None) -> RangeIterator[Scalar]: ...


@overload
def unroll(
    *args: RangeArg, init_values: tuple[T1], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1]]]: ...


@overload
def unroll(
    *args: RangeArg, init_values: tuple[T1, T2], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2]]]: ...


@overload
def unroll(
    *args: RangeArg, init_values: tuple[T1, T2, T3], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3]]]: ...


@overload
def unroll(
    *args: RangeArg, init_values: tuple[T1, T2, T3, T4], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4]]]: ...


@overload
def unroll(
    *args: RangeArg, init_values: tuple[T1, T2, T3, T4, T5], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4, T5]]]: ...


def unroll(
    *args: RangeArg,
    init_values: tuple[typing.Any, ...] | None = None,
    attrs: _Attrs = None,
) -> RangeIterator[typing.Any]:
    """Create an unroll range iterator for compile-time loop unrolling.

    Host execution behaves identically to range(). The distinction is used by
    the parser to emit ForKind.Unroll instead of ForKind.Sequential.

    Explicit ``init_values`` represent transitional SSA IR for print/parse
    roundtrips (UnrollLoops requires non-SSA input: in executable kernels use
    ordinary assignments and run UnrollLoops before ConvertToSSA; the tracing
    frontend binds each channel before the loop and rebinds it at the end of
    the body).

    The trip count is capped at 1024 iterations by the ``UnrollLoops`` pass
    (``kMaxUnrollIterations``); a larger loop raises ``ValueError`` — use
    ``range()`` (a device-side loop) for larger trip counts.

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument must be an int literal (compile-time constant).
        init_values: Initial values for iteration arguments in transitional SSA IR.
        attrs: Machine-only ForStmt attrs round-tripped through the printer.

    Returns:
        If no init_values: RangeIterator yielding loop variable (Scalar)
        If init_values: RangeIterator yielding (loop_var, (iter_args...))

    Examples:
        >>> for i in pl.unroll(4):
        ...     x = pl.add(x, 1.0)
        >>> for i in pl.unroll(0, 6, 2):
        ...     x = pl.add(x, i)
    """
    return _make_range_iterator(*args, func_name="unroll", attrs=attrs, init_values=init_values)


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: None = None, attrs: _Attrs = None
) -> RangeIterator[Scalar]: ...


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: tuple[T1], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1]]]: ...


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: tuple[T1, T2], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2]]]: ...


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: tuple[T1, T2, T3], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3]]]: ...


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: tuple[T1, T2, T3, T4], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4]]]: ...


@overload
def pipeline(
    *args: RangeArg, stage: int, init_values: tuple[T1, T2, T3, T4, T5], attrs: _Attrs = None
) -> RangeIterator[tuple[Scalar, tuple[T1, T2, T3, T4, T5]]]: ...


def pipeline(
    *args: RangeArg,
    stage: int,
    init_values: tuple[typing.Any, ...] | None = None,
    attrs: _Attrs = None,
) -> RangeIterator[typing.Any]:
    """Create a software-pipelined loop iterator.

    Replicates the loop body ``stage`` times per outer iteration to enable
    ping-pong buffering. The outer loop advances in strides of ``stage * step``;
    a tail dispatch covers the remainder when the trip count is not divisible
    by ``stage``. Lowered by the ``LowerPipelineLoops`` pass at the tile level.

    Positional args match ``pl.range``: (stop) / (start, stop) / (start, stop, step).
    The ``stage`` kwarg is required and must be a positive integer.

    Args:
        *args: 1-3 positional args — same shape as ``pl.range()``.
        stage: Pipeline depth (positive integer, typically 2-4).
        init_values: Loop-carried state, same semantics as ``pl.range()``.
        attrs: Machine-only ForStmt attrs round-tripped through the printer.

    Examples:
        >>> for i in pl.pipeline(64, stage=4):
        ...     tile = pl.tile.load(...)
        ...     pl.tile.store(...)
    """
    return _make_range_iterator(
        *args,
        pipeline_stages=_pin_stage(stage),
        func_name="pipeline",
        attrs=attrs,
        init_values=init_values,
    )


class WhileIterator(Generic[W]):
    """Iterator for ``pl.while_()`` — the parser lane's while-loop spelling.

    Yields its ``init_values`` exactly once: the AST parser intercepts the
    ``for ... in pl.while_(...)`` statement and builds the ``WhileStmt``, so
    the runtime object never drives a real iteration. It exists so the name
    is importable and so a direct (erroneous) call outside a parsed body
    behaves as the single handout above.
    """

    def __init__(self, *, init_values: tuple[Any, ...] | None = None):
        """Initialize with the loop-carried initial values (required)."""
        if init_values is None:
            raise ValueError("while_() requires init_values to be specified")
        self.init_values = init_values
        self._exhausted = False

    def __iter__(self) -> WhileIterator[W]:
        """Return iterator."""
        return self

    @overload
    def __next__(self: WhileIterator[tuple[T1]]) -> tuple[T1]: ...

    @overload
    def __next__(self: WhileIterator[tuple[T1, T2]]) -> tuple[T1, T2]: ...

    @overload
    def __next__(self: WhileIterator[tuple[T1, T2, T3]]) -> tuple[T1, T2, T3]: ...

    @overload
    def __next__(self: WhileIterator[tuple[T1, T2, T3, T4]]) -> tuple[T1, T2, T3, T4]: ...

    @overload
    def __next__(self: WhileIterator[tuple[T1, T2, T3, T4, T5]]) -> tuple[T1, T2, T3, T4, T5]: ...

    @overload
    def __next__(self: WhileIterator[tuple[Any, ...]]) -> tuple[Any, ...]: ...

    def __next__(self) -> tuple[Any, ...]:
        """Hand out the init values once — the parser owns the real loop."""
        if self._exhausted:
            raise StopIteration
        self._exhausted = True
        return self.init_values


def _reject_inside_trace(name: str) -> None:
    """Fail loudly when a parser-only loop spelling runs inside a trace.

    ``pl.while_`` / ``pl.cond`` are resolved syntactically by the AST parser;
    the tracing dialect spells the same constructs as ordinary Python
    ``while``. A runtime call with an active trace frame therefore means a
    traced body used the parser spelling, which the collector would
    otherwise execute as host code (one bogus iteration) instead of staging.
    """
    from pypto.language.ambient import active_frame  # noqa: PLC0415 — cycle-free laziness

    if active_frame() is not None:
        raise TypeError(
            f"pl.{name} is parser-frontend syntax and cannot run inside a trace — "
            "the tracing dialect uses ordinary Python `while` (keep this kernel on "
            "@pl.jit / @pl.function, or rewrite the loop)"
        )


@overload
def while_(*, init_values: tuple[T1]) -> WhileIterator[tuple[T1]]: ...


@overload
def while_(*, init_values: tuple[T1, T2]) -> WhileIterator[tuple[T1, T2]]: ...


@overload
def while_(*, init_values: tuple[T1, T2, T3]) -> WhileIterator[tuple[T1, T2, T3]]: ...


@overload
def while_(*, init_values: tuple[T1, T2, T3, T4]) -> WhileIterator[tuple[T1, T2, T3, T4]]: ...


@overload
def while_(*, init_values: tuple[T1, T2, T3, T4, T5]) -> WhileIterator[tuple[T1, T2, T3, T4, T5]]: ...


def while_(*, init_values: tuple[Any, ...] | None = None) -> WhileIterator[tuple[Any, ...]]:
    """Create a while iterator for the parser lane's while loops.

    Always requires ``init_values`` to specify loop-carried state. The loop
    condition is the first statement of the body, via :func:`cond`. The AST
    parser intercepts this call; the runtime object only hands out the init
    values once (see :class:`WhileIterator`).

    Args:
        init_values: Initial values for the iteration arguments (required).

    Returns:
        A :class:`WhileIterator` yielding the tuple of iter args once.

    Raises:
        ValueError: If ``init_values`` is not provided.
        TypeError: If called inside a traced kernel body.
    """
    _reject_inside_trace("while_")
    if init_values is None:
        raise ValueError("while_() requires init_values to be specified")
    return WhileIterator(init_values=init_values)


def cond(condition: bool | Scalar) -> None:
    """Specify the condition of a ``pl.while_()`` loop (parser lane).

    Must be the first statement in a ``pl.while_()`` loop body. Purely
    syntactic — the parser extracts the condition onto the ``WhileStmt``;
    the runtime call is a no-op.

    Args:
        condition: While loop condition (bool literal or Scalar variable).

    Raises:
        TypeError: If called inside a traced kernel body.
    """
    _reject_inside_trace("cond")


__all__ = [
    "range",
    "hardware_loop",
    "parallel",
    "unroll",
    "pipeline",
    "while_",
    "cond",
    "WhileIterator",
    "RangeIterator",
    "RangeArg",
    "infer_loop_var_dtype",
]
