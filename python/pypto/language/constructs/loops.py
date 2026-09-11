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
from typing import TYPE_CHECKING, Generic, TypeVar, Union

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
        """
        self.start = start
        self.stop = stop
        self.step = step
        self.pipeline_stages = pipeline_stages
        self.kind = kind
        self.attrs = attrs
        self.current = start
        self._begun: bool = False  # whether begin fired (never-begun no-op for auto-unwind)

    def __trace_iter__(self, ctx: TraceContext) -> Scalar:
        """Return a loop-target binding proxy through the trace-iter hook.

        `Frame` recognizes this object by protocol membership and calls the
        hook with the tracing context. The returned ``_LoopTargetBinding``
        defers ``for_loop`` construction until its target binding reaches the
        exact PtoForConstruct.

        Loop-carried channels are stated through ``pl.intrin.carry`` (the
        assembly dialect) before the loop; the round-trip
        ``init_values=`` spelling is printer/parser-owned text.
        """
        del ctx  # the context is consumed by the binding proxy's __trace_bind__, not here
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
        if self.kind == "pipeline":
            # ForKind.Pipeline's replication depth rides in the ForStmt attrs
            # (lower_pipeline_loops: kind == Pipeline ⇔ pipeline_stages attr
            # present). The pl.pipeline() builder (dsl_api) validates stage >= 1,
            # so pipeline_stages is always set here.
            attrs = dict(self.attrs) if self.attrs else {}
            attrs["pipeline_stages"] = self.pipeline_stages
        else:
            attrs = self.attrs
        loop_var = builder.var(name, _ir.ScalarType(DataType.INDEX))
        kind = {
            "parallel": _ir.ForKind.Parallel,
            "unroll": _ir.ForKind.Unroll,
            "pipeline": _ir.ForKind.Pipeline,
        }.get(self.kind, _ir.ForKind.Sequential)
        builder.begin_for_loop(
            loop_var,
            _unwrap_bound(self.start),
            _unwrap_bound(self.stop),
            _unwrap_bound(self.step),
            kind=kind,
            attrs=attrs,
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
) -> RangeIterator[Scalar]:
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
    return RangeIterator[Scalar](
        stop, start, step, pipeline_stages=pipeline_stages, kind=func_name, attrs=attrs
    )


def range(
    *args: RangeArg,
    attrs: _Attrs = None,
) -> RangeIterator[Scalar]:
    """Create a range iterator for for loops.

    For software pipelining (body replication for ping-pong buffering), use
    ``pl.pipeline(N, stage=F)`` instead — it is a sibling loop iterator.
    Loop-carried channels are stated through ``pl.intrin.carry`` before the
    loop (assembly dialect); the ``init_values=`` spelling exists only in
    printer/parser-owned round-trip text.

    Args can be int literals or Scalar variables:
        for i in pl.range(n):  # n is pl.Scalar[pl.INT64]
        for i in pl.range(0, n, 1):
        for i in pl.range(n * 2 + 1):

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument can be an int literal or a pl.Scalar value.
        attrs: Machine-only ForStmt attrs round-tripped through the printer
            (e.g. ``pipeline_membership``) — carried to ``begin_for_loop``.
    """
    return _make_range_iterator(
        *args,
        func_name="range",
        attrs=attrs,
    )


def parallel(
    *args: RangeArg,
    attrs: _Attrs = None,
) -> RangeIterator[Scalar]:
    """Create a parallel range iterator for parallel for loops.

    Host execution behaves identically to range(). The trace iterable emits a
    ``ForStmt`` with ``ForKind.Parallel`` (vs. ``Sequential`` for
    ``pl.range``); the scheduler assigns cores/blocks at lowering.

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument can be an int literal or a pl.Scalar value.
        attrs: Machine-only ForStmt attrs round-tripped through the printer.
    """
    return _make_range_iterator(*args, func_name="parallel", attrs=attrs)


def unroll(
    *args: RangeArg,
    attrs: _Attrs = None,
) -> RangeIterator[Scalar]:
    """Create an unroll range iterator for compile-time loop unrolling.

    Host execution behaves identically to range(). The distinction is used by
    the parser to emit ForKind.Unroll instead of ForKind.Sequential.

    Args:
        *args: Positional arguments (stop) or (start, stop) or (start, stop, step).
            Each argument must be an int literal (compile-time constant).
        attrs: Machine-only ForStmt attrs round-tripped through the printer.

    Returns:
        RangeIterator yielding loop variable (Scalar)

    Examples:
        >>> for i in pl.unroll(4):
        ...     x = pl.add(x, 1.0)
        >>> for i in pl.unroll(0, 6, 2):
        ...     x = pl.add(x, i)
    """
    return _make_range_iterator(*args, func_name="unroll", attrs=attrs)


def pipeline(
    *args: RangeArg,
    stage: int,
    attrs: _Attrs = None,
) -> RangeIterator[Scalar]:
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
    )


__all__ = [
    "range",
    "parallel",
    "unroll",
    "pipeline",
    "RangeIterator",
    "RangeArg",
]
