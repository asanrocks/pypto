"""Acceptance for the construct-recipe authoring tier (``@pl.authoring.construct``).

``blocked_spmd`` below is written exactly the way a kernel author outside the
frontend would write it — a recipe composed over ``pl.spmd``, with no import
from the tracer. The parity tests are the acceptance gates of the refactoring:

* **IR parity** — a recipe-authored construct and the hand-written ``pl.spmd``
  grid it replaces must trace to structurally identical IR (scopes, attrs,
  carrier, statements).
* **Contract** — the driver enforces the recipe discipline (one yield, paired
  regions, ``as`` handling, no host use) loudly enough that no private
  convention is needed to author one.
"""

import types
from typing import TypeAlias, cast

import pypto.language as pl
import pytest
from pypto.language.ambient import reset_active_frame, set_active_frame
from pypto.language.diagnostics import InvalidOperationError
from pypto.language.tracing.collector.driver import instrument
from pypto.language.tracing.semantics import PtoSema
from pypto.pypto_core import ir

F32 = pl.FP32


T8x8: TypeAlias = pl.Tensor[[8, 8], F32]


T8x16: TypeAlias = pl.Tensor[[8, 16], F32]


OUT8x8: TypeAlias = pl.Out[T8x8]


OUT8x16: TypeAlias = pl.Out[T8x16]


class _BlockAxis:
    """One grid axis: the staged tile window of this axis' block coordinate."""

    def __init__(self, tile_size: int, coord) -> None:
        self._tile = tile_size
        self._coord = coord  # () -> pl.Scalar: this axis' block coordinate

    @property
    def offset(self) -> pl.Scalar:
        """Element offset of this axis' tile (a fresh block-index read per access)."""
        coord = self._coord()
        return coord if self._tile == 1 else coord * self._tile

    @property
    def size(self) -> int:
        """Static tile capacity along this axis."""
        return self._tile


class _TensorRegion:
    """A block's projection onto a tensor: the mapped axes' tile windows."""

    def __init__(self, tensor, projections) -> None:
        self._tensor = tensor
        self._projections = tuple(projections)  # (axis descriptor, tensor axis)

    def _rank(self) -> int:
        return len(self._tensor.unwrap().type.shape)

    def _windows(self):
        shape = self._tensor.unwrap().type.shape
        rank = len(shape)
        mapped = {}
        for axis, tensor_axis in self._projections:
            if not 0 <= tensor_axis < rank:
                raise ValueError(f"blocked_spmd: region tensor axis {tensor_axis} out of rank {rank}")
            mapped[tensor_axis] = (axis.offset, axis.size)
        offsets, shapes = [], []
        for tensor_axis in range(rank):
            if tensor_axis in mapped:
                offset, size = mapped[tensor_axis]
                offsets.append(offset)
                shapes.append(size)
            else:
                extent = shape[tensor_axis]
                if isinstance(extent, bool) or not isinstance(extent, int):
                    # A static shape prints back as ConstInt, not a Python int.
                    extent = getattr(extent, "value", None)
                    if not isinstance(extent, int) or isinstance(extent, bool):
                        raise ValueError(
                            f"blocked_spmd: region axis {tensor_axis} has a dynamic extent — "
                            "map it onto a grid axis or slice it first"
                        )
                offsets.append(0)
                shapes.append(extent)
        return offsets, shapes

    def load(self, **kw) -> pl.Tile:
        offsets, shapes = self._windows()
        return pl.load(self._tensor, offsets, shapes, **kw)

    def store(self, tile: pl.Tile, **kw):
        offsets, _ = self._windows()
        return pl.store(tile, offsets, self._tensor, **kw)


class _BlockGrid(pl.authoring.ConstructHandle):
    """The ``as``-target of a ``blocked_spmd`` scope: axes, regions, and the TaskId."""

    def __init__(self, blocks, tile, tid_var) -> None:
        if len(blocks) == 1:
            coords = (lambda: pl.get_block_idx(),)
        else:
            along_axis1 = blocks[1]
            coords = (
                lambda: pl.get_block_idx() // along_axis1,
                lambda: pl.get_block_idx() % along_axis1,
            )
        self._axes = tuple(_BlockAxis(t, c) for t, c in zip(tile, coords))
        self._tid = pl.Scalar(expr=tid_var) if tid_var is not None else None

    @property
    def axes(self) -> tuple:
        return self._axes

    @property
    def task_id(self) -> pl.Scalar:
        if self._tid is None:
            raise TypeError("blocked_spmd: no dispatch TaskId — bind the construct with `as`")
        return self._tid

    def region(self, tensor, *, axes=None) -> _TensorRegion:
        if axes is None:
            axes = tuple(range(len(self._axes)))
        return _TensorRegion(tensor, zip(self._axes, axes))


def _trace(
    kernel: types.FunctionType,
    annotations: dict[str, object],
    *,
    directions: dict[str, ir.ParamDirection] | None = None,
) -> ir.Function:
    """Trace ``kernel`` through the collector with a fresh one-shot PtoSema."""
    sema = PtoSema(name=kernel.__name__)
    facades = sema.bind_parameters(annotations, directions=directions or {}, scope=kernel.__globals__)
    frame = sema.frame
    assert frame is not None
    token = set_active_frame(frame)
    try:
        instrument(kernel, sema)(*facades)
    finally:
        reset_active_frame(token)
    return sema.result()


class _ScopeCollect(ir.IRVisitor):
    """Collect the scope statements of one body in traversal order."""

    def __init__(self) -> None:
        super().__init__()
        self.scopes: list[ir.Stmt] = []

    def visit_in_core_scope_stmt(self, op: ir.InCoreScopeStmt) -> None:
        self.scopes.append(op)
        super().visit_in_core_scope_stmt(op)

    def visit_spmd_scope_stmt(self, op: ir.SpmdScopeStmt) -> None:
        self.scopes.append(op)
        super().visit_spmd_scope_stmt(op)

    def visit_runtime_scope_stmt(self, op: ir.RuntimeScopeStmt) -> None:
        self.scopes.append(op)
        super().visit_runtime_scope_stmt(op)

    def visit_cluster_scope_stmt(self, op: ir.ClusterScopeStmt) -> None:
        self.scopes.append(op)
        super().visit_cluster_scope_stmt(op)

    def visit_graph_scope_stmt(self, op: ir.GraphScopeStmt) -> None:
        self.scopes.append(op)
        super().visit_graph_scope_stmt(op)


def _collect_scopes(fn: ir.Function) -> list[ir.Stmt]:
    assert fn.body is not None
    collector = _ScopeCollect()
    collector.visit_stmt(fn.body)
    return collector.scopes


class _SplitAivCollect(ir.IRVisitor):
    """Collect the InCore and SplitAiv scopes of one body, in traversal order."""

    def __init__(self) -> None:
        super().__init__()
        self.scopes: list[ir.Stmt] = []

    def visit_in_core_scope_stmt(self, op: ir.InCoreScopeStmt) -> None:
        self.scopes.append(op)
        super().visit_in_core_scope_stmt(op)

    def visit_split_aiv_scope_stmt(self, op: ir.SplitAivScopeStmt) -> None:
        self.scopes.append(op)
        super().visit_split_aiv_scope_stmt(op)


def _collect_split_aiv(fn: ir.Function) -> list[ir.Stmt]:
    assert fn.body is not None
    collector = _SplitAivCollect()
    collector.visit_stmt(fn.body)
    return collector.scopes


def test_construct_rejects_a_function_without_a_yield():
    """The decorator validates the recipe shape at authoring time, not first use."""

    with pytest.raises(TypeError, match="no yield"):

        @pl.authoring.construct
        def not_a_generator(ctx):
            return None


@pl.authoring.construct
def core_group(ctx, *, name_hint="", deps=None):
    """A minimal InCore scope recipe — the open_scope/capture_task_id primitives.

    Authored the way ``pl.at`` itself now is: attrs in canonical order, the
    TaskId capture, then the deferred scope. The parity test below is the
    conversion's acceptance gate — the recipe and ``pl.at`` must be
    indistinguishable in IR.
    """
    scope_attrs: list[tuple[str, object]] = []
    if deps:
        edges = ctx.lower_deps(deps, "core_group")
        if edges:
            scope_attrs.append(("manual_dep_edges", edges))
    as_binding = None
    if ctx.binding_name:
        as_binding, tid_attr = ctx.capture_task_id(ctx.binding_name)
        scope_attrs.append(tid_attr)
    region = ctx.open_scope(ir.ScopeKind.InCore, attrs=scope_attrs, split=None, name_hint=name_hint)
    yield as_binding
    ctx.exit(region)


def test_the_scope_recipe_takes_cache_policy_declarations():
    """A leading pl.set_cache_policy joins the recipe scope's attrs (the deferred contract)."""

    def at_recipe(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
        with core_group(name_hint="fuse"):
            pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
            t = pl.load(x, [0, 0], [8, 8])
            pl.store(pl.add(t, t), [0, 0], out)

    fn = _trace(at_recipe, {"x": T8x8, "out": OUT8x8}, directions={"out": ir.ParamDirection.Out})
    scopes = _collect_scopes(fn)
    assert [type(op) for op in scopes] == [ir.InCoreScopeStmt]
    incore = scopes[0]
    attrs = dict(getattr(incore, "attrs", []) or [])
    assert "cache_policy_vars" in attrs


@pl.authoring.construct
def grid_loop(ctx, blocks, *, name_hint=""):
    """A for-spelling recipe — `for i in grid_loop(n)` is `for i in pl.spmd(n)`.

    Authored the way ``pl.spmd``'s for-form now is: mint the loop variable,
    open the dispatch scope, open the carrier with the block-index prologue,
    and yield the loop variable. No import below ``pypto.language``."""
    loop_var = ctx.loop_var()
    spmd_region = ctx.open_scope(
        ir.ScopeKind.Spmd,
        attrs=[],
        defer=False,
        core_num=blocks,
        sync_start=False,
        name_hint=f"{name_hint}_spmd",
    )
    carrier = ctx.open_carrier(
        ir.ScopeKind.InCore,
        attrs=[],
        # The carrier — not the dispatch scope around it — is the recipe's
        # structural surface (the loop variable's prologue and the body's
        # statements both land inside it).
        primary=True,
        flush_on_emit=True,
        split=None,
        name_hint=name_hint,
        prologue=[(loop_var, pl.tile.get_block_idx().unwrap())],
    )
    yield loop_var
    ctx.exit(carrier)
    ctx.exit(spmd_region)


def test_a_with_spelling_recipe_rejects_the_for_spelling():
    """A recipe that yields a handle or a capture binding serves `with` only — loud in a `for`."""

    def kernel(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
        for i in core_group(name_hint="fuse"):  # noqa: B007 — the loop shape is what is rejected
            t = pl.load(x, [i, 0], [2, 8])
            pl.store(t, [0, 0], out)

    with pytest.raises(TypeError, match="loop variable"):
        _trace(kernel, {"x": T8x8, "out": OUT8x8}, directions={"out": ir.ParamDirection.Out})


def test_a_for_recipe_without_a_primary_carrier_is_loud():
    """A for-recipe whose structural surface is not the carrier mis-lowers silently — rejected."""

    @pl.authoring.construct
    def no_primary_carrier(ctx, blocks):
        loop_var = ctx.loop_var()
        region = ctx.open_scope(ir.ScopeKind.Spmd, attrs=[], core_num=blocks)
        carrier = ctx.open_carrier(
            ir.ScopeKind.InCore,
            attrs=[],
            flush_on_emit=True,
            prologue=[(loop_var, pl.tile.get_block_idx().unwrap())],
        )
        yield loop_var
        ctx.exit(carrier)
        ctx.exit(region)

    def kernel(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
        for i in no_primary_carrier(4):  # noqa: B007 — the structural surface is what is rejected
            pl.store(pl.load(x, [i, 0], [2, 8]), [i, 0], out)

    with pytest.raises(TypeError, match="primary=True"):
        _trace(kernel, {"x": T8x8, "out": OUT8x8}, directions={"out": ir.ParamDirection.Out})


def test_the_for_spelling_rejects_host_use():
    """Outside a trace the `for` spelling is loud, not a silent zero-iteration loop."""
    with pytest.raises(RuntimeError, match="decorated"):
        for i in grid_loop(4):  # noqa: B007 — the host iteration is what is rejected
            del i


@pl.authoring.construct
def counted(ctx, n):
    """A recipe-authored sequential loop — `for i in counted(n)` is `pl.range(n)`."""
    with ctx.open_loop(0, n, 1) as i:
        yield i


class Accumulator(pl.authoring.ConstructHandle):
    """The carried-channel handle: the body reads `.value`, hands back the next.

    The channel is a caller-trace name bound before the loop (ctx.bind) and
    rebound in the body (the handle's `bind`); the SSA pass infers the
    loop-carried channel from that rebinding, exactly as it does for a
    hand-written `pl.range` accumulator.
    """

    def __init__(self, ctx, name: str, facade) -> None:
        self._ctx = ctx
        self._name = name
        self._value = facade

    @property
    def value(self):
        """The channel's current value — the initial value, then each rebind's."""
        return self._value

    def bind(self, v) -> None:
        """Rebind the channel for this iteration (the body's carried next value)."""
        self._value = self._ctx.bind(self._name, v)


@pl.authoring.construct
def accumulate(ctx, n, init):
    """A with-shaped accumulation region: one carried tile channel over a loop."""
    acc = ctx.bind("accumulate_acc", init)
    with ctx.open_loop(0, n, 1):
        yield Accumulator(ctx, "accumulate_acc", acc)


def _loops_of(fn: ir.Function) -> list[ir.ForStmt]:
    """The function's `ForStmt`s in body order (their bodies included)."""

    class _LoopCollect(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.loops: list[ir.ForStmt] = []

        def visit_for_stmt(self, op: ir.ForStmt) -> None:
            self.loops.append(op)
            super().visit_for_stmt(op)

    assert fn.body is not None
    collector = _LoopCollect()
    collector.visit_stmt(fn.body)
    return collector.loops


def _lowered(fn: ir.Function) -> ir.Function:
    """One traced function lowered through SSA — the carry-inference lane."""
    from pypto.pypto_core import passes  # noqa: PLC0415 - the pass registry

    program = passes.convert_to_ssa()(ir.Program([fn], "probe", ir.Span.unknown()))
    return next(iter(program.functions.values()))


def test_multiple_carried_channels_preserve_order_and_types():
    """A tile and a scalar carried through one region: two channels, both real."""

    class Pair(pl.authoring.ConstructHandle):
        def __init__(self, ctx, names, facades) -> None:
            self._ctx = ctx
            self._channels = dict(zip(names, facades))

        @property
        def total(self):
            return self._channels["accumulate_total"]

        @property
        def count(self):
            return self._channels["accumulate_count"]

        def bind(self, total, count) -> None:
            self._channels["accumulate_total"] = self._ctx.bind("accumulate_total", total)
            self._channels["accumulate_count"] = self._ctx.bind("accumulate_count", count)

    @pl.authoring.construct
    def accumulate_pair(ctx, n, init_total, init_count):
        total = ctx.bind("accumulate_total", init_total)
        count = ctx.bind("accumulate_count", init_count)
        with ctx.open_loop(0, n, 1):
            yield Pair(ctx, ("accumulate_total", "accumulate_count"), (total, count))

    def recipe(n: pl.Scalar[pl.INDEX], x: T8x8, out: pl.Out[T8x8]):
        t = pl.load(x, [0, 0], [8, 8])
        c = pl.const(0, pl.FP32)
        with accumulate_pair(n, t, c) as red:
            red.bind(pl.add(red.total, t), red.count + pl.const(1, pl.FP32))
        pl.store(red.total, [0, 0], out)

    annotations: dict[str, object] = {"n": pl.Scalar[pl.INDEX], "x": T8x8, "out": OUT8x8}
    fn = _trace(recipe, annotations, directions={"out": ir.ParamDirection.Out})
    (loop,) = _loops_of(_lowered(fn))
    assert len(loop.iter_args) == 2 and len(loop.return_vars) == 2
    # The two carried channels are the tile and the scalar (the pass's own
    # discovery order, not the source order).
    carried = {type(ia.type) for ia in loop.iter_args}
    assert carried == {ir.TileType, ir.ScalarType}
    counts = [ia for ia in loop.iter_args if isinstance(ia.type, ir.ScalarType)]
    assert isinstance(counts[0].type, ir.ScalarType)
    assert counts[0].type.dtype == pl.FP32


def test_a_parallel_recipe_loop_rejects_break():
    """The kind's transfer rule crosses the capability: parallel rejects the keyword."""

    @pl.authoring.construct
    def par_counted(ctx, n):
        with ctx.open_loop(0, n, 1, kind="parallel") as i:
            yield i

    with pytest.raises(InvalidOperationError, match="'break' not supported in parallel loops"):

        def kernel(n: pl.Scalar[pl.INDEX], x: T8x8, out: pl.Out[T8x8]):
            t = pl.load(x, [0, 0], [8, 8])
            for i in par_counted(n):
                if i > 2:
                    break
                t = pl.add(t, t)
            pl.store(t, [0, 0], out)

        _trace(
            kernel,
            {"n": pl.Scalar[pl.INDEX], "x": T8x8, "out": OUT8x8},
            directions={"out": ir.ParamDirection.Out},
        )


def test_open_loop_rejects_an_unknown_kind():
    """The kind vocabulary is closed — the `pl.range` family's kinds only."""

    @pl.authoring.construct
    def bad_kind(ctx, n):
        with ctx.open_loop(0, n, 1, kind="strange") as i:
            yield i

    with pytest.raises(TypeError, match="ctx.open_loop kind must be one of"):

        def kernel(n: pl.Scalar[pl.INDEX], x: T8x8, out: pl.Out[T8x8]):
            for i in bad_kind(n):
                pl.load(x, [0, 0], [8, 8])

        _trace(
            kernel,
            {"n": pl.Scalar[pl.INDEX], "x": T8x8, "out": OUT8x8},
            directions={"out": ir.ParamDirection.Out},
        )


def test_bind_refuses_the_recipes_own_target_name():
    """A channel named after the recipe's own target (`as` handle or loop
    variable) is refused — the channel would silently shadow it."""

    @pl.authoring.construct
    def channel_named_red(ctx, n, init):
        acc = ctx.bind("red", init)  # 'red' is the user's as target
        with ctx.open_loop(0, n, 1):
            yield Accumulator(ctx, "red", acc)

    with pytest.raises(TypeError, match="bind: 'red' is this recipe's own target name"):

        def kernel(n: pl.Scalar[pl.INDEX], x: T8x8, out: pl.Out[T8x8]):
            t = pl.load(x, [0, 0], [8, 8])
            with channel_named_red(n, t) as red:
                red.bind(pl.add(red.value, t))
            pl.store(red.value, [0, 0], out)

        _trace(
            kernel,
            {"n": pl.Scalar[pl.INDEX], "x": T8x8, "out": OUT8x8},
            directions={"out": ir.ParamDirection.Out},
        )


def test_bind_refuses_the_loop_target_name():
    """The for-spelling's loop target is equally refused — the channel would
    collide with the loop variable the user's body reads."""

    @pl.authoring.construct
    def channel_named_i(ctx, n, init):
        ctx.bind("i", init)  # 'i' is the loop target
        with ctx.open_loop(0, n, 1) as i:
            yield i

    with pytest.raises(TypeError, match="bind: 'i' is this recipe's own target name"):

        def kernel(n: pl.Scalar[pl.INDEX], x: T8x8, out: pl.Out[T8x8]):
            t = pl.load(x, [0, 0], [8, 8])
            for i in channel_named_i(n, t):
                t = pl.add(t, t)
            pl.store(t, [0, 0], out)

        _trace(
            kernel,
            {"n": pl.Scalar[pl.INDEX], "x": T8x8, "out": OUT8x8},
            directions={"out": ir.ParamDirection.Out},
        )


def test_bind_refuses_a_user_local_collision():
    """A channel colliding with a caller local is refused — without the
    refusal the recipe's bind would silently rebind the user's name (a
    post-loop read of the user's local would see the channel's value)."""

    with pytest.raises(TypeError, match="bind: 'accumulate_acc' collides with a local"):

        def kernel(n: pl.Scalar[pl.INDEX], x: T8x8, out: pl.Out[T8x8]):
            t = pl.load(x, [0, 0], [8, 8])
            accumulate_acc = pl.const(42, pl.FP32)  # a user local the recipe must not touch
            with accumulate(n, t) as red:
                red.bind(pl.add(red.value, t))
            stored = cast(pl.Tile, accumulate_acc)
            pl.store(stored, [0, 0], out)

        _trace(
            kernel,
            {"n": pl.Scalar[pl.INDEX], "x": T8x8, "out": OUT8x8},
            directions={"out": ir.ParamDirection.Out},
        )


def test_a_carried_channel_rebinds_across_the_body():
    """The refusal keeps the legitimate lane open: a fresh channel name
    first-bound by the recipe rebinds through the handle (same ctx) without
    tripping the collision check."""

    def recipe(n: pl.Scalar[pl.INDEX], x: T8x8, out: pl.Out[T8x8]):
        t = pl.load(x, [0, 0], [8, 8])
        with accumulate(n, t) as red:
            red.bind(pl.add(red.value, t))
            red.bind(pl.add(red.value, t))  # the second rebind is still the channel
        pl.store(red.value, [0, 0], out)

    fn = _trace(
        recipe,
        {"n": pl.Scalar[pl.INDEX], "x": T8x8, "out": OUT8x8},
        directions={"out": ir.ParamDirection.Out},
    )
    (loop,) = _loops_of(_lowered(fn))
    assert len(loop.iter_args) == 1 and len(loop.return_vars) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
