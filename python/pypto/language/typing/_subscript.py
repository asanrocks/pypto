# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numpy-style subscript reads for the ``Tensor`` / ``Tile`` facades
(the residual in-scope stdlib gap).

Ports the AST parser's ``_parse_tensor_subscript`` / ``_parse_tile_subscript``
numpy-indexing surface (rank-reducing scalar reads via ``*_read``, slice
views via ``*_slice`` with ``drop_dims``, the tile valid-shape intersection
and the tile 2D clamp) onto the existing ``tensor.slice`` / ``tensor.read`` /
``tile.slice`` / ``tile.read`` DSL ops. The facade ``__getitem__`` receives
*evaluated* Python indices (``int`` / ``Scalar`` facade / ``slice``), which
is simpler than the AST's node walking — same semantics, minus the
arith-simplifier step (raw ``sub`` / ``min_`` / ``max_`` nodes are emitted;
the passes can simplify later, correctness is preserved).

Loud gates mirror the AST's: slice ``step`` is rejected, and tile slice
*lower* bounds must be compile-time constants ("Dynamic lower bounds are
not supported in tile subscript"). These are the AST's own rejections, so
the corpus never exercises them.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, overload

if TYPE_CHECKING:
    from pypto.language.typing.tensor import Tensor
    from pypto.language.typing.tile import Tile

from pypto.language.typing.scalar import Scalar
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir

_SPAN = _ir.Span.unknown()
_INDEX_ZERO = _ir.ConstInt(0, DataType.INDEX, _SPAN)


def _const_int_value(value: object) -> int | None:
    """Extract the integer value of a compile-time constant, else None."""
    if isinstance(value, int):
        return value
    if isinstance(value, _ir.ConstInt):
        return value.value
    if isinstance(value, _ir.Neg) and isinstance(value.operand, _ir.ConstInt):
        return -value.operand.value
    return None


def _to_index_expr(v: int | Scalar | _ir.Expr) -> _ir.Expr:
    """Lift a subscript bound/index to an INDEX-typed ``ir.Expr`` (the
    operand boundary — the one automatic host-literal lift)."""
    if isinstance(v, Scalar):
        return v.unwrap()
    if isinstance(v, _ir.Expr):
        return v
    # narrowed to int (the declared union minus the two branches above)
    return _ir.ConstInt(v, DataType.INDEX, _SPAN)


def _extent(upper: int | _ir.Expr, lower: int | _ir.Expr) -> int | _ir.Expr:
    """A slice extent, folding compile-time constants to an int.

    A symbolic extent (``x[pos : pos + 1]`` → ``sub(add(pos, 1), pos)``) is
    simplified through the same arithmetic analyzer the AST parser uses
    for its slice extents — the two frontends must converge on the same
    canonical shape (the parser folds this to ``1``; an unsimplified
    tracing-side extent diverges structurally and trips type checks the
    folded form passes)."""
    up, lo = _const_int_value(upper), _const_int_value(lower)
    if up is not None and lo is not None:
        return up - lo
    simplified = _extent_analyzer().simplify(_ir.sub(_to_index_expr(upper), _to_index_expr(lower)))
    const = _const_int_value(simplified)
    if const is not None:
        return const
    return simplified


_EXTENT_ANALYZER: list[Any] = []


def _extent_analyzer() -> Any:
    """The cached arithmetic analyzer (the parser caches one per parse for
    the same amortization; here one per interpreter)."""
    if not _EXTENT_ANALYZER:
        from pypto.pypto_core import arith  # noqa: PLC0415

        _EXTENT_ANALYZER.append(arith.Analyzer())
    return _EXTENT_ANALYZER[0]


def _clamped_extent(upper: int | _ir.Expr, lower: int | _ir.Expr) -> int | _ir.Expr:
    """A non-negative slice extent (the AST's ``_build_clamped_slice_shape_expr``)."""
    folded = _extent(upper, lower)
    const = _const_int_value(folded)
    if const is not None:
        return max(const, 0)
    return _ir.max_(_to_index_expr(folded), _INDEX_ZERO, _SPAN)


def _intersect_upper(requested: int | _ir.Expr, source: int | _ir.Expr) -> int | _ir.Expr:
    """Intersect an explicit slice upper bound with the source logical extent."""
    rq, src = _const_int_value(requested), _const_int_value(source)
    if rq is not None and src is not None:
        return min(rq, src)
    return _ir.min_(_to_index_expr(requested), _to_index_expr(source), _SPAN)


def _alloc_extent(
    static_extent: int | _ir.Expr,
    lower: int | _ir.Expr,
    upper: int | _ir.Expr | None,
) -> int | _ir.Expr:
    """The AST's ``_build_tile_alloc_extent`` port: the largest static extent
    that can safely back the slice while ``valid_shape`` carries any runtime
    narrowing. Tile slices must keep a compile-time allocation shape."""
    lower_const = _const_int_value(lower)
    static_const = _const_int_value(static_extent)
    upper_const = None if upper is None else _const_int_value(upper)

    max_extent = None
    if lower_const is not None and static_const is not None:
        max_extent = max(static_const - lower_const, 0)

    if upper_const is not None:
        extent = max(upper_const - lower_const, 0) if lower_const is not None else upper_const
        if max_extent is not None:
            extent = min(extent, max_extent)
        if extent <= 0:
            raise TypeError("tile subscript must produce a positive static extent")
        return extent

    if max_extent is not None:
        if max_extent <= 0:
            raise TypeError("tile subscript must produce a positive static extent")
        return max_extent

    return static_extent


def _extents_match(lhs: int | _ir.Expr, rhs: int | _ir.Expr) -> bool:
    """True when two slice extents are obviously equivalent."""
    lc, rc = _const_int_value(lhs), _const_int_value(rhs)
    if lc is not None and rc is not None:
        return lc == rc
    return lhs is rhs


if TYPE_CHECKING:
    # A subscript axis: a scalar-like index (int / Scalar / Expr) or a slice
    # with any bound kind. Bare ``slice`` is slice[Unknown, ...] to pyright
    # (and slice is not runtime-subscriptable), so the alias is check-only;
    # every use site is a lazy annotation.
    _SubscriptAxis = int | Scalar | _ir.Expr | slice[Any, Any, Any]


def _normalize_seq(indices: _SubscriptAxis | tuple[_SubscriptAxis, ...]) -> tuple[_SubscriptAxis, ...]:
    """A bare index is a 1-tuple; a tuple passes through."""
    if isinstance(indices, tuple):
        return indices
    return (indices,)


def _tensor_slice_args(
    seq: tuple[_SubscriptAxis, ...], full_shape: list[int | _ir.Expr]
) -> tuple[list[int | _ir.Expr], list[int | _ir.Expr], list[int]]:
    """Convert mixed/partial tensor indices into slice shape/offset/drop_dims
    (the AST's ``_build_subscript_slice_args`` port). shape/offset are always
    full-rank: a scalar index contributes ``shape=1, offset=i`` and joins
    ``drop_dims``; a slice keeps its dim; trailing dims are implicit ``:``."""
    shape: list[int | _ir.Expr] = []
    offset: list[int | _ir.Expr] = []
    drop_dims: list[int] = []

    for dim, idx in enumerate(seq):
        if not isinstance(idx, slice):
            offset.append(_to_index_expr(idx))
            shape.append(1)
            drop_dims.append(dim)
            continue

        if idx.step is not None:
            raise TypeError("slice step is not supported in tensor subscript")
        if idx.start is None:
            offset.append(0)
            shape.append(full_shape[dim] if idx.stop is None else _to_index_expr(idx.stop))
            continue

        lower = _to_index_expr(idx.start)
        offset.append(lower)
        upper = full_shape[dim] if idx.stop is None else _to_index_expr(idx.stop)
        shape.append(_extent(upper, lower))

    for dim in range(len(seq), len(full_shape)):
        offset.append(0)
        shape.append(full_shape[dim])

    return shape, offset, drop_dims


def _tile_slice_args(
    seq: tuple[_SubscriptAxis, ...], tile_type: _ir.TileType
) -> tuple[list[int | _ir.Expr], list[int | _ir.Expr], list[int | _ir.Expr] | None, list[int]]:
    """The AST's ``_build_tile_subscript_slice_args`` port: same rank-reducing
    rules as the tensor, plus the valid-shape intersection against the
    source's logical (view) extent and the compile-time alloc extent."""
    static_shape = list(tile_type.shape)
    tile_view = tile_type.tile_view
    logical_shape = (
        list(tile_view.valid_shape)
        if tile_view is not None and len(tile_view.valid_shape) == len(static_shape)
        else list(static_shape)
    )

    shape: list[int | _ir.Expr] = []
    offset: list[int | _ir.Expr] = []
    valid: list[int | _ir.Expr] = []
    drop_dims: list[int] = []
    needs_valid_shape = False

    for dim, idx in enumerate(seq):
        if not isinstance(idx, slice):
            offset.append(_to_index_expr(idx))
            shape.append(1)
            valid.append(1)
            drop_dims.append(dim)
            continue

        if idx.step is not None:
            raise TypeError("slice step is not supported in tile subscript")
        lower = 0 if idx.start is None else _to_index_expr(idx.start)
        if idx.start is not None and _const_int_value(lower) is None:
            raise TypeError("dynamic lower bounds are not supported in tile subscript")
        offset.append(lower)

        parsed_upper = None if idx.stop is None else _to_index_expr(idx.stop)
        valid_upper = (
            logical_shape[dim] if parsed_upper is None else _intersect_upper(parsed_upper, logical_shape[dim])
        )
        valid_extent = valid_upper if idx.start is None else _clamped_extent(valid_upper, lower)
        alloc_extent = _alloc_extent(static_shape[dim], lower, parsed_upper)

        shape.append(alloc_extent)
        valid.append(valid_extent)
        needs_valid_shape = needs_valid_shape or not _extents_match(alloc_extent, valid_extent)

    for dim in range(len(seq), len(static_shape)):
        offset.append(0)
        shape.append(static_shape[dim])
        valid.append(logical_shape[dim])
        needs_valid_shape = needs_valid_shape or not _extents_match(static_shape[dim], logical_shape[dim])

    return shape, offset, valid if needs_valid_shape else None, drop_dims


def _render_dim(dim: int | _ir.Expr) -> str:
    """Render a shape dim for the tile 2D-promotion warning."""
    value = _const_int_value(dim)
    return str(value) if value is not None else "?"


def _warn_tile_promotion(shape: Sequence[int | _ir.Expr], drop_dims: list[int], rank: int) -> None:
    """The AST's non-fatal 2D-promotion warning for a rank-reduced tile view."""
    natural_rank = rank - len(drop_dims)
    if drop_dims and natural_rank < 2:
        kept = [shape[i] for i in range(rank) if i not in set(drop_dims)]
        promoted = [1] * (2 - len(kept)) + list(kept)
        promoted_repr = "[" + ", ".join(_render_dim(d) for d in promoted) + "]"
        warnings.warn(
            f"tile subscript reduces to {natural_rank}D; auto-promoting to 2D shape {promoted_repr} "
            f"— use an explicit tile.reshape if you want a different layout",
            UserWarning,
            stacklevel=3,
        )


@overload
def subscript_read(
    facade: Tensor, indices: _SubscriptAxis | tuple[_SubscriptAxis, ...], *, tile: Literal[False]
) -> Tensor | Scalar: ...


@overload
def subscript_read(
    facade: Tile, indices: _SubscriptAxis | tuple[_SubscriptAxis, ...], *, tile: Literal[True]
) -> Tile | Scalar: ...


def subscript_read(
    facade: Tensor | Tile, indices: _SubscriptAxis | tuple[_SubscriptAxis, ...], *, tile: bool
) -> Tensor | Tile | Scalar:
    """The numpy-style subscript read: all-scalar full-rank
    → a scalar ``*_read``; anything else (partial, sliced, rank-reducing) →
    a ``*_slice`` view carrying ``drop_dims`` for the scalar-indexed axes.

    The ``tile`` flag and the facade's class agree by contract (the two
    facade ``__getitem__`` hooks are the only callers); the casts narrow
    the union for the op-module boundary rather than assert anything new.
    """
    from pypto.language.op import tensor_ops, tile_ops  # noqa: PLC0415 - local: avoid the op-module cycle

    seq = _normalize_seq(indices)
    expr = facade.unwrap()
    # Subscript targets are shaped values: the cast narrows the nanobind
    # ``Expr.type`` property (typed as the base ``Type``) to the two shaped
    # flavors whose ``.shape`` is read below.
    typ = cast("_ir.TensorType | _ir.TileType", expr.type)
    rank = len(typ.shape)
    if len(seq) > rank:
        raise TypeError(f"subscript has {len(seq)} indices but the operand is {rank}D")

    has_slice = any(isinstance(i, slice) for i in seq)
    if not has_slice and len(seq) == rank:
        # seq is slice-free here (the has_slice guard); the filter narrows the
        # axis union to the scalar-like read form.
        idx_exprs = [_to_index_expr(i) for i in seq if not isinstance(i, slice)]
        if tile:
            return tile_ops.read(cast("Tile", facade), idx_exprs)
        return tensor_ops.read(cast("Tensor", facade), idx_exprs)

    if tile:
        shape, offset, valid, drop_dims = _tile_slice_args(seq, cast("_ir.TileType", typ))
        _warn_tile_promotion(shape, drop_dims, rank)
        return tile_ops.slice(cast("Tile", facade), shape, offset, valid_shape=valid, drop_dims=drop_dims)

    shape, offset, drop_dims = _tensor_slice_args(seq, list(typ.shape))
    return tensor_ops.slice(cast("Tensor", facade), shape, offset, drop_dims=drop_dims)


__all__ = ["subscript_read"]
