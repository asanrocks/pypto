# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile block queries and cross-core transfers."""

from typing import TypeVar
from typing import cast as _typing_cast

from pypto.ir.op import tile_ops as _ir_ops
from pypto.ir.utils import _get_span_or_capture
from pypto.language.ambient import active_frame
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import (
    Span,
)

from ..typing import Scalar, Tensor, Tile

# Constrained TypeVar for the split-axis reshape wrappers (aiv_shard / aic_gather):
# the operand is either a Tile (legacy @pl.program form) or a Tensor (@pl.jit /
# pl.spmd form), and the result is the SAME kind as the input.
#
# Constrained, NOT bound, and deliberately unlike the loop-carry TypeVars in
# dsl_api: a bound ``Tensor | Tile`` would propagate a DistributedTensor operand
# into the result type, but AIV/AIC split does not accept one. The deducer
# matches the operand with ``As<TensorType>``, which is precise-match and returns
# null for DistributedTensorType by design, so ``tensor.aiv_shard(dist)`` is
# rejected ("requires argument to be a (non-distributed) TensorType", see
# DeduceSplitReshapeTensor in src/ir/op/tile_ops/cross_core.cpp). Collapsing the
# subclass here keeps the annotation from advertising a result the op cannot
# produce; widening it needs both operation contracts updated first.
_SplitOperandT = TypeVar("_SplitOperandT", Tensor, Tile)


def _split_transfer_attrs(
    op_name: str,
    x: _SplitOperandT,
    split: int | None,
    lane_stride: int | None,
) -> tuple[str, dict[str, int]]:
    """The split-transfer authoring contract, mirroring the AST parser's
    ``_parse_split_transfer_op``: two forms, never mixed.

    * **Region form** — no ``split=``: the mode is inherited from the enclosing
      ``for aiv_id in pl.split_aiv(mode=...)`` scope (the tracing-side region
      channel); a ``Tile`` or a high-level ``Tensor`` operand is valid.
    * **Outlined form** — an explicit integer ``split=``: the printer's
      spelling for a function already lowered into the explicit form
      (``LowerAutoVectorSplit`` / ``OutlineIncoreScopes`` output, or a
      hand-written split_aiv kernel). The split rides on the op itself, so the
      form round-trips without re-synthesizing the loop wrapper; it is
      tile-only, and rejected inside a ``split_aiv`` region (the scope owns
      the mode there).

    Returns the op namespace and the op kwargs to stamp.
    """
    from ..distributed.typing import DistributedTensor  # noqa: PLC0415 - the rejection check (AST parity)

    if isinstance(x, DistributedTensor):
        raise TypeError(f"pl.{op_name} does not support distributed tensors (AIV/AIC split only)")
    frame = active_frame()
    region_mode = frame.current_split_aiv_mode() if frame is not None else None
    if split is not None:
        if region_mode is not None:
            raise TypeError(
                f"pl.{op_name}() does not take a split= argument inside a "
                "'for ... in pl.split_aiv(...)' loop — the split mode is inherited from that scope"
            )
        if isinstance(x, Tensor):
            raise TypeError(
                f"pl.{op_name}(x, split=N) does not accept a high-level Tensor operand — "
                "the outlined split= form is tile-only; wrap the Tensor in a "
                "'for aiv_id in pl.split_aiv(...)' region instead"
            )
        if not isinstance(split, int):  # type: ignore[reportUnnecessaryIsInstance]  # defensive; the annotation is the contract
            raise TypeError(
                f"pl.{op_name}(..., split=N) requires an integer split (the lowered "
                f"SplitMode value), got {split!r}"
            )
        attrs = {"split": int(split)}
        if lane_stride is not None:
            if op_name == "aic_gather":
                raise TypeError(
                    "pl.aic_gather() does not take a lane_stride= argument: only the "
                    "Cube -> Vector shard is ever rebalanced onto a ragged boundary's "
                    "valid region, so the gather always re-joins the lanes on the box partition"
                )
            if not isinstance(lane_stride, int):  # type: ignore[reportUnnecessaryIsInstance]  # defensive; the annotation is the contract
                raise TypeError(
                    f"pl.{op_name}(..., lane_stride=N) requires an integer partition "
                    f"stride, got {lane_stride!r}"
                )
            attrs["lane_stride"] = int(lane_stride)
        return "tile", attrs
    if region_mode is None:
        raise RuntimeError(
            f"pl.{op_name}() must be used inside a 'for ... in pl.split_aiv(...)' loop "
            "(or pass an explicit integer 'split=' in the outlined form)"
        )
    if lane_stride is not None:
        raise TypeError(
            f"pl.{op_name}() does not take a lane_stride= argument inside a "
            "'for ... in pl.split_aiv(...)' loop — the partition follows the region's "
            "own per-lane offsets"
        )
    ns = "tensor" if isinstance(x, Tensor) else "tile"
    return ns, {"split": int(region_mode.value)}


def aiv_shard(
    x: _SplitOperandT, span: Span | None = None, *, split: int | None = None, lane_stride: int | None = None
) -> _SplitOperandT:
    """Bring a cube-produced operand onto the AIV lane (AIC -> AIV crossing).

    In a data-parallel region (``UP_DOWN`` / ``LEFT_RIGHT``) the crossing also
    **halves** the operand along the split axis, so each lane gets one half; in a
    task-parallel ``mode=NONE`` region there is no split axis, so it crosses and
    **preserves the shape**. Either way, writing it is how a C->V crossing into a
    region is named — the ``AivSplitValid`` verifier rejects an unnamed one.

    Two authoring forms, never mixed (the contract mirrors the AST parser's
    ``_parse_split_transfer_op``):

    * **Region form** ``pl.aiv_shard(x)`` inside a
      ``for aiv_id in pl.split_aiv(mode=...)`` loop — the mode is inherited
      from that scope and never passed here. The region's trace iterable
      pushes its mode onto the split_aiv region channel and this facade reads
      it, stamping the ``split`` attr exactly as the parser does. The operand
      may be a ``Tile`` (``tile.aiv_shard``) or a high-level ``Tensor``
      (``tensor.aiv_shard``, lowered 1:1 at ConvertTensorToTileOps).
    * **Outlined form** ``pl.tile.aiv_shard(tile, split=N)`` — the printer's
      spelling for a function already lowered into the explicit form. The
      split is stated on the op verbatim; the form is tile-only and rejected
      inside a ``split_aiv`` region. ``lane_stride=`` (an integer partition
      stride stamped by ``LowerAutoVectorSplit`` when it balances a ragged
      boundary) rides only in this form.

    Distributed tensors are not supported.

    Args:
        x: Input operand (2D Tile or Tensor)
        span: Optional source span
        split: Outlined-form only: the lowered integer ``SplitMode`` value
        lane_stride: Outlined-form only: the integer partition stride

    Returns:
        Operand of the same kind: the split axis halved in a data-parallel region,
        the shape unchanged in a ``mode=NONE`` one.
    """
    ns, attrs = _split_transfer_attrs("aiv_shard", x, split, lane_stride)
    actual_span = _get_span_or_capture(span, frame_offset=1)
    call_expr = _ir_core.create_op_call(f"{ns}.aiv_shard", [x.unwrap()], attrs, actual_span)
    if isinstance(x, Tensor):
        return _typing_cast(_SplitOperandT, Tensor(expr=call_expr))
    return _typing_cast(_SplitOperandT, Tile(expr=call_expr))


def aic_gather(
    x: _SplitOperandT, span: Span | None = None, *, split: int | None = None, lane_stride: int | None = None
) -> _SplitOperandT:
    """Hand a vector-produced operand to the cube (AIV -> AIC crossing).

    Inverse of [`aiv_shard`][pypto.language.tile.aiv_shard]: in a data-parallel region it **rejoins** the two
    lanes' halves along the split axis, and in a task-parallel ``mode=NONE`` region
    it crosses and **preserves the shape**. It is how a V->C crossing out of a
    region is named; an unnamed one is rejected by the ``AivSplitValid`` verifier.

    Out of a ``mode=NONE`` region the two lanes share one destination slot with no
    per-lane offset and nothing arbitrates between them: both push, so when they
    hold different values the cube receives an **unspecified** one of the two.
    Guarding the *production* of the value does not help — lane 1 still reaches
    the push and still sends its own tile. Gather only a value both lanes agree
    on; if they must contribute different data, use a data-parallel region.

    Like [`aiv_shard`][pypto.language.tile.aiv_shard], two authoring forms,
    never mixed (the contract mirrors the AST parser's
    ``_parse_split_transfer_op``): the **region form** ``pl.aic_gather(x)``
    inherits the mode from the enclosing ``for aiv_id in pl.split_aiv(...)``
    scope (``Tile`` or ``Tensor`` operand), and the **outlined form**
    ``pl.tile.aic_gather(tile, split=N)`` states the lowered integer split on
    the op itself — tile-only, rejected inside a region, and never taking
    ``lane_stride=`` (only the Cube -> Vector shard is ever rebalanced).

    The operand may be a ``Tile`` (legacy ``@pl.program`` form ->
    ``tile.aic_gather``) or a high-level ``Tensor`` (``@pl.jit`` /
    ``pl.spmd`` form -> ``tensor.aic_gather``, lowered 1:1 to
    ``tile.aic_gather`` at ConvertTensorToTileOps). The Tensor form is
    region-only. Distributed tensors are not supported.

    Args:
        x: Input operand (2D Tile or Tensor)
        span: Optional source span
        split: Outlined-form only: the lowered integer ``SplitMode`` value

    Returns:
        Operand of the same kind: the split axis doubled in a data-parallel region,
        the shape unchanged in a ``mode=NONE`` one.
    """
    ns, attrs = _split_transfer_attrs("aic_gather", x, split, lane_stride)
    actual_span = _get_span_or_capture(span, frame_offset=1)
    call_expr = _ir_core.create_op_call(f"{ns}.aic_gather", [x.unwrap()], attrs, actual_span)
    if isinstance(x, Tensor):
        return _typing_cast(_SplitOperandT, Tensor(expr=call_expr))
    return _typing_cast(_SplitOperandT, Tile(expr=call_expr))


def get_block_idx() -> Scalar:
    """Get the current block index.

    This operation returns the index of the current compute tile. It is typically
    used in tile-level programming to identify which block of data is being processed.

    Returns:
        Scalar wrapping the get_block_idx operation (INDEX type)

    Example:
        >>> block_idx = pl.tile.get_block_idx()
        >>> if block_idx < 10:
        >>>     # Process first 10 blocks differently
        >>>     ...
    """
    call_expr = _ir_ops.get_block_idx()
    return Scalar(expr=call_expr)


def get_subblock_idx() -> Scalar:
    """Get the current sub-block (vector core) index.

    Returns the index of the current vector core within a split execution.
    Core 0 returns 0, core 1 returns 1.

    Returns:
        Scalar wrapping the get_subblock_idx operation (INDEX type)
    """
    call_expr = _ir_ops.get_subblock_idx()
    return Scalar(expr=call_expr)


def get_block_num() -> Scalar:
    """Get the total number of blocks in the current SPMD task.

    This operation returns the total count of blocks dispatched for the current
    task. Used with get_block_idx() for SPMD work partitioning.

    Returns:
        Scalar wrapping the get_block_num operation (INDEX type)

    Example:
        >>> block_idx = pl.tile.get_block_idx()
        >>> block_num = pl.tile.get_block_num()
    """
    call_expr = _ir_ops.get_block_num()
    return Scalar(expr=call_expr)


from ._validation import wrap_module_ops  # noqa: E402

wrap_module_ops(__name__)
