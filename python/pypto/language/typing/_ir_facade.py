# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Annotation-mode facades built from a concrete IR type.

The reverse of ``annotation_to_type``: every field the type carries crosses
into the facade — kind, extents, dtype, layout, view, storage, tile memory
space — so a body collected against the facade is collected against exactly
the value the type describes. Both a traced callee's call site (the tracer's
callee lane) and the parser-lane import bridge for traced inline helpers
build facades this way, from a value that already exists; reconstructing a
type from source metadata instead loses the fields no source-level model
records (an explicit tile memory space, a view's valid region, the
distributed kind of a window).
"""

from __future__ import annotations

from typing import Any

from pypto.pypto_core import ir

__all__ = ["facade_from_ir_type"]


def facade_from_ir_type(ty: ir.Type) -> Any:
    """The annotation-mode facade stating exactly ``ty``.

    Shape entries carry over as-is — an ``int`` or an extent ``Expr`` —
    because the annotation accepts both and the type boundary normalizes.
    Raises :class:`TypeError` for a kind with no facade form, so an
    unmapped type is loud rather than approximated by a plain tensor.

    Args:
        ty: The concrete IR type a parsed value carries.

    Returns:
        A ``DistributedTensor`` / ``Tensor`` / ``Tile`` / ``Scalar`` / ``Array``
        facade in annotation-only mode.
    """
    # The DistributedTensorType check must precede TensorType: the pybind
    # flavour IS-A TensorType, so the plain branch would silently downgrade a
    # distributed window to a plain tensor annotation — the retype the
    # type-soundness contract forbids.
    if isinstance(ty, ir.DistributedTensorType):
        from pypto.language.distributed.typing import (  # noqa: PLC0415 — lazy: distributed pulls the op tier
            DistributedTensor,
        )

        return DistributedTensor(
            list(ty.shape),  # type: ignore[arg-type]
            ty.dtype,
            layout=ty.tensor_view,  # type: ignore[arg-type]  # slot takes a TensorView too
            memref=ty.memref,
            _annotation_only=True,
        )
    if isinstance(ty, ir.TensorType):
        from pypto.language.typing.tensor import Tensor  # noqa: PLC0415 — import cycle

        return Tensor(
            list(ty.shape),  # type: ignore[arg-type]  # int or extent Expr, normalized at the type boundary
            ty.dtype,
            layout=ty.tensor_view,  # type: ignore[arg-type]  # slot takes a TensorView too
            memref=ty.memref,
            _annotation_only=True,
        )
    if isinstance(ty, ir.TileType):
        from pypto.language.typing.tile import Tile  # noqa: PLC0415 — import cycle

        # The whole tile type crosses — memory space and view included — so
        # the callee's parameter is the argument's type, not a looser
        # approximation: a body whose ops depend on the tile's placement
        # (a Mat-resident matmul operand) traces against the real space.
        return Tile(
            list(ty.shape),  # type: ignore[arg-type]  # extents stay Exprs
            ty.dtype,
            memref=ty.memref,
            memory_space=ty.memory_space,
            tile_view=ty.tile_view,
            _annotation_only=True,
        )
    if isinstance(ty, ir.ScalarType):
        from pypto.language.typing.scalar import Scalar  # noqa: PLC0415 — import cycle

        return Scalar(ty.dtype, _annotation_only=True)
    if isinstance(ty, ir.ArrayType):
        from pypto.language.typing import Array  # noqa: PLC0415 — import cycle

        return Array(ty.extent, ty.dtype, _annotation_only=True)  # type: ignore[arg-type]  # Expr extent
    raise TypeError(
        f"IR type {ty} has no annotation facade form (Tensor/DistributedTensor/Tile/Scalar/Array only)"
    )
