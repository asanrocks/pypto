# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile wrapper type for PyPTO Language DSL.

Tile represents a tile in unified buffer memory, used for tile-level programming.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, MemorySpace, MemRef, TileView

if TYPE_CHECKING:
    from pypto.language.typing.scalar import Scalar

# A subscript key: one axis (index or slice) or a tuple of them. Quoted —
# Scalar is a TYPE_CHECKING-only import to avoid the facade cycle.
_SubscriptKey: TypeAlias = (
    "int | Scalar | Expr | slice[Any, Any, Any] | tuple[int | Scalar | Expr | slice[Any, Any, Any], ...]"
)


class TileMeta(type):
    """Metaclass for Tile to enable subscript notation."""

    def __getitem__(cls, item: tuple[Any, ...]) -> "Tile":
        """Enable Tile[[shape], dtype] annotation syntax.

        Args:
            item: Tuple of:
                - (shape, dtype)
                - (shape, dtype, memory_space_or_tile_view)
                - (shape, dtype, memref, memory_space)
                - (shape, dtype, memref, memory_space, tile_view)

        Returns:
            Tile instance with shape, dtype, and optional memref / memory space / tile view
        """
        # The implicit ``cls`` is typed as the metaclass object, which the
        # ``__call__`` overloads (``cls: type[TensorT]``) cannot bind — rebind
        # to the Tile-family class it actually is.
        tile_cls = cast("type[Tile]", cls)
        if len(item) not in (2, 3, 4, 5):
            raise TypeError(
                "Tile requires [shape, dtype], [shape, dtype, memory_space_or_tile_view], "
                "[shape, dtype, memref, memory_space], or "
                "[shape, dtype, memref, memory_space, tile_view] notation"
            )

        shape, dtype, *extras = item
        memref = None
        memory_space = None
        tile_view = None

        for extra in extras:
            if isinstance(extra, MemRef):
                if memref is not None:
                    raise TypeError("Tile annotation can contain at most one MemRef")
                memref = extra
                continue
            if isinstance(extra, MemorySpace):
                if memory_space is not None:
                    raise TypeError("Tile annotation can contain at most one memory space")
                memory_space = extra
                continue
            if isinstance(extra, TileView):
                if tile_view is not None:
                    raise TypeError("Tile annotation can contain at most one TileView")
                tile_view = extra
                continue
            raise TypeError(
                "Tile trailing arguments must be MemRef, MemorySpace, or TileView, "
                f"got {type(extra).__name__}"
            )

        if memref is not None and memory_space is None:
            raise TypeError("Tile annotation with MemRef must also specify an explicit MemorySpace")

        return tile_cls(
            shape,
            dtype,
            memref=memref,
            memory_space=memory_space,
            tile_view=tile_view,
            _annotation_only=True,
        )

    def __call__(
        cls,
        shape: Sequence[int] | tuple[Any, Any] | None = None,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        memref: "MemRef | None" = None,
        memory_space: "MemorySpace | None" = None,
        tile_view: "TileView | None" = None,
        _annotation_only: bool = False,
    ) -> "Tile":
        """Enable both Tile((shape), dtype) syntax and runtime wrapping."""
        if (
            isinstance(shape, tuple)
            and len(shape) == 2
            and not isinstance(shape[0], int)
            and dtype is None
            and expr is None
        ):
            real_shape, real_dtype = shape
            return type.__call__(
                cls,
                real_shape,
                real_dtype,
                expr=None,
                memref=memref,
                memory_space=memory_space,
                tile_view=tile_view,
                _annotation_only=_annotation_only,
            )
        return type.__call__(
            cls,
            shape,
            dtype,
            expr=expr,
            memref=memref,
            memory_space=memory_space,
            tile_view=tile_view,
            _annotation_only=_annotation_only,
        )


class Tile(metaclass=TileMeta):
    """Tile type for PyPTO Language DSL.

    Tile represents a tile in unified buffer (UB) memory. It is used for
    tile-level programming with operations like load, store, add, mul, etc.

    Annotation mode (used in type hints):
        x: pl.Tile[[64, 64], pl.FP32]

    Runtime mode (wraps IR expressions):
        tile = pl.load(tensor, [0, 0], [64, 64])
        # Returns Tile wrapping the Call expression

    Examples:
        >>> import pypto.language as pl
        >>>
        >>> @pl.function
        ... def my_func(input: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        ...     tile: pl.Tile[[64, 64], pl.FP32] = pl.load(input, [0, 0], [64, 64])
        ...     result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
        ...     return pl.store(result, [0, 0], input)
    """

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        memref: MemRef | None = None,
        memory_space: MemorySpace | None = None,
        tile_view: TileView | None = None,
        _annotation_only: bool = False,
    ):
        """Initialize Tile.

        Args:
            shape: Shape (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            memref: Optional memory reference
            _annotation_only: Whether this is annotation-only mode
        """
        if _annotation_only:
            self.shape = shape
            self.dtype = dtype
            self.memref = memref
            self.memory_space = memory_space
            self.tile_view = tile_view
            self._expr = None
        elif expr is not None:
            self._expr = expr
            self.shape = None
            self.dtype = None
            self.memref = None
            self.memory_space = None
            self.tile_view = None
        else:
            raise ValueError(
                "Tile must be initialized with either (shape, dtype) for "
                "annotations or expr for runtime wrapping"
            )

    def unwrap(self) -> Expr:
        """Get underlying IR expression.

        Returns:
            The wrapped Expr/Call object

        Raises:
            ValueError: If called on an annotation-only Tile
        """
        if self._expr is None:
            raise ValueError("Cannot unwrap annotation-only Tile (used in type hints)")
        return self._expr

    @classmethod
    def __class_getitem__(cls, item: tuple[Sequence[int], DataType]) -> "Tile":
        """The static-checker hook for ``Tile[[shape], dtype]`` syntax.

        Runtime never calls this — the ``TileMeta.__getitem__`` subscript
        protocol takes precedence — but pyright resolves class subscripts
        through this signature.
        """
        raise NotImplementedError

    @overload
    def __getitem__(self, indices: "slice[Any, Any, Any] | tuple[slice[Any, Any, Any], ...]") -> "Tile": ...

    @overload
    def __getitem__(self, indices: _SubscriptKey) -> "Tile | Scalar": ...

    def __getitem__(self, indices: _SubscriptKey) -> "Tile | Scalar":
        """Numpy-style subscript read: all-scalar full-rank
        → a scalar ``tile.read``; partial / sliced / rank-reducing → a
        ``tile.slice`` view with ``drop_dims`` and the valid-shape
        intersection + 2D clamp (the AST's ``_parse_tile_subscript``
        surface).

        The overloads mirror that runtime split where it is statically
        knowable: a key made only of slices always takes the slice path, so
        it is typed ``Tile``. Any other key is rank-dependent — ``t[i, j]``
        is a scalar only at full rank, ``t[i]`` is a rank-reducing view
        above it — and the operand rank lives in the IR type, so the honest
        static result is the union. Narrow with ``isinstance`` when the
        context needs one flavor.
        """
        from pypto.language.typing._subscript import subscript_read  # noqa: PLC0415 - local: avoid the cycle

        return subscript_read(self, indices, tile=True)

    def __setitem__(
        self,
        key: _SubscriptKey,
        value: "Tile",
    ) -> None:
        """``t[i:j] = src`` → ``tile.assemble(t, src, offsets)`` (functional rebind).

        The traced frontend's subscript-write sugar for tile targets (the
        AST's ``_parse_subscript_assignment`` tile path): the source rewriter
        leaves the statement as a plain Python item assignment on the
        facade, so this hook fires during the trace, asks the
        active Frame (``active_frame()``) to resolve the target ``ir.Var``,
        and routes the resulting source-name bind through the trace
        interpreter. The written indices follow the numpy rule (offsets
        always full-rank: a bare ``t[:]`` expands to every axis, dims past
        the written indices are implicit full slices); scalar-index
        rank-reducing writes are loud — the AST's full rank-lifting surface
        (unit-axis padding for the tile 2D floor) is not ported. Mirrors
        ``Tensor.__setitem__``.
        """
        from pypto.language.ambient import active_frame  # noqa: PLC0415
        from pypto.language.op import tile as _tile_dsl  # noqa: PLC0415
        from pypto.language.typing._subscript import subscript_write_offsets  # noqa: PLC0415

        frame = active_frame()
        if frame is None:
            raise TypeError(
                "Tile item assignment is only supported inside a traced kernel (the tracing frontend)"
            )
        offsets = subscript_write_offsets(self.unwrap(), key, kind="Tile")
        new = _tile_dsl.assemble(self, value, offsets)
        owner, name = frame.resolve_staged_local(self.unwrap())
        if owner is None:
            raise TypeError(
                "Tile item-assignment target must be a bound local name, "
                f"got an expression-backed Tile ({self!r})"
            )
        owner.rt_bind(name, new, via=frame)

    def __repr__(self) -> str:
        """String representation."""
        if self._expr is not None:
            return f"Tile(expr={self._expr})"
        parts = [f"[{self.shape}]", f"{self.dtype}"]
        if self.memref is not None:
            parts.append(f"{self.memref}")
        if self.memory_space is not None:
            parts.append(f"{self.memory_space}")
        if self.tile_view is not None:
            parts.append(f"{self.tile_view}")
        return f"Tile[{', '.join(parts)}]"


__all__ = ["Tile"]
