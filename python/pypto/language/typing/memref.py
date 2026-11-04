# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MemRef wrapper type for PyPTO Language DSL.

DSL subclass of ``ir.MemRef`` supporting traced declaration naming and
expression-valued slot selection. Constructor overloads also widen ``base``
and ``byte_offset`` for the printer's ``@pl.program`` forms:

* ``base`` also accepts the ``pl.Ptr`` wrapper — for
  ``pl.MemRef(ptr_var, offset, size)`` where ``ptr_var`` is annotated as
  ``pl.Ptr`` (what ``pl.tile.alloc`` / ``pld.alloc_window_buffer`` return).
* ``byte_offset`` also accepts ``Scalar`` — the printer renders a non-constant
  offset as a ``Scalar`` arithmetic expression (e.g. ``pos * 128 * 4``), and a
  constant offset as ``pl.const(0, pl.INT64)``, which is statically a ``Scalar``.
"""

from typing import TYPE_CHECKING, Any, overload

from pypto.language.ambient import active_frame
from pypto.pypto_core import DataType

if TYPE_CHECKING:
    from pypto.language.protocol import TraceContext

from pypto.pypto_core.ir import (
    ConstInt,
    Expr,
    MemorySpace,
    ScalarType,
    Span,
    Var,
)
from pypto.pypto_core.ir import (
    MemRef as _IrMemRef,
)

from .ptr import Ptr
from .scalar import Scalar

# A printed MemRef byte offset is either a constant (rendered by the printer as
# ``pl.const(...)``, statically a ``Scalar``), a DSL ``Scalar`` arithmetic
# expression, or a raw IR ``Expr`` when the MemRef is built programmatically.
# ``int`` stays for MemRefs built programmatically with a plain offset literal.
_ByteOffset = int | Expr | Scalar


class MemRef(_IrMemRef):
    """DSL-level memory reference accepting ``pl.Ptr`` bases and Scalar offsets.

    Runtime slot selection retains the integer IR expression. During tracing,
    declaration bindings acquire a source name without emitting a data value.
    Constructor overloads additionally allow printed IR with ``pl.Ptr`` bases
    and ``Scalar`` offsets to type-check when reloaded as ``@pl.program``.

    Called with **no offset and size**, it declares an allocation of your own
    rather than describing an existing one: size and address are left for
    ``InitMemRef`` to derive, and the compiler's opportunistic reuse never packs
    anything else into it. The allocation takes the name of the variable it is
    bound to, so the name is written once::

        scratch = pl.MemRef()

        t0: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.load(x, [0, 0], [64, 64])
        t1: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.exp(t0)

    Pass ``slots=N`` for N equally-sized slots of one allocation, then pick one by
    subscript. The slots are contiguous and identically sized, so rotating through
    them is a ping-pong the packer cannot collapse::

        l0c = pl.MemRef(slots=2)

        ping: pl.Tile[[64, 64], pl.FP32, l0c[0], pl.Mem.Acc] = pl.tile.matmul(q, b0)
        pong: pl.Tile[[64, 64], pl.FP32, l0c[1], pl.Mem.Acc] = pl.tile.matmul(q, b1)

    The subscript is an ordinary index expression, so it may be a **runtime**
    value — a rotation needs no unrolling::

        for i in pl.range(N):
            t: pl.Tile[[64, 64], pl.FP32, l0c[i % 2], pl.Mem.Acc] = pl.tile.matmul(q, b)

    A constant index folds into a static address; a runtime one becomes the tile's
    address at run time. Nothing here is tied to a particular loop form — the
    index is just an expression, so `pl.range`, a `pl.pipeline` sub-index, or a
    function parameter all work.

    Reference it by variable, so a misspelling is a ``NameError`` rather than a
    second allocation. Since the variable *is* the name, one declaration may not
    be reached through two names (``b = a``) and two declarations may not claim
    one name; both are rejected. ``pl.MemRef("other")`` names it explicitly,
    overriding the variable — that is the form the IR printer emits, so a dumped
    program reparses without a surrounding Python scope.

    Co-liveness is checked **per slot**: two tiles on different slots are meant to
    be live together (that is the ping-pong), and only two tiles landing on the
    *same* slot can corrupt each other. A runtime index has no static slot to
    attribute a tile to, so the check is skipped there — the rotation is yours to
    get right — while isolation from every other allocation still holds. Tiles
    sharing one declared allocation must also agree on memory space.

    Declaring an allocation inside a ``pl.pipeline(stage=2)`` body is rejected —
    the cloned stages would make a tile co-live with itself. Declaring slots and
    asking the compiler to multi-buffer are alternatives, not layers: to
    hand-manage a level, drive it with ``pl.range`` and give it its own slots.

    Note: ``pl.MemRef(...)`` calls inside a ``@pl.program`` body are resolved
    by the parser (``parser/type_resolver.py``), not dispatched through this
    ``__init__``. A ``Scalar`` byte offset is therefore only ever seen by
    pyright; it never reaches the underlying ``ir.MemRef`` constructor.
    """

    def __trace_bind__(self, ctx: "TraceContext", name: str) -> None:
        """Record a declaration's source name without making it a staged data value."""
        frame = active_frame()
        if frame is not None and self.is_pinned_:
            frame.register_storage(self, name)

    def __getitem__(self, slot: object) -> _IrMemRef:
        """Select one slot of a multi-slot declared allocation.

        The slot is an expression, not a host array lookup. Preserve a staged
        index so SSA substitution and memory planning can resolve its address.

        Args:
            slot: Slot index — a constant, or an index expression in a program body

        Returns:
            The same declaration bound to that slot
        """
        if self.slot_count_ <= 1:
            raise ValueError("MemRef slot selection requires slots > 1")
        if isinstance(slot, bool):
            raise TypeError("MemRef slot index must be an integer scalar, got bool")
        if isinstance(slot, int):
            index = ConstInt(slot, DataType.INDEX, self.span)
        elif isinstance(slot, Scalar):
            index = slot.unwrap()
        elif isinstance(slot, Expr):
            index = slot
        else:
            raise TypeError(f"MemRef slot index must be an integer scalar, got {type(slot).__name__}")
        if not isinstance(index.type, ScalarType) or not index.type.dtype.is_int():
            raise TypeError(f"MemRef slot index must be an integer scalar, got {index.type}")
        if isinstance(index, ConstInt) and not 0 <= index.value < self.slot_count_:
            raise ValueError(f"MemRef slot index {index.value} is outside [0, {self.slot_count_})")
        if self.is_pinned_:
            return _IrMemRef(self.base_, 0, 0, self.span, is_pinned=True, slots=self.slot_count_, slot=index)
        return _IrMemRef(
            self.base_, self.byte_offset_, self.size_, self.span, slots=self.slot_count_, slot=index
        )

    @overload
    def __init__(self, span: Span = ..., slots: int = ...) -> None: ...
    @overload
    def __init__(self, name: str, span: Span = ..., slots: int = ...) -> None: ...
    # The resolved forms take ``slots`` / ``slot`` too: a MemRef stays slot k of an
    # N-slot allocation after InitMemRef has folded the index into the offset, and
    # `pl.MemRef(base, offset, size, slots=N)[k]` is what the printer emits for one.
    @overload
    def __init__(
        self,
        base: Var,
        byte_offset: _ByteOffset,
        size: int,
        span: Span = ...,
        slots: int = ...,
        slot: "Expr | int | Scalar | None" = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        base: str,
        byte_offset: _ByteOffset,
        size: int,
        span: Span = ...,
        slots: int = ...,
        slot: "Expr | int | Scalar | None" = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        base: Ptr,
        byte_offset: _ByteOffset,
        size: int,
        span: Span = ...,
        slots: int = ...,
        slot: "Expr | int | Scalar | None" = ...,
    ) -> None: ...
    @overload
    def __init__(self, addr: int, size: int, id: int, span: Span = ...) -> None: ...
    @overload
    def __init__(
        self, memory_space: MemorySpace, addr: Expr | int, size: int, id: int, span: Span = ...
    ) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


__all__ = ["MemRef"]
