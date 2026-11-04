# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit storage for detached operation results; assignments never choose storage."""

from collections.abc import Sequence
from typing import overload

from pypto.pypto_core import ir

from .ambient import active_frame
from .typing import Tile, Tuple


def validate_result_storage(producer: ir.Call, destinations: Sequence[ir.MemRef | None]) -> None:
    """Refuse ill-formed destinations at the facade, before the IR checks.

    Both frontends reach ``ir.bind_result_storage`` through this helper (the
    tracing facade and the parser's ``pl.bind_storage`` spelling), so a
    non-pinned destination, an unselected multi-slot allocation or an
    unresolved output memory space is reported with its position instead of
    the terse registry CHECK. The registry keeps its own checks as the
    backstop for direct IR construction.

    Args:
        producer: The detached producer call whose results the destinations name.
        destinations: One destination per result; ``None`` keeps automatic
            placement for that result.

    Raises:
        TypeError: A destination is not a pinned allocation declaration, or a
            multi-slot allocation was given without selecting a slot.
        ValueError: A bound result has no resolved output memory space.
    """
    result_type = producer.type
    types = result_type.types if isinstance(result_type, ir.TupleType) else [result_type]
    for i, destination in enumerate(destinations):
        if destination is None:
            continue
        if not isinstance(destination, ir.MemRef):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"bind_storage destination {i} must be a MemRef, got {type(destination).__name__}"
            )
        if not destination.is_pinned_:
            raise TypeError(
                f"bind_storage destination {i} must be a pinned allocation declaration "
                "(pl.MemRef(...), optionally with a selected slot); got an already-resolved MemRef"
            )
        if destination.slot_count_ > 1 and destination.slot_index_ is None:
            raise TypeError(
                f"bind_storage destination {i} must select a slot of the "
                f"{destination.slot_count_}-slot allocation "
                f"{destination.base_.name_hint!r} before binding (e.g. pool[0] or pool[i % 2])"
            )
        if i < len(types):
            result_type = types[i]
            if isinstance(result_type, ir.TileType) and not result_type.memory_space:
                raise ValueError(
                    f"bind_storage requires a resolved output memory space for result {i}; "
                    "pass target_memory to the producer when placing it in explicit storage "
                    "(e.g. pl.load(..., target_memory=pl.Mem.Vec))"
                )


@overload
def bind_storage(value: Tile, memref: ir.MemRef) -> Tile: ...
@overload
def bind_storage(value: Tuple, memref: Sequence[ir.MemRef | None]) -> Tuple: ...
def bind_storage(value: object, memref: ir.MemRef | Sequence[ir.MemRef | None]) -> Tile | Tuple:
    """Place a producer's results in declared storage, preserving inferred value types.

    Args:
        value: A detached allocating operation result. An already bound variable,
            view, or in-place update cannot be redirected to different storage.
        memref: One allocation/slot for a Tile, or one reference per tuple result.
            None entries retain automatic placement for those tuple results.

    Returns:
        The same producer with explicit result storage. This introduces no copy.

    Raises:
        TypeError: The value is not a detached Tile or staged tuple producer.
        ValueError: Storage conflicts with the producer's allocation contract.
    """
    if not isinstance(value, (Tile, Tuple)):
        raise TypeError("bind_storage requires a detached Tile or staged tuple producer")
    producer = value.unwrap()
    if not isinstance(producer, ir.Call):
        raise TypeError("bind_storage requires a detached Tile or staged tuple producer")
    if isinstance(memref, ir.MemRef):
        destinations = [memref]
    elif isinstance(memref, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance] — None is documented for the tuple spelling, not here
        destinations = list(memref)
    else:
        # None (and any non-sequence) is the tuple spelling's per-result
        # "automatic placement" marker, not a valid single-Tile destination.
        raise TypeError(
            "bind_storage requires a MemRef or a sequence of MemRef/None per "
            f"tuple result, got {type(memref).__name__}"
        )
    validate_result_storage(producer, destinations)
    frame = active_frame()
    if frame is not None:
        destinations = [frame.resolve_storage(m) if m is not None else None for m in destinations]
    result = ir.bind_result_storage(producer, destinations)
    return Tile(expr=result) if isinstance(value, Tile) else Tuple(expr=result)
