# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parameter direction wrapper types for PyPTO Language DSL.

These types are used as param direction annotations:
- InOut[T]: Read-write parameter
- Out[T]: Write-only output parameter

The default direction (In) requires no wrapper.

At runtime, ``Out[T]`` / ``InOut[T]`` carry their direction **on the returned
value**: a shallow copy of a facade instance, or a dynamic subclass for a
bare facade class (``Out[pl.Tensor]``). ``typing.get_type_hints`` therefore
preserves the direction, and the signature reader
(``decorator.signature.param_directions``) reads the annotation values
directly — no source reconstruction, so exec'd / generated kernels keep
their directions (the tracing frontend's contract: the executed path is
the source of truth). The copy/subclass stays a valid facade for every
other consumer (``isinstance`` checks and ``annotation_to_type`` see the
same shape/dtype).

For type checkers, they are Annotated[T, ...] aliases so Out[Tensor]
resolves to Tensor.
"""

import copy
from typing import TYPE_CHECKING, TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from typing import Annotated

    InOut = Annotated[T, "InOut"]
    Out = Annotated[T, "Out"]
else:

    class _DirectionWrapper:
        def __class_getitem__(cls, item: T) -> T:
            """Carry the direction on the returned annotation value.

            A facade instance gets a shallow copy carrying
            ``_param_direction``; a bare facade class gets a dynamic
            subclass (the copy of a class is the class itself, so the
            direction would leak to every user of it). Values that cannot
            carry the attribute (immutables, non-subclassable externals)
            pass through unchanged — such params die loudly at the
            signature validation anyway.
            """
            direction = cls.__name__
            if isinstance(item, type):
                try:
                    directed = type(f"_{direction}{item.__name__}", (item,), {"_param_direction": direction})
                except TypeError:
                    return item
                directed.__module__ = item.__module__
                return directed
            try:
                directed = copy.copy(item)
                directed._param_direction = direction
                return directed
            except (AttributeError, TypeError):
                return item

    class InOut(_DirectionWrapper):
        """Wrapper for InOut parameter direction in parameter annotations.

        Usage::

            def kernel(output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]) -> ...:
        """

    class Out(_DirectionWrapper):
        """Wrapper for Out parameter direction in parameter annotations.

        Usage::

            def kernel(result: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> ...:
        """


__all__ = ["InOut", "Out"]
