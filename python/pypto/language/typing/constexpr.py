# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-time parameter marker for PyPTO Language DSL."""

from typing import TYPE_CHECKING, Any, TypeAlias, cast


class ConstexprMarker:
    """Annotation marking a parameter the compiler resolves at specialization time.

    A ``pl.Scalar[dtype]`` parameter is a **runtime value**: it survives into the
    generated program, its value arrives at dispatch, and one artifact serves
    every value. ``pl.constexpr`` is the opposite end of that axis — the call
    site's value is folded into the body at every use, joins the compilation
    key, and the parameter is **absent from the generated program and from the
    dispatch ABI**. Nothing is passed for it at run time.

    Use it for a value the body needs the compiler to see: a tile shape, an
    unroll extent, or a branch the compiler should resolve away.

    A constexpr parameter is exactly a module-level constant supplied per call
    site, so it folds by the same rule and admits the same value types: ``int``,
    ``float``, ``bool``, ``str``, a ``DataType``, an enum member such as
    ``pl.Mem.Vec``, and lists of those. A value the specializer cannot render
    into source is rejected with a diagnostic naming the parameter.

    Carries no dtype: unlike ``pl.Scalar[dtype]`` it never becomes an IR
    parameter, so there is no type for the ABI to agree on.

    Examples:
        >>> import pypto.language as pl
        >>>
        >>> @pl.jit  # doctest: +SKIP
        ... def kernel(
        ...     x: pl.Tensor[[64], pl.FP32],
        ...     out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ...     scale: pl.Scalar[pl.FP32],   # runtime: one artifact, any value
        ...     BLOCK: pl.constexpr,         # folded: one artifact per value
        ... ): ...
        >>>
        >>> kernel(x, out, scale=1.0, BLOCK=128)  # doctest: +SKIP
        >>> kernel(x, out, scale=2.0, BLOCK=128)  # reuses that artifact
        >>> kernel(x, out, scale=2.0, BLOCK=256)  # a second artifact
    """

    def __repr__(self) -> str:
        """Return the marker's canonical spelling."""
        return "pl.constexpr"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Reject a call, which almost always means a missing annotation colon.

        Raises:
            TypeError: Always — the marker is an annotation, not a function.
        """
        raise TypeError(
            "pl.constexpr is an annotation, not a function. Annotate the parameter "
            "with it -- 'BLOCK: pl.constexpr' -- and pass the value at the call site."
        )


def constexpr_value_key(value: Any) -> tuple[Any, ...] | None:
    """Canonical identity for a value a ``pl.constexpr`` slot accepts, or None.

    The docstring above states the admissible value types; this function is
    that statement made executable, so every frontend answers the same
    question with the same rule. The tracer bakes the value into the body
    directly (no source rendering), so unlike the parsing frontend it does
    not need the value reachable as a ``pl.``/``pld.`` symbol — but the value
    *set* is the one the contract documents, and a ``DataType`` or enum
    member outside it is refused here rather than silently baked.

    The return value doubles as a hashable variant key: two calls baking
    different constants must never share a traced variant, and lists (legal
    constexpr values, unhashable as-is) fold into nested tuples.

    Args:
        value: The call-site value bound to a constexpr parameter

    Returns:
        A canonical ``(kind, identity)`` tuple, or ``None`` when ``value`` is
        outside the documented set
    """
    import enum  # noqa: PLC0415 — keep the stdlib dependency lazy like the dtype one below
    from collections.abc import Sequence  # noqa: PLC0415

    if value is None or isinstance(value, (bool, int, float, str)):
        return ("const", type(value).__name__, value)
    if isinstance(value, enum.Enum):
        return ("enum", f"{type(value).__module__}.{type(value).__qualname__}", value.name)
    try:
        from pypto.pypto_core import DataType  # noqa: PLC0415 — cycle-free laziness

        if isinstance(value, DataType):
            return ("dtype", str(value))
    except ImportError:  # pragma: no cover — the extension is a hard runtime dep
        return None
    if isinstance(value, (list, tuple)):
        parts: list[Any] = [constexpr_value_key(element) for element in cast("Sequence[Any]", value)]
        if any(part is None for part in parts):
            return None
        return ("seq" if isinstance(value, list) else "tuple", tuple(parts))
    return None


if TYPE_CHECKING:
    # The marker is a value, so a type checker would reject ``BLOCK: pl.constexpr``
    # as "variable not allowed in type expression". Present it as the permissive
    # alias instead, which is also what the annotation means: the parameter
    # accepts any compile-time constant. Same device ``pl.Out`` / ``pl.InOut``
    # use for their runtime-passthrough wrappers.
    constexpr: TypeAlias = Any
else:
    constexpr = ConstexprMarker()
    """Singleton ``ConstexprMarker`` — see the class docstring."""


__all__ = ["ConstexprMarker", "constexpr", "constexpr_value_key"]
