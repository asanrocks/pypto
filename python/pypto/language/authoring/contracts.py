# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""The authoring contracts — the typed specifications a recipe pushes.

``ParamSpec`` / ``FunctionContract`` declare a function unit's ABI (the
typed dual of the jit envelope's decoration kwargs); ``CollectedBody`` is
the receipt of one collection; ``FunctionArtifact`` is the inert, finalized
result awaiting a placement (``emit_call`` / ``materialize_entry``).
Nothing here executes — the session (:mod:`pypto.language.authoring.session`)
owns the lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pypto.pypto_core import ir

if TYPE_CHECKING:
    from pypto.language.authoring.session import _FunctionUnit

__all__ = ["CollectedBody", "FunctionArtifact", "FunctionContract", "ParamSpec"]


# -----------------------------------------------------------------------------------------------------------
# The contracts
# -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """One declared parameter of a :class:`FunctionContract`.

    The typed triple the entrance foundation's ``bind_typed_parameters``
    consumes: name, ``ir.Type``, and direction. A derived-signature unit (a
    contract without ``params``) mints the same triples from the collection
    arguments' staged types.
    """

    name: str
    type: ir.Type
    direction: ir.ParamDirection = ir.ParamDirection.In


@dataclass(frozen=True)
class FunctionContract:
    """The specification of one function a recipe pushes.

    The typed dual of the jit envelope's decoration kwargs: the function's
    classification (``function_type`` / ``level`` / ``role`` / ``attrs``),
    its name, and — for a declared ABI — its parameters and returns. A
    contract without ``params`` derives the signature from the collection
    arguments' staged types (the parser's annotationless-callee rule: types
    from the call site, values at the call).
    """

    name: str
    function_type: ir.FunctionType = ir.FunctionType.Opaque
    level: ir.Level | None = None
    role: ir.Role | None = None
    attrs: dict[str, Any] | None = None
    params: tuple[ParamSpec, ...] | None = None
    returns: tuple[ir.Type, ...] | None = None
    strict_ssa: bool = False
    requires_runtime_binding: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"a FunctionContract needs a non-empty name, got {self.name!r}")
        if self.params is not None:
            seen: set[str] = set()
            for spec in self.params:
                if not isinstance(spec, ParamSpec):
                    raise TypeError(f"FunctionContract params entries must be ParamSpec, got {spec!r}")
                if spec.name in seen:
                    raise TypeError(f"FunctionContract declares parameter {spec.name!r} twice")
                seen.add(spec.name)


# -----------------------------------------------------------------------------------------------------------
# The results
# -----------------------------------------------------------------------------------------------------------


class CollectedBody:
    """The result of one :meth:`FunctionContext.collect_body`.

    Carries the owning unit (the identity a ``finalize`` checks) and the
    body's own return values as they crossed back to the recipe. The builder
    function is already closed at this point — collection is the runner's
    normal completion — so this object is the *receipt*, not a handle:
    nothing further can be collected into the unit, and the contract step is
    :meth:`FunctionContext.finalize`."""

    __slots__ = ("_unit", "value")

    def __init__(self, unit: _FunctionUnit, value: object) -> None:
        self._unit = unit
        self.value = value

    @property
    def body_name(self) -> str:
        """The collected body's display name."""
        return self._unit.body_name


class FunctionArtifact:
    """One finalized function — the inert result of :meth:`FunctionContext.finalize`.

    Immutable and safe to pass upward. Nothing happens with it until a
    placement says so: :meth:`RecipeContext.emit_call` (the caller-side call
    site) or :meth:`RecipeContext.materialize_entry` (the root entry mark)."""

    __slots__ = ("_function", "_contract", "_entry")

    def __init__(self, function: ir.Function, contract: FunctionContract, *, entry: bool = False) -> None:
        self._function = function
        self._contract = contract
        self._entry = entry

    @property
    def function(self) -> ir.Function:
        """The built ``ir.Function``."""
        return self._function

    @property
    def contract(self) -> FunctionContract:
        """The contract the function was pushed under."""
        return self._contract

    @property
    def is_entry(self) -> bool:
        """Whether ``materialize_entry`` marked this artifact as the root entry."""
        return self._entry
