# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tuple value type for PyPTO Language DSL.

Dual-purpose, mirroring ``Scalar`` / ``Tensor``:

- **Annotation mode** — ``pl.Tuple[T1, T2, ...]`` (and the legacy
  ``pl.Tuple([T1, T2, ...])``) in a function signature. Resolves to
  ``ir.TupleType`` (the return annotation is the inspection contract).
- **Runtime mode** — ``pl.Tuple(a, b, ...)`` yields a staged tuple value:
  an ``ir.MakeTuple`` wrapped in this facade. This is the authentic staged
  representation of a multi-value return (a host Python tuple
  ``(a, b)`` is a *host* container of facades, not a staged value; the
  staged form is ``pl.Tuple(a, b)``, the operator that yields a
  ``TupleType`` value). Element access ``t[i]`` desugars to
  ``ir.TupleGetItemExpr``; unpacking ``a, b = t`` iterates the projections.

The type-sound counterpart to the multi-value ``ReturnStmt([a, b])`` list
the AST parser desugars from ``return (a, b)``. ``pl.Tuple`` retires the
host-tuple type lie at the surface (the retirement rule — multi-value
return *duplicates* what ``pl.Tuple`` expresses).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pypto.language.typing.type_utils import wrap_expr

if TYPE_CHECKING:
    from pypto.pypto_core.ir import Expr as _Expr

_T = TypeVar("_T")


def _is_runtime_arg(arg: Any) -> bool:
    """A runtime ``pl.Tuple(a, b, ...)`` argument: a staged facade (has the
    ``unwrap`` dunder) or a raw ``ir.Expr``. Annotation-mode args (type
    wrappers, plain classes) are neither."""
    if hasattr(arg, "unwrap"):
        return True
    from pypto.pypto_core import ir  # noqa: PLC0415 (avoid import cycle)

    return isinstance(arg, ir.Expr)


def _unwrap_arg(arg: Any) -> Any:
    if hasattr(arg, "unwrap"):
        return arg.unwrap()
    return arg


class TupleMeta(type):
    """Metaclass enabling ``pl.Tuple[T1, T2, ...]`` annotation syntax and
    ``pl.Tuple(a, b, ...)`` runtime construction."""

    def __getitem__(cls, item: Any) -> Tuple:  # noqa: PYI019 - metaclass
        # isinstance cannot parameterize an Any input, so the tuple branch
        # goes through a cast — the annotation surface accepts any subscript
        # payload and normalizes it structurally.
        if isinstance(item, tuple):
            types: tuple[Any, ...] = cast("tuple[Any, ...]", item)
        else:
            types = (item,)
        inst = type.__call__(cls)
        inst._types = types
        inst._annotation_only = True
        return inst

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Internal runtime construction: ``Tuple(expr=<MakeTuple|Call>)`` —
        # the Sema's read-path re-wrap (``_rewrap_as_base_facade``) and ``_call_facade``
        # build an expr-wrapping facade this way; route to the expr= path.
        if "expr" in kwargs:
            inst = type.__call__(cls)
            inst._expr = kwargs["expr"]
            inst._annotation_only = False
            return inst
        # Runtime mode: ``pl.Tuple(a, b, ...)`` with staged/Expr args →
        # a MakeTuple-wrapping facade. Detect the runtime shape (≥1 arg,
        # every arg a facade/Expr) before falling back to the legacy
        # annotation call ``pl.Tuple([T1, T2])``.
        if args and not kwargs and all(_is_runtime_arg(a) for a in args):
            from pypto.pypto_core import ir  # noqa: PLC0415 (avoid import cycle)

            elements: list[Any] = [_unwrap_arg(a) for a in args]
            expr = ir.MakeTuple(elements, ir.Span.unknown())
            inst = type.__call__(cls)
            inst._expr = expr
            inst._annotation_only = False
            return inst
        # Legacy annotation call: ``pl.Tuple([T1, T2, ...])``.
        inst = type.__call__(cls)
        if args:
            # isinstance cannot parameterize the Any arg — cast the list form.
            inst._types = tuple(cast("list[Any]", args[0])) if isinstance(args[0], list) else args
        inst._annotation_only = True
        return inst


class Tuple(metaclass=TupleMeta):
    """Tuple value/annotation for PyPTO Language DSL.

    Annotation: ``result: pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Scalar[pl.INT32]]``
    Runtime:    ``return pl.Tuple(a, b)`` — the staged multi-value return.
    """

    _types: tuple[Any, ...] = ()
    _expr: Any = None
    _annotation_only: bool = True
    # Lazily-created shared emit state for the ``y, z = call`` unpack case
    # (a raw Call/MakeTuple underlying, not yet bound to an ``ir.Var``). The
    # Var case (``ret = call`` then ``ret[i]``) projects directly — no emit.
    _emit_state: Any = None
    # The value-owned bind-hook slot, installed instance-level
    # by the structural emit paths (``_emit.py`` projections, the
    # single-LHS submit facade in ``constructs/submit.py``). Annotation-only — no
    # default — so a bare Tuple does not satisfy ``__trace_bind__`` protocol
    # membership until a hook is actually installed. ``Callable[..., object]``:
    # the receiver is the explicit tracing context, a flow-tier type this tier
    # cannot name.
    __trace_bind__: Callable[..., object]

    def __init__(self) -> None:  # noqa: D401 - metaclass builds instances
        # All state is set by the metaclass / class attrs; ``__init__`` is a
        # no-op so ``type.__call__(cls)`` (which invokes it) stays cheap.
        pass

    @classmethod
    def __class_getitem__(cls, item: type[_T] | tuple[type[_T], ...] | Any) -> tuple[_T, ...]:
        """The static-checker hook for ``pl.Tuple[T1, T2, ...]`` notation.

        Runtime never calls this — ``TupleMeta.__getitem__`` wins the
        subscript protocol — but pyright resolves class subscripts through
        it and has no metaclass fallback (the Tensor/Tile/Array/Scalar
        precedent). The static view of a ``pl.Tuple[T1, ...]`` annotation is
        the *host* tuple of its element types: the ergonomic source forms
        ``return a, b`` and ``x, y = call`` are plain Python tuples, and the
        symbolic ``pl.Tuple(a, b)`` facade (an ``Any`` at this tier) remains
        assignable to it.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        if self._annotation_only:
            inner = ", ".join(repr(t) for t in self._types)
            return f"Tuple[{inner}]"
        return f"Tuple(expr={self._expr})"

    # —— runtime mode: facade protocol ————————————————————————————————————
    @classmethod
    def _runtime(cls, expr: Any) -> Tuple:
        """Build a runtime Tuple facade wrapping ``expr`` (a MakeTuple or a
        TupleType-typed Call) — bypasses the metaclass's annotation/runtime
        detection (the internal construction path)."""
        inst = type.__call__(cls)
        inst._expr = expr
        inst._annotation_only = False
        return inst

    def unwrap(self) -> _Expr:
        if self._annotation_only:
            raise RuntimeError("Cannot unwrap annotation-only Tuple")
        assert self._expr is not None
        return self._expr

    @property
    def type(self) -> Any:
        """The IR type of the wrapped MakeTuple (a ``TupleType``)."""
        return self.unwrap().type

    def __trace_replace_expr__(self, expr: Any) -> Tuple:
        """Replace a detached tuple producer while preserving shared emit state."""
        self._expr = expr
        if self._emit_state is not None:
            self._emit_state.node = expr
            self._emit_state.temp = None
        self._annotation_only = False
        return self

    def __getitem__(self, index: int) -> Any:
        """``t[i]`` → a facade of element ``i``'s type wrapping a
        ``TupleGetItemExpr``. If the underlying expr is already a Var (a
        staged local from ``ret = call``), project it directly; if it is a raw
        Call/MakeTuple (the ``y, z = call`` unpack case, not yet bound),
        attach ``__trace_bind__`` so the Sema names the shared node once as
        a ``_tuple_tmp`` temp (mirrors the AST parser's ``_bind_unpack_target``)
        and the projections reference the temp — the type-sound
        ``_tuple_tmp`` hack (lent-builder, narrowed to a
        value-side side-effect)."""
        from pypto.language.typing._emit import _EmitState, _structural_projection  # noqa: PLC0415
        from pypto.pypto_core import ir  # noqa: PLC0415

        t = self.unwrap()
        assert isinstance(t.type, ir.TupleType)
        elt_t = t.type.types[index]
        if isinstance(t, ir.Var):
            # Already a staged local (e.g. ``ret = call`` then ``ret[i]``) — the
            # Var IS the temp; project directly, no emit.
            return _wrap_element(ir.TupleGetItemExpr(t, index, ir.Span.unknown()), elt_t)
        # Raw Call/MakeTuple (the ``y, z = call`` unpack) — share a temp.
        if self._emit_state is None:
            self._emit_state = _EmitState(t)
        elt = ir.TupleGetItemExpr(t, index, ir.Span.unknown())  # placeholder; re-wrapped on emit
        return _structural_projection(_wrap_element(elt, elt_t), self._emit_state, index, "tuple_tmp")

    def __iter__(self) -> Any:
        """``a, b = t`` → iterate the element projections (in order). Shares
        the ``_tuple_tmp`` temp across all projections when the underlying
        expr is a raw Call/MakeTuple (the unpack case); projects directly
        when it is a Var (already staged).

        The element types are dynamic facades derived from the tuple's IR
        element types — unknowable statically — so the generator is typed
        ``Any``; the bind hooks carry the real types.
        """
        from pypto.language.typing._emit import _EmitState, _structural_projection  # noqa: PLC0415
        from pypto.pypto_core import ir  # noqa: PLC0415

        t = self.unwrap()
        assert isinstance(t.type, ir.TupleType)
        if isinstance(t, ir.Var):
            for i, elt_t in enumerate(t.type.types):
                yield _wrap_element(ir.TupleGetItemExpr(t, i, ir.Span.unknown()), elt_t)
            return
        if self._emit_state is None:
            self._emit_state = _EmitState(t)
        for i, elt_t in enumerate(t.type.types):
            elt = ir.TupleGetItemExpr(t, i, ir.Span.unknown())
            yield _structural_projection(_wrap_element(elt, elt_t), self._emit_state, i, "tuple_tmp")


def _wrap_element(expr: Any, t: Any) -> Any:
    """Wrap a ``TupleGetItemExpr`` projection in the facade matching its
    element type (the type→facade funnel — ``wrap_expr``)."""
    return wrap_expr(expr, t)


__all__ = ["Tuple"]
