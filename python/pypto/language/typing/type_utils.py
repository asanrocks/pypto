# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The facade-conversion funnel — one sanctioned path for
structural staged↔host conversion.

Three axes, three tiers:

  A — facade↔``ir.Expr`` (this module): ``to_expr`` / ``from_expr`` / ``wrap_expr``
  B — host→typed-constant (the operand boundary, ``_host.coerce_host_scalar``
      / ``const``); the *only* automatic host-literal lifting
  C — static-fold/query (IR-tier): ``to_int`` / ``as_int``

The IR tier (``ir/``, ``ir.Expr``) stays conversion-free — the funnel's
lower edge. ``_UnwrapsToExpr`` is the canonical structural-unwrap Protocol
(relocated from ``ir/op/system_ops.py``); the IR-op tier's local copy is
grandfathered debt (removed when ``_consumer_buf_operand`` lifts
to ``language/op/`` and IR-op takes ``Expr``-only).

Discipline: the funnel symbols are *defined only* here; the A-axis
conditional unwrap and the C-axis ``isinstance(_, ConstInt)`` pattern stay
out of ``language/typing/``'s callers (single-line mechanical patterns);
``*_unchecked`` call sites must stay deliberate — the skipped
``isinstance`` dispatch is the audit point.

The B-axis discipline: ``to_expr`` does NOT coerce
``int``/``float`` to ``ConstInt`` — that is the operand boundary
(``coerce_host_scalar`` / ``from_int``), the *only* automatic
host-literal lifting. Unifying A and B would silently reify literals
everywhere, which the dialect forbids. ``to_expr`` is typed ``-> Expr`` (the
"total" contract); a bare ``int``/``float`` passes through
uncoerced at runtime — benign, since every IR op that accepts an ``Expr``
operand also accepts a bare ``int``/``float`` via the C++ deduce tier — so
the one ``cast`` in the passthrough branch is the explicit acknowledgment
of that B-axis reality, not a type hole.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import TYPE_CHECKING, Protocol, cast, get_args, get_origin, runtime_checkable

from pypto.pypto_core import DataType
from pypto.pypto_core import arith as _arith
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import Expr

if TYPE_CHECKING:
    # ``Scalar`` is the return type of ``from_int``; a top-level import would
    # cycle (``scalar`` imports ``type_utils`` for ``to_expr``), so it is only
    # needed for type-checking.
    from pypto.language.typing.scalar import Scalar

__all__ = [
    "_UnwrapsToExpr",
    "annotation_to_return_types",
    "annotation_to_type",
    "apply_result_type",
    "as_int",
    "from_expr",
    "from_int",
    "register_facade",
    "to_expr",
    "to_expr_unchecked",
    "to_int",
    "to_int_unchecked",
    "wrap_expr",
]


@runtime_checkable
class _UnwrapsToExpr(Protocol):
    """A facade that carries an ``ir.Expr`` accessible via ``unwrap()``.

    The canonical structural-unwrap Protocol, relocated from
    ``ir/op/system_ops.py`` so the funnel owns it. Structural (duck-typed)
    — every ``Tensor``/``Tile``/``Scalar``/``Array``/``Ptr``/``DynVar``/
    ``CommCtx`` facade satisfies it without inheritance. The
    ``runtime_checkable`` decorator keeps ``isinstance(x, _UnwrapsToExpr)``
    usable where a Protocol check is still wanted (e.g. ``to_expr``'s
    dispatch).
    """

    def unwrap(self) -> Expr: ...


# ---- Axis A — facade↔Expr (total) ---------------------------------------
# A pure structural relabel: a facade *is* an ``Expr`` wrapper; unwrapping
# never changes semantics. This is the safe, uniform, zero-cost axis —
# the one with the most copy-paste (~55 sites) before the funnel.


def to_expr(x: _UnwrapsToExpr | Expr | int | float) -> Expr:
    """Unwrap a facade to its ``ir.Expr``; pass ``ir.Expr`` and bare host
    primitives through unchanged (A-axis).

    Leaves bare ``int``/``float`` untouched — the C++ deduce tier types
    them by context (the B-axis discipline: ``to_expr`` must not
    coerce literals; that is ``coerce_host_scalar`` / ``from_int``, the
    operand boundary). Typed ``-> Expr`` (the "total" contract): a
    bare primitive returns as-is at runtime, which every ``Expr``-accepting
    IR op also accepts — the ``cast`` below is the explicit acknowledgment
    of that, not a type hole.
    """
    if isinstance(x, _UnwrapsToExpr):
        return x.unwrap()
    return cast("Expr", x)


def from_expr(expr: _UnwrapsToExpr | Expr | int | float) -> Expr:
    """Symmetric A-axis entry: ensure ``expr`` is in ``ir.Expr`` form.

    Identical to ``to_expr`` — kept for API symmetry, deliberately
    (Python does not auto-derive the dual, so the surface cost is paid
    with intent: "symmetry is a trap when the language doesn't
    auto-derive the duals").
    """
    return to_expr(expr)


def to_expr_unchecked(x: _UnwrapsToExpr) -> Expr:
    """Assertion-stripped ``to_expr``.

    Assumes ``x`` is a facade; skips the ``isinstance`` dispatch. A
    mis-assumption raises ``AttributeError`` (loud, not UB). Call sites
    must stay deliberate — the skipped dispatch is the audit point.
    """
    return x.unwrap()


# ---- Axis A, type-driven — wrap_expr (total) ---------------------------
# The reverse direction: given an ``ir.Expr`` + its (deduced) ``ir.Type``,
# wrap it in the matching facade. Replaces the 4× ``isinstance(t,
# ir.*Type): return Facade(expr=)`` chain. MRO-sorted registry
# so a pybind-registered subclass (``DistributedTensorType`` ⊂
# ``TensorType``) matches before its base — the silent-downgrade failure
# mode the live chain had; the distributed tier inserts its
# type via ``register_facade`` so ``typing`` never imports ``distributed``
# (no cycle).

# (``ir.Type`` subclass, facade factory). Insertion order is not load-bearing
# — ``register_facade`` inserts subclasses before bases, and ``wrap_expr``
# scans in registry order (subclasses precede bases).
_REGISTRY: list[tuple[type[_ir.Type], Callable[[Expr], _UnwrapsToExpr]]] = []


def register_facade(type_cls: type[_ir.Type], factory: Callable[[Expr], _UnwrapsToExpr]) -> None:
    """Register an ``ir.Type`` subclass → facade factory for ``wrap_expr``.

    The distributed tier calls this at import time to insert
    ``DistributedTensorType`` (before ``TensorType`` — pybind registers it
    as a ``TensorType`` subclass, so the plain branch would swallow it)
    and ``CommCtxType``. Idempotent. Subclasses are inserted
    before their registered bases so the ``isinstance`` scan sees them
    first.
    """
    for i, (t, _) in enumerate(_REGISTRY):
        if t is type_cls:
            _REGISTRY[i] = (type_cls, factory)
            return
    # Insert before the first registered type that type_cls is a strict
    # subclass of, so subclasses precede bases in the scan.
    for i, (t, _) in enumerate(_REGISTRY):
        if type_cls is not t and issubclass(type_cls, t):
            _REGISTRY.insert(i, (type_cls, factory))
            return
    _REGISTRY.append((type_cls, factory))


@cache
def _ensure_base_registry() -> None:
    """Populate the base type→facade set (the live ``_call_facade`` chain).

    Deferred imports break the typing↔facade cycle (facades import from
    ``pypto_core``; ``type_utils`` is imported by them transitively).
    ``@cache`` makes it idempotent (runs once) and does not wipe
    distributed's ``register_facade`` additions — they land directly on
    ``_REGISTRY``.
    """
    from pypto.language.typing.array import Array  # noqa: PLC0415
    from pypto.language.typing.ptr import Ptr  # noqa: PLC0415
    from pypto.language.typing.scalar import Scalar  # noqa: PLC0415
    from pypto.language.typing.tensor import Tensor  # noqa: PLC0415
    from pypto.language.typing.tile import Tile  # noqa: PLC0415

    _REGISTRY.extend(
        [
            (_ir.TensorType, lambda e: Tensor(expr=e)),
            (_ir.TileType, lambda e: Tile(expr=e)),
            (_ir.ScalarType, lambda e: Scalar(expr=e)),
            (_ir.ArrayType, lambda e: Array(expr=e)),
            (_ir.PtrType, lambda e: Ptr(expr=e)),
        ]
    )


def wrap_expr(expr: Expr, t: _ir.Type) -> _UnwrapsToExpr:
    """Wrap an ``ir.Expr`` in the facade matching its (deduced) type
    (A-axis, type-driven).

    Replaces ``marker._call_facade``'s tail + ``tuple._wrap_element`` (the
    parser frontend's copies are its dual). MRO-sorted registry lookup: a
    pybind subclass (``DistributedTensorType`` ⊂ ``TensorType``) matches
    before its base — fixing the live chain's silent downgrade of a
    distributed return to a plain ``Tensor``. ``TupleType`` is
    special-cased (the multi-return lane).
    """
    _ensure_base_registry()
    for type_cls, factory in _REGISTRY:
        if isinstance(t, type_cls):
            return factory(expr)
    if isinstance(t, _ir.TupleType):
        from pypto.language.typing.tuple import Tuple  # noqa: PLC0415

        return Tuple._runtime(expr)
    raise TypeError(
        f"cannot wrap an Expr of type {t} in a facade (expected Tensor/Tile/Scalar/Array/Ptr/Tuple)"
    )


# ---- Axis A′ — annotation→Type (the ``wrap_expr`` dual, D-axis)
# The reverse direction of ``wrap_expr`` at the annotation tier: an
# annotation-mode facade (``pl.Tensor[[s], dt]``) → its ``ir.Type``. Reads
# annotation-mode facade state (``.dtype``/``.shape``/``.memref``/``.layout``),
# not runtime state (``.unwrap()``→``.expr``) — so the A-axis ``to_expr`` does
# not subsume it. The funnel owns the annotation→Type path too; the parser
# frontend's
# ``type_resolver.resolve_param_type``/``resolve_type`` is its dual.
# Deferred facade imports break the cycle
# (``scalar`` imports ``type_utils`` for ``to_expr``).


def _annotation_tensor_view(layout: object) -> _ir.TensorView | None:
    """The third-slot layout of a ``Tensor``/``DistributedTensor`` annotation
    → its ``TensorView`` (the AST's ``resolve_type``: a bare layout becomes
    ``TensorView([], layout)``; a full ``pl.TensorView(...)`` passes through;
    absent layout → ``None``). The facade stores both forms in its ``layout``
    field (``TensorMeta.__getitem__`` routes the third slot there)."""
    if layout is None:
        return None
    if isinstance(layout, _ir.TensorView):
        return layout
    return _ir.TensorView([], cast("_ir.TensorLayout", layout))


def _annotation_tile_view(view: object) -> _ir.TileView | None:
    """Return the explicit ``TileView`` carried by a Tile annotation.

    Tile annotations are also used as producer result descriptors by the
    assembly ``transmute`` intrinsic, so dropping this field would make an
    explicit post-pass view indistinguishable from the implicit view.
    """
    if view is None:
        return None
    if isinstance(view, _ir.TileView):
        return view
    raise TypeError(f"invalid TileView annotation payload: {view!r}")


def _implicit_tile_layout_from_shape(shape: Sequence[_ir.Expr | int]) -> _ir.TileLayout:
    """Infer the implicit block layout represented by omitted TileView syntax.

    Mirrors C++ ``InferImplicitTileLayoutFromShape`` (tile_view_semantics.h):
    a 2-D tile whose column extent is 1 (and rows > 1) implicitly carries
    ``col_major``; everything else is ``row_major``.
    """
    if len(shape) != 2:
        return _ir.TileLayout.row_major

    def const_int(value: object) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, _ir.ConstInt):
            return value.value
        if isinstance(value, _ir.Neg) and isinstance(value.operand, _ir.ConstInt):
            return -value.operand.value
        return None

    rows_value = const_int(shape[0])
    cols_value = const_int(shape[1])
    if rows_value is None or cols_value is None:
        return _ir.TileLayout.row_major
    return _ir.TileLayout.col_major if cols_value == 1 and rows_value > 1 else _ir.TileLayout.row_major


def implicit_tile_view_defaults(
    shape: Sequence[_ir.Expr | int],
    memory_space: _ir.MemorySpace | None = None,
) -> tuple[_ir.TileLayout, _ir.TileLayout, int]:
    """Return the ``(blayout, slayout, fractal)`` implicit defaults for a tile
    of ``shape`` living in ``memory_space``.

    The single Python mirror of C++ ``GetImplicitTileLayout``
    (tile_view_semantics.h) — the space→layout table both frontends share.
    The parser materializes these when TileView syntax omits a field; the
    tracer's annotation funnel (``annotation_to_type``) materializes them
    when a field sits at the bare ``TileView()`` default, which is the
    same convention the printer's elision uses (fields equal to the bare
    default are indistinguishable from omitted).
    """
    default_blayout = _implicit_tile_layout_from_shape(shape)
    default_slayout = _ir.TileLayout.none_box
    default_fractal = _ir.TileView().fractal
    if memory_space in (_ir.MemorySpace.Mat, _ir.MemorySpace.Left):
        default_blayout = _ir.TileLayout.col_major
        default_slayout = _ir.TileLayout.row_major
    elif memory_space == _ir.MemorySpace.Right:
        default_slayout = _ir.TileLayout.col_major
    elif memory_space == _ir.MemorySpace.Acc:
        default_blayout = _ir.TileLayout.col_major
        default_slayout = _ir.TileLayout.row_major
        default_fractal = 1024
    elif memory_space == _ir.MemorySpace.LeftScale:
        # ISA TileLeftScale: RowMajor / RowMajor, MX scale fractal size 32.
        default_blayout = _ir.TileLayout.row_major
        default_slayout = _ir.TileLayout.row_major
        default_fractal = 32
    elif memory_space == _ir.MemorySpace.RightScale:
        # ISA TileRightScale: ColMajor / ColMajor, MX scale fractal size 32.
        default_blayout = _ir.TileLayout.col_major
        default_slayout = _ir.TileLayout.col_major
        default_fractal = 32
    return default_blayout, default_slayout, default_fractal


def _materialized_tile_view(
    view: _ir.TileView | None,
    shape: list[_ir.Expr],
    memory_space: _ir.MemorySpace | None,
) -> _ir.TileView | None:
    """Materialize a Tile annotation's view with the implicit layout defaults.
    The parser's ``resolve_type`` fills any TileView field the syntax omits
    with the ``(shape, memory_space)`` implicit default (the printer elides
    those same fields, so the round trip is stable). An annotation object
    cannot record omission — a ``pl.TileView(valid_shape=...)`` constructor
    leaves unset layout fields at the bare ``TileView()`` defaults — so the
    tracer's funnel treats fields at the bare defaults as omitted and
    materializes the same implicit values. Without this, an annotated
    ``-> pl.Tile[..., pl.TileView(...)]`` carries different layout fields
    than the shared C++ op inference derives for the same tile, and the two
    frontends disagree on the IR dataflow (the inline pass splices the
    callee's inferred value into a binding typed by the annotation).
    """
    if view is None:
        return None
    impl_blayout, impl_slayout, impl_fractal = implicit_tile_view_defaults(shape, memory_space)
    bare = _ir.TileView()
    return _ir.TileView(
        valid_shape=view.valid_shape,
        stride=view.stride,
        start_offset=view.start_offset,
        blayout=impl_blayout if view.blayout == bare.blayout else view.blayout,
        slayout=impl_slayout if view.slayout == bare.slayout else view.slayout,
        fractal=impl_fractal if view.fractal == bare.fractal else view.fractal,
        pad=view.pad,
        compact=view.compact,
    )


def _shape_dim(d: int | _ir.Expr | Scalar) -> _ir.Expr:
    """A shape dimension from an annotation: ints materialize as
    ``ConstInt(INDEX)`` (the parser's dtype discipline); a ``Scalar``
    dim is a ``DynVar`` extent symbol — unwrapped to its shared ``ir.Var``
    (annotation-determined dyn-dims, shared by name; DynVar-shaped kernels
    are eagerly traceable)."""
    if isinstance(d, _ir.Expr):
        return d
    from pypto.language.typing.scalar import Scalar  # noqa: PLC0415

    if isinstance(d, Scalar):
        return d.unwrap()
    return _ir.ConstInt(int(d), DataType.INDEX, _ir.Span.unknown())


def annotation_to_return_types(annotation: object) -> list[_ir.Type]:
    """Convert a return annotation to the frontend's flat return list.

    ``pl.Tuple[...]`` is a structural tuple annotation and therefore remains
    one ``TupleType`` return. Python's builtin ``tuple[T1, T2]`` spelling is
    the flat multi-return syntax used by the parser, so its elements become
    individual function return types.
    """
    try:
        return [annotation_to_type(annotation)]
    except TypeError:
        if get_origin(annotation) is not tuple:
            return []
        elements = get_args(annotation)
        if not elements or elements[-1] is Ellipsis:
            raise TypeError("tuple return annotations must list concrete facade element types") from None
        return [annotation_to_type(element) for element in elements]


def annotation_to_type(facade: object) -> _ir.Type:
    """Convert an annotation-mode facade to its ``ir.Type`` (the params and
    the return type of the tracing frontend, matching the parser frontend's
    ``resolve_param_type``/``resolve_type``: a plain ``pl.Tensor[[s], dt]``
    yields ``TensorType(shape, dtype)`` with no view/memref; a third-slot
    layout ``[s, dt, pl.NZ]`` flows into the ``tensor_view``).

    The ``wrap_expr`` dual (A′-axis). MRO-ordered like the
    ``wrap_expr`` registry: ``DistributedTensor`` (a ``Tensor`` subclass via
    pybind) is checked before ``Tensor`` or the plain branch swallows it —
    the same silent-downgrade failure mode.
    """
    from pypto.language.distributed.typing import (  # noqa: PLC0415
        CommCtx,
        DistributedTensor,
    )
    from pypto.language.typing.array import Array  # noqa: PLC0415
    from pypto.language.typing.ptr import Ptr  # noqa: PLC0415
    from pypto.language.typing.scalar import Scalar  # noqa: PLC0415
    from pypto.language.typing.tensor import Tensor  # noqa: PLC0415
    from pypto.language.typing.tile import Tile  # noqa: PLC0415
    from pypto.language.typing.tuple import Tuple  # noqa: PLC0415

    if isinstance(facade, DistributedTensor):
        assert facade.dtype is not None
        assert facade.shape is not None
        return _ir.DistributedTensorType(
            [_shape_dim(d) for d in facade.shape],
            facade.dtype,
            facade.memref,
            _annotation_tensor_view(facade.layout),
        )
    if isinstance(facade, Tensor):
        assert facade.dtype is not None
        assert facade.shape is not None
        return _ir.TensorType(
            [_shape_dim(d) for d in facade.shape],
            facade.dtype,
            facade.memref,
            _annotation_tensor_view(facade.layout),
        )
    if isinstance(facade, Tile):
        assert facade.dtype is not None
        assert facade.shape is not None
        shape = [_shape_dim(d) for d in facade.shape]
        return _ir.TileType(
            shape,
            facade.dtype,
            facade.memref,
            _materialized_tile_view(_annotation_tile_view(facade.tile_view), shape, facade.memory_space),
            facade.memory_space,
        )
    if isinstance(facade, Scalar):
        assert facade.dtype is not None
        return _ir.ScalarType(facade.dtype)
    if isinstance(facade, Array):
        assert facade.dtype is not None and facade.extent is not None
        return _ir.ArrayType(facade.dtype, _shape_dim(facade.extent))
    if isinstance(facade, Ptr):
        return _ir.PtrType()
    if isinstance(facade, Tuple):
        # Annotation mode: ``pl.Tuple[T1, T2, ...]`` → ``TupleType`` over
        # the element types (the return annotation is the inspection
        # contract). The element annotations are themselves facades.
        return _ir.TupleType([annotation_to_type(t) for t in facade._types])
    if isinstance(facade, CommCtx):
        return _ir.CommCtxType()
    if facade is CommCtx:
        # The parameterless kinds have no subscript form — the bare class is
        # the parser's annotation spelling (``pld.CommCtx``), admitted as
        # itself (see ``is_staged_annotation`` in ``language/protocol.py``).
        return _ir.CommCtxType()
    raise TypeError(
        f"cannot derive an IR type from annotation {facade!r}: expected a "
        f"pl.Tensor/pl.Tile/pl.Scalar/pl.Array/pl.Ptr annotation"
    )


def apply_result_type(expr: _ir.Expr, descriptor: object) -> _ir.Expr:
    """Copy a detached Call/Submit with an explicit stored result Type."""
    result_type = annotation_to_type(descriptor)
    if isinstance(expr, _ir.Call):
        return _ir.Call(expr.op, expr.args, expr.kwargs, expr.attrs, result_type, expr.span)
    if isinstance(expr, _ir.Submit):
        return _ir.Submit(
            expr.op,
            expr.args,
            expr.deps,
            expr.kwargs,
            expr.attrs,
            result_type,
            expr.span,
            core_num=expr.core_num,
            sync_start=expr.sync_start,
            allow_early_resolve=expr.allow_early_resolve,
            predicate=expr.predicate,
        )
    raise TypeError("apply_result_type expects a Call or Submit")


# ---- Axis C — static-fold/query (partial) ------------------------------
# An IR-tier query: "is this ``ConstInt`` (compile-time known) or dynamic?"
# Returns ``int`` (partial — raise / ``None``), NOT an ``Expr`` — a
# different shape from A/B, so it stays a distinct funnel entry, not folded
# into ``to_expr``. The ``_subscript._const_int_value`` helper
# is the richer intra-typing variant (also folds ``Neg(ConstInt)``); the
# two may later unify.


def to_int(x: Expr | int) -> int:
    """Extract the compile-time int value of a ``ConstInt`` or bare ``int``
    (C-axis). Raise ``TypeError`` otherwise — the *strict*
    partial form; use ``as_int`` for the dual-path ``int | None`` form.

    ``bool`` is excluded at runtime (a Python bool is an int subclass, but
    a boolean is not an integer constant — the operand boundary lifts it via ``ConstBool``
    on the B-axis, not here). The ``int`` parameter type subsumes ``bool``
    (a Python typing limitation); the runtime guard keeps the discipline.
    """
    if isinstance(x, _ir.ConstInt):
        return x.value
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    raise TypeError(f"expected a ConstInt or int, got {type(x).__name__}")


def as_int(x: Expr | int) -> int | None:
    """The dual-path C-axis query: ``to_int`` or ``None``.

    For callers that branch on "statically known?" without raising (the
    one justified exception to "strict out").
    """
    if isinstance(x, _ir.ConstInt):
        return x.value
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    return None


def to_int_unchecked(x: _ir.ConstInt) -> int:
    """Assertion-stripped ``to_int``. Assumes ``x`` is a
    ``ConstInt``; mis-assumption raises ``AttributeError``."""
    return x.value


# ---- Axis B — host→typed-constant (total) ------------------------------
# ``from_int`` = ``pl.const`` (the already-sanctioned explicit-dtype
# constructor). The canonical-dtype operand-boundary variant is
# ``coerce_host_scalar`` (``_host.py``); the shared ``isinstance`` ladder
# may later be factored out of the two.


def from_int(value: int | float, dtype: DataType) -> Scalar:
    """The B-axis host→typed-constant path = ``pl.const``.

    Lifts a host ``int``/``float``/``bool`` to a typed ``Const`` with an
    explicit dtype — the sanctioned explicit-dtype constructor.
    The operand-boundary variant (canonical dtypes, no explicit dtype) is
    ``coerce_host_scalar`` (``_host.py``).
    """
    from pypto.language.typing.const import const  # noqa: PLC0415

    return const(value, dtype)


# ---- Syntax-level type matching (shared by both frontends) ------------
# The rebind/annotation type-comparison seam: the AST parser and the tracing
# Sema must accept and reject the *same* kernels, so the comparison lives in
# the shared typing layer, never in one frontend (the sibling frontends do
# not import each other; see test_collector_contract's layering guard).


def _const_int_value(value: object) -> int | None:
    """Extract integer value from a compile-time constant, or None."""
    if isinstance(value, int):
        return value
    if isinstance(value, _ir.ConstInt):
        return value.value
    if isinstance(value, _ir.Neg) and isinstance(value.operand, _ir.ConstInt):
        return -value.operand.value
    return None


def _infer_implicit_tile_layout_from_shape(shape: Sequence[_ir.Expr | int]) -> _ir.TileLayout:
    """Infer the block layout represented by omitted TileView syntax."""
    if len(shape) != 2:
        return _ir.TileLayout.row_major
    rows_value = _const_int_value(shape[0])
    cols_value = _const_int_value(shape[1])
    if rows_value is None or cols_value is None:
        return _ir.TileLayout.row_major
    return _ir.TileLayout.col_major if cols_value == 1 and rows_value > 1 else _ir.TileLayout.row_major


def _implicit_tile_view_defaults(
    shape: Sequence[_ir.Expr | int],
    memory_space: _ir.MemorySpace | None = None,
) -> tuple[_ir.TileLayout, _ir.TileLayout, int]:
    """Return (blayout, slayout, fractal) implicit defaults for a given shape + memory space."""
    default_blayout = _infer_implicit_tile_layout_from_shape(shape)
    default_slayout = _ir.TileLayout.none_box
    default_fractal = _ir.TileView().fractal
    if memory_space in (_ir.MemorySpace.Mat, _ir.MemorySpace.Left):
        default_blayout = _ir.TileLayout.col_major
        default_slayout = _ir.TileLayout.row_major
    elif memory_space == _ir.MemorySpace.Right:
        default_slayout = _ir.TileLayout.col_major
    elif memory_space == _ir.MemorySpace.Acc:
        default_blayout = _ir.TileLayout.col_major
        default_slayout = _ir.TileLayout.row_major
        default_fractal = 1024
    return default_blayout, default_slayout, default_fractal


def _shape_exprs_match(lhs: Sequence[_ir.Expr], rhs: Sequence[_ir.Expr]) -> bool:
    """Return whether two shape-like expression lists are statically identical."""
    if len(lhs) != len(rhs):
        return False
    for lhs_dim, rhs_dim in zip(lhs, rhs):
        if lhs_dim is rhs_dim:
            continue
        lhs_value = _const_int_value(lhs_dim)
        rhs_value = _const_int_value(rhs_dim)
        if lhs_value is None or rhs_value is None or lhs_value != rhs_value:
            return False
    return True


def _has_printable_tile_view(
    tile_view: _ir.TileView | None,
    shape: Sequence[_ir.Expr],
    memory_space: _ir.MemorySpace | None = None,
) -> bool:
    """Return whether a TileView carries non-default metadata in Python text form."""
    if tile_view is None:
        return False
    if tile_view.valid_shape and not _shape_exprs_match(tile_view.valid_shape, shape):
        return True
    if tile_view.stride:
        return True
    if tile_view.start_offset is not None:
        return True
    default_blayout, default_slayout, default_fractal = _implicit_tile_view_defaults(shape, memory_space)
    if tile_view.blayout != default_blayout:
        return True
    if tile_view.slayout != default_slayout:
        return True
    if tile_view.fractal != default_fractal:
        return True
    if tile_view.pad != _ir.PadValue.null:
        return True
    return False


def _normalize_type_for_syntax_match(type_: _ir.Type | None) -> _ir.Type | None:
    """Normalize omitted default TileView metadata before syntax-level comparison."""
    if isinstance(type_, _ir.TileType):
        tile_view = type_.tile_view
        if tile_view is not None and not _has_printable_tile_view(tile_view, type_.shape, type_.memory_space):
            tile_view = None
        if tile_view is type_.tile_view:
            return type_
        return _ir.TileType(type_.shape, type_.dtype, type_.memref, tile_view, type_.memory_space)

    return type_


def _shape_has_symbolic_dim(type_: _ir.Type) -> bool:
    """Return True when *type_* is a shaped type with at least one non-ConstInt dim."""
    if not isinstance(type_, (_ir.TensorType, _ir.TileType)):
        return False
    return any(not isinstance(d, _ir.ConstInt) for d in type_.shape)


def _simplify_shape_dims(type_: _ir.Type, analyzer: _arith.Analyzer) -> _ir.Type:
    """Rebuild a TensorType/TileType with each outer shape dim passed through
    the arithmetic simplifier.

    Needed so that symbolic dims that reduce to the same value — e.g.
    ``(k + C) - k`` and ``C`` — compare equal under structural ``==``.

    Scope: only the outer ``shape`` is simplified. TileView fields
    (``valid_shape``, ``stride``, ``start_offset``) and TensorView
    fields retain their original expressions and still compare structurally.
    """
    if not _shape_has_symbolic_dim(type_):
        return type_
    assert isinstance(type_, (_ir.TensorType, _ir.TileType))
    simplified_shape = [analyzer.simplify(d) for d in type_.shape]
    if isinstance(type_, _ir.TensorType):
        return _ir.TensorType(simplified_shape, type_.dtype, type_.memref, type_.tensor_view)
    return _ir.TileType(simplified_shape, type_.dtype, type_.memref, type_.tile_view, type_.memory_space)


def _types_match(lhs: _ir.Type | None, rhs: _ir.Type | None) -> bool:
    """Return whether two IR types are structurally equal after TileView normalization.

    Uses C++ ``structural_equal`` (via ``==``) instead of comparing printed
    strings.  Note:  ``structural_equal`` intentionally skips the ``memref``
    field on TileType/TensorType — memref identity is not relevant for
    frontend-level type matching because the Var (not the Call) carries the
    authoritative memref via ``override_type``.

    Outer shape dimensions are simplified via a shared arithmetic analyzer
    before comparison so expressions like ``(k + C) - k`` match ``C``.  The
    analyzer is only constructed when at least one shape actually contains a
    symbolic dim, keeping the all-static fast path free of arith overhead.
    """
    if lhs is rhs:
        return True
    if lhs is None or rhs is None:
        return lhs is rhs
    lhs = _normalize_type_for_syntax_match(lhs)
    rhs = _normalize_type_for_syntax_match(rhs)
    assert lhs is not None
    assert rhs is not None
    if type(lhs) is not type(rhs):
        return False
    if _shape_has_symbolic_dim(lhs) or _shape_has_symbolic_dim(rhs):
        analyzer = _arith.Analyzer()
        lhs = _simplify_shape_dims(lhs, analyzer)
        rhs = _simplify_shape_dims(rhs, analyzer)
    return lhs == rhs


def _tile_types_differ_only_by_unset_memory_space(lhs: _ir.Type | None, rhs: _ir.Type | None) -> bool:
    """Return whether two TileTypes are equal once an unset space is a wildcard.

    ``memory_space=None`` is the IR's "the compiler will place this", not a
    distinct type. Ops that leave it unset (``pl.load``, ``pl.tile.create``) and
    ops whose space is fixed by hardware (every vector op, via
    ``set_output_memory(Vec)``) therefore mix freely in ordinary user code::

        t = pl.load(a, [0, 0], [16, 64])   # unset
        t = pl.add(t, t)                   # Vec

    Treating that as a type change would reject the most common shape of DSL
    there is. All other fields still compare exactly.

    Kept separate from ``_types_match``: annotation-vs-inferred comparisons must
    stay strict, so that a *stated* space which contradicts the producer is
    still an error.
    """
    if not isinstance(lhs, _ir.TileType) or not isinstance(rhs, _ir.TileType):
        return False
    if (lhs.memory_space is None) == (rhs.memory_space is None):
        return False  # both set or both unset -> plain _types_match already decided
    unified = _ir.TileType(rhs.shape, rhs.dtype, rhs.memref, rhs.tile_view, lhs.memory_space)
    return _types_match(lhs, unified)
