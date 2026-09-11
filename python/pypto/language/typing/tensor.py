# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor wrapper type for PyPTO Language DSL."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast, overload

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import Expr, MemRef, TensorLayout, TensorType, Var

if TYPE_CHECKING:
    from pypto.language.typing.dynamic import DynVar
    from pypto.language.typing.scalar import Scalar

TensorT = TypeVar("TensorT", bound="Tensor")

# A subscript key: one axis (index or slice) or a tuple of them. Quoted —
# Scalar is a TYPE_CHECKING-only import to avoid the facade cycle.
_SubscriptKey: TypeAlias = (
    "int | Scalar | Expr | slice[Any, Any, Any] | tuple[int | Scalar | Expr | slice[Any, Any, Any], ...]"
)


def _validate_tensor_meta_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Validate TensorMeta.__call__ argument structure."""
    allowed_kwargs = {"shape", "dtype", "expr", "layout", "memref", "_annotation_only"}
    unexpected = set(kwargs) - allowed_kwargs
    if unexpected:
        name = sorted(unexpected)[0]
        raise TypeError(f"Tensor() got an unexpected keyword argument '{name}'")

    if len(args) > 6:
        raise TypeError(f"Tensor() takes at most 6 positional arguments but {len(args)} were given")

    param_names = ("shape", "dtype", "expr", "layout", "memref", "_annotation_only")
    for _index, name in enumerate(param_names[: len(args)]):
        if name in kwargs:
            raise TypeError(f"Tensor() got multiple values for argument '{name}'")


class TensorMeta(type):
    """Metaclass for Tensor to enable subscript notation."""

    def __getitem__(cls, item: tuple[Any, ...]) -> "Tensor":
        """Enable Tensor[[shape], dtype], Tensor[[shape], dtype, layout_or_memref],
        and Tensor[[shape], dtype, layout, memref] notation.

        Args:
            item: Tuple of 2, 3, or 4 elements.

        Returns:
            Tensor instance with shape, dtype, optional layout/memref.
        """
        # The implicit ``cls`` is typed as the metaclass object, which the
        # ``__call__`` overloads (``cls: type[TensorT]``) cannot bind — rebind
        # to the Tensor-family class it actually is.
        tensor_cls = cast("type[Tensor]", cls)
        if len(item) not in (2, 3, 4):
            raise TypeError(
                "Tensor requires [shape, dtype], [shape, dtype, layout_or_memref], "
                "or [shape, dtype, layout, memref] notation"
            )

        if len(item) == 4:
            shape, dtype, layout, memref = item
            return tensor_cls(
                shape,
                dtype,
                layout=layout,
                memref=memref,
                _annotation_only=True,
            )
        if len(item) == 3:
            shape, dtype, third = item
            if isinstance(third, MemRef):
                return tensor_cls(shape, dtype, memref=third, _annotation_only=True)
            return tensor_cls(shape, dtype, layout=third, _annotation_only=True)
        shape, dtype = item
        return tensor_cls(shape, dtype, _annotation_only=True)

    @overload
    def __call__(
        cls: type[TensorT],
        shape: Sequence[int] | None = None,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        layout: "TensorLayout | None" = None,
        memref: "MemRef | None" = None,
        _annotation_only: bool = False,
    ) -> TensorT: ...

    @overload
    def __call__(
        cls: type[TensorT],
        shape: tuple[Any, Any],
        dtype: None = None,
        expr: None = None,
        layout: "TensorLayout | None" = None,
        memref: "MemRef | None" = None,
        _annotation_only: bool = False,
    ) -> TensorT: ...

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Enable both Tensor((shape), dtype) syntax and runtime wrapping.

        Args:
            shape: Shape tuple or list (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            layout: Optional tensor layout (ND, DN, NZ)
            memref: Optional memory reference
            _annotation_only: Internal flag for annotation-only mode

        Returns:
            Tensor instance
        """
        _validate_tensor_meta_call(args, kwargs)

        # Support metaclass instantiation for annotations
        shape: Sequence[int] | tuple[Any, Any] | None = kwargs.get(
            "shape", args[0] if len(args) > 0 else None
        )
        dtype: DataType | None = kwargs.get("dtype", args[1] if len(args) > 1 else None)
        expr: Expr | None = kwargs.get("expr", args[2] if len(args) > 2 else None)
        layout: TensorLayout | None = kwargs.get("layout", args[3] if len(args) > 3 else None)
        memref: MemRef | None = kwargs.get("memref", args[4] if len(args) > 4 else None)
        annotation_only: bool = kwargs.get("_annotation_only", args[5] if len(args) > 5 else False)

        if (
            isinstance(shape, tuple)
            and len(shape) == 2
            and not isinstance(shape[0], int)
            and dtype is None
            and expr is None
        ):
            real_shape, real_dtype = shape
            return type.__call__(cls, real_shape, real_dtype, None, layout, memref, annotation_only)

        if dtype is not None and expr is None and not annotation_only:
            annotation_only = True

        return type.__call__(cls, shape, dtype, expr, layout, memref, annotation_only)


class Tensor(metaclass=TensorMeta):
    """Tensor type for PyPTO Language DSL.

    This class serves dual purposes:
    1. Type annotation helper for function signatures
    2. Runtime wrapper around IR Expr/Call objects

    Annotation mode (used in type hints):
        x: pl.Tensor[[64, 128], pl.FP16]
        y: pl.Tensor[[64, 128], pl.FP16, pl.NZ]

    Runtime mode (wraps IR expressions):
        tensor = pl.create_tensor([64, 128], dtype=pl.FP32)
        # Returns Tensor wrapping the Call expression

    Examples:
        >>> import pypto.language as pl
        >>>
        >>> @pl.function
        ... def my_func(x: pl.Tensor[[64, 128], pl.FP16, pl.NZ]) -> pl.Tensor[[64, 128], pl.FP32]:
        ...     result: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
        ...     return result
    """

    def __init__(
        self,
        shape: Sequence[int] | None = None,
        dtype: DataType | None = None,
        expr: Expr | None = None,
        layout: TensorLayout | None = None,
        memref: MemRef | None = None,
        _annotation_only: bool = False,
    ):
        """Initialize Tensor.

        Args:
            shape: Shape (for annotation mode)
            dtype: Data type (for annotation mode)
            expr: IR expression to wrap (for runtime mode)
            layout: Optional tensor layout (ND, DN, NZ)
            memref: Optional memory reference
            _annotation_only: Whether this is annotation-only mode
        """
        if _annotation_only:
            self.shape = shape
            self.dtype = dtype
            self.layout = layout
            self.memref = memref
            self._expr = None
        elif expr is not None:
            self._expr = expr
            self.shape = None
            self.dtype = None
            self.layout = None
            self.memref = None
        else:
            raise ValueError(
                "Tensor must be initialized with either (shape, dtype) for "
                "annotations or expr for runtime wrapping"
            )

    def unwrap(self) -> Expr:
        """Get underlying IR expression.

        Returns:
            The wrapped Expr/Call object

        Raises:
            ValueError: If called on an annotation-only Tensor
        """
        if self._expr is None:
            raise ValueError("Cannot unwrap annotation-only Tensor (used in type hints)")
        return self._expr

    @classmethod
    def __class_getitem__(cls, item: tuple[Sequence[int], DataType]) -> "Tensor":
        """The static-checker hook for ``Tensor[[shape], dtype]`` syntax.

        Runtime never calls this — the ``TensorMeta.__getitem__`` subscript
        protocol takes precedence — but pyright resolves class subscripts
        through this signature.
        """
        raise NotImplementedError

    def bind_dynamic(self, dim: int, var: "DynVar") -> None:
        """Declare that dimension ``dim`` is runtime-dynamic — loud under
        the tracing frontend.

        The old semantics lived in the ``@pl.jit`` specializer's AST scan:
        the call was a runtime no-op and the *scan* promoted the dimension
        to a DynVar. Tracing executes the call, so the check runs here:

        * the dimension is already dynamic (the annotation-embedded
          ``pl.dynamic()`` form — ``pl.Tensor[[M, 128], pl.FP32]``) → a
          benign no-op (redundant with the annotation);
        * the dimension is static → ``NotImplementedError``: the override
          form (static shape + ``bind_dynamic``) has no tracing semantics.
          Declare the dimension dynamic in the annotation instead. The
          quarantined AST runtime (``pl.jit_specializer``) still honors
          the old form.

        Args:
            dim: Zero-based dimension index to mark as dynamic.
            var: The DynVar object (created with ``pl.dynamic()``) the
                dimension is bound to.
        """
        if self._expr is not None:
            # Runtime mode (inside a trace): the wrapped expr's type is
            # the minted param type — its shape entry is the extent.
            ty = self._expr.type
            if not isinstance(ty, TensorType):
                raise TypeError(f"bind_dynamic on a value of non-tensor type {ty} — tag a tensor dimension")
            extent: object = ty.shape[dim]
        elif self.shape is not None:
            # Annotation mode: the shape entry is the DynVar (or a static int).
            extent = self.shape[dim]
        else:
            raise TypeError("bind_dynamic on a facade with neither a shape nor a wrapped expr")
        # Dynamic ⇔ an ir.Var (runtime-mode / minted extent) or a DynVar
        # (annotation-mode shape entry); anything else (int, ConstInt) is static.
        if isinstance(extent, Var) or hasattr(extent, "_ir_var"):
            return None  # already dynamic — the benign redundancy
        raise NotImplementedError(
            f"bind_dynamic: dimension {dim} of this tensor is static ({extent!r}); the "
            "override form (static shape + bind_dynamic) has no tracing semantics — "
            "declare the dimension dynamic in the annotation instead "
            f"(pl.Tensor[[{getattr(var, 'name', 'M')}, ...], dtype] with "
            "M = pl.dynamic(...)). The quarantined pl.jit_specializer still "
            "honors the old form."
        )

    @overload
    def __getitem__(self, indices: "slice[Any, Any, Any] | tuple[slice[Any, Any, Any], ...]") -> "Tensor": ...

    @overload
    def __getitem__(self, indices: _SubscriptKey) -> "Tensor | Scalar": ...

    def __getitem__(self, indices: _SubscriptKey) -> "Tensor | Scalar":
        """Numpy-style subscript read: all-scalar full-rank
        → a scalar ``tensor.read``; partial / sliced / rank-reducing → a
        ``tensor.slice`` view carrying ``drop_dims`` (the AST's
        ``_parse_tensor_subscript`` surface).

        The overloads mirror that runtime split where it is statically
        knowable: a key made only of slices always takes the slice path, so
        it is typed ``Tensor``. Any other key is rank-dependent — ``t[i, j]``
        is a scalar only at full rank, ``t[i]`` is a rank-reducing view
        above it — and the operand rank lives in the IR type, so the honest
        static result is the union. Narrow with ``isinstance`` when the
        context needs one flavor.
        """
        from pypto.language.typing._subscript import subscript_read  # noqa: PLC0415 - local: avoid the cycle

        return subscript_read(self, indices, tile=False)

    def __setitem__(
        self,
        key: _SubscriptKey,
        value: "Tensor",
    ) -> None:
        """``t[i:j] = src`` → ``tensor.assemble(t, src, offsets)`` (functional rebind).

        The traced frontend's subscript-write sugar for tensor targets (the
        AST's ``_parse_subscript_assignment`` surface): the source rewriter
        leaves the statement as a plain Python item assignment on the
        facade, so this hook fires during the trace, asks the
        active Frame (``active_frame()``) to resolve the
        target ``ir.Var``, and routes the resulting source-name bind through
        the trace interpreter. The written indices follow the numpy rule
        (offsets always full-rank: a bare ``t[:]`` expands to every axis,
        dims past the written indices are implicit full slices); scalar-index
        rank-reducing writes are loud — the AST's full rank-lifting surface
        is not ported.
        """
        from pypto.language.ambient import active_frame  # noqa: PLC0415
        from pypto.language.op import tensor as _tensor_dsl  # noqa: PLC0415
        from pypto.language.typing._subscript import subscript_write_offsets  # noqa: PLC0415

        frame = active_frame()
        if frame is None:
            raise TypeError(
                "Tensor item assignment is only supported inside a traced kernel (the tracing frontend)"
            )
        offsets = subscript_write_offsets(self.unwrap(), key, kind="Tensor")
        new = _tensor_dsl.assemble(self, value, offsets)
        owner, name = frame.resolve_staged_local(self.unwrap())
        if owner is None:
            raise TypeError(
                "Tensor item-assignment target must be a bound local name, "
                f"got an expression-backed Tensor ({self!r})"
            )
        owner.rt_bind(name, new, via=frame)

    def __repr__(self) -> str:
        """String representation."""
        if self._expr is not None:
            return f"Tensor(expr={self._expr})"
        if self.layout is not None:
            return f"Tensor[[{self.shape}], {self.dtype}, {self.layout}]"
        return f"Tensor[[{self.shape}], {self.dtype}]"


class NoDepMarker(Tensor):
    """The runtime wrapper for ``pl.no_dep(t)`` (a parser-recognized marker).

    Wraps the same IR expr as ``t`` so it is indistinguishable from ``t``
    everywhere except the traced sub-kernel call path: ``Marker.__call__``
    recognizes the marker and threads the argument slot into the Call's
    ``arg_direction_overrides`` attrs — mirroring the AST parser's
    ``_strip_call_arg_markers`` (the marker's per-arg override is
    parser-level syntax; the tracing frontend must route it at trace time).
    Outside a call argument list it behaves as the wrapped tensor (the
    parser contract's no-op), because it IS a ``Tensor``.

    Constructed as ``NoDepMarker(expr=t.unwrap())`` — the ``TensorMeta``
    funnel routes the keyword ``expr=`` form to expr wrapping, so no
    custom ``__init__`` is needed.
    """


class DumpTagMarker(Tensor):
    """The runtime wrapper for ``pl.dump_tag(t)`` (a parser-recognized marker).

    Wraps the same IR expr as ``t``. The traced ``bare_stmt`` recognizes the
    marker, records the tagged Var in the current function's forward-sticky
    dump set (mirroring the AST parser's ``_handle_dump_tag``), and emits no
    IR; every *subsequent* kernel call that consumes that exact Var gets it
    merged into the Call's ``dump_vars`` attr (``Marker.__call__``). Outside
    a statement position it behaves as the wrapped tensor, because it IS a
    ``Tensor``. Constructed as ``DumpTagMarker(expr=t.unwrap())`` (see
    ``NoDepMarker``).
    """


class CachePolicyDecl:
    """The runtime marker for ``pl.set_cache_policy(t, policy)``.

    A plain host object, never a value: ``pl.set_cache_policy`` returns it
    only inside a trace, and the traced ``bare_stmt`` recognizes it and
    routes the declaration to the enclosing deferred scope — the scope's
    attrs are fixed at ``begin_scope``, so the declaration must join them
    before the scope's first statement opens it (mirroring the AST parser,
    which hoists the markers before building the scope). Carries the
    declared ``Var`` (identity-tracked) and the ``CachePolicy`` as ``int``
    (the form the scope attr and the ``tile.load`` ``cache`` kwarg hold).
    """

    __slots__ = ("var", "policy")

    def __init__(self, var: Var, policy: int) -> None:
        self.var = var
        self.policy = policy


__all__ = ["Tensor", "NoDepMarker", "DumpTagMarker", "CachePolicyDecl"]
