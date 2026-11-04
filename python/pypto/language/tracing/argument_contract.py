# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The argument-side half of the tracing type contract.

The tracing frontend derives each callee parameter from the *call site* — that
is what makes a call specialize. The declaration is the ceiling on that
derivation: ``pl.Scalar[pl.INT32]`` states the parameter is an INT32 value, so
a call site holding an INDEX value must reach the callee *as* INT32 rather than
retyping the callee. Without this guard the derivation wins silently: the tracer
emits whatever the submitted value carried and the two frontends disagree about
the device ABI with no diagnostic (a per-rank call passing an INDEX loop index
into a ``pl.Scalar[pl.INT32]`` slot).

Three guards live here:

- :func:`coerce_declared_arguments` — the call/submit boundary. An argument
  inside the declared slot's integer lattice (one end ``pl.INDEX``, the other a
  fixed-width integer) is converted to the declared dtype and bound to a
  temporary, so the call carries the declared type and the printed program is
  the hand-written spelling. Any other mismatch raises, naming the parameter and
  the one-word fix (``pl.cast``). The lattice itself lives in
  :mod:`pypto.language.scalar_conversion`, which both frontends consult.
- :func:`bind_declared_keywords` — the same boundary, on the *name* side. A call
  site may name any declared parameter; the binding folds the keywords into the
  declaration order so every lane below reads one positional list.
- :func:`reject_bare_data_type_annotations` — a bare marker (``vlen:
  pl.INDEX``) states a dtype but no *world*, so the parameter silently became
  a host knob and the artifact specialized per value. Name ``pl.Scalar[...]``.
- the world checks stay in the jit decorator, because a host knob is a
  declaration-time fact rather than a type fact.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any, cast

from pypto.language.ambient import active_frame
from pypto.language.protocol import is_staged
from pypto.language.scalar_conversion import (
    ArgumentDisposition,
    classify_scalar_conversion,
)
from pypto.language.typing import annotation_to_type
from pypto.language.typing._ir_facade import facade_from_ir_type
from pypto.pypto_core import DataType, ir

__all__ = [
    "CALL_SITE_KEYWORDS",
    "bind_declared_keywords",
    "check_declared_argument",
    "check_declared_arguments",
    "coerce_declared_arguments",
    "reject_bare_data_type_annotations",
]

# Keyword arguments a jit call accepts beside the declared parameters. Both ride
# the detached ``ir.Call`` as call-site metadata (``attrs=`` always, ``device=``
# on an Orchestrator callee) and are never bound to a parameter, which is why the
# sub-kernel surface spells the same two names. A declared parameter of the same
# name loses to the metadata — the printer's call-kwarg surface owns them.
CALL_SITE_KEYWORDS: frozenset[str] = frozenset({"attrs", "device"})

# Kind order matters twice over: ``isinstance`` must see the pybind subclass
# first (``DistributedTensorType`` IS-A ``TensorType``), and two arguments are
# the same kind only when they hit the same entry.
_KIND_ORDER: tuple[type, ...] = (
    ir.DistributedTensorType,
    ir.TensorType,
    ir.TileType,
    ir.ScalarType,
    ir.ArrayType,
    ir.PtrType,
)


def _kind(ty: ir.Type) -> type | None:
    return next((kind for kind in _KIND_ORDER if isinstance(ty, kind)), None)


def _is_dynamic_dim(dim: object) -> bool:
    """A declared extent the call site specializes: a dynamic symbol
    (``pl.dynamic``) is an ``ir.Var``, and the facade-level ``DynVar`` is
    recognized by the same marker the eager reuse predicate reads."""
    return isinstance(dim, ir.Var) or hasattr(dim, "_ir_var")


def _describe(ty: ir.Type) -> str:
    """Render an IR type in the spelling the author would write for it.

    ``ir.Type.__str__`` is the printer's pl spelling (``pl.Tensor[[64, 64],
    pl.FP32]``), so a diagnostic cannot drift from printed IR.
    """
    return str(ty)


def _declared_type(hint: object) -> ir.Type | None:
    """The type a parameter declaration states, or ``None`` when it states no
    concrete type.

    Two spellings defer to the call site by construction and must keep doing
    so: a bare facade class (``pl.Tensor`` — the lazy entry's deferred-shape
    form) and a host-knob annotation (``float``, a ``pl.constexpr`` marker),
    which is not a runtime value at all.
    """
    if hint is None or isinstance(hint, type) or not is_staged(hint):
        return None
    return annotation_to_type(hint)


def _extent_int(dim: Any) -> int | None:
    """The literal value of an extent, when it has one (``int`` or a
    constant ``ir.Expr``); ``None`` for a symbolic extent."""
    if isinstance(dim, int):
        return dim
    return getattr(dim, "value", None)


def _same_extent(declared_dim: Any, actual_dim: Any) -> bool:
    declared_value = _extent_int(declared_dim)
    actual_value = _extent_int(actual_dim)
    if declared_value is not None and actual_value is not None:
        return declared_value == actual_value
    return ir.structural_equal(declared_dim, actual_dim, enable_auto_mapping=True)


def _shape_mismatch(declared: ir.Type, actual: ir.Type) -> str | None:
    """The first declared static extent the argument contradicts."""
    declared_shape = list(getattr(declared, "shape", []) or [])
    actual_shape = list(getattr(actual, "shape", []) or [])
    if len(declared_shape) != len(actual_shape):
        return f"rank {len(actual_shape)} against a declared rank {len(declared_shape)}"
    for index, (declared_dim, actual_dim) in enumerate(zip(declared_shape, actual_shape)):
        if _is_dynamic_dim(declared_dim):
            continue
        if not _same_extent(declared_dim, actual_dim):
            return f"dim {index} = {actual_dim} against the declared static extent {declared_dim}"
    return None


def _effective_view_strides(owner: ir.Type, view: Any) -> list[Any] | None:
    """The strides a view states, with a blank stride resolved to the packed
    canonical form it abbreviates (RFC #1300).

    ``tensor.view`` derives an explicit stride with the same
    ``BuildLogicalStridesFromLayout`` helper this consults, so a declaration
    written as a bare layout tag and the view the op produced state one stride
    vector rather than two spellings of it. ``None`` when the stride is blank
    and the owning shape has a symbolic extent: the canonical packing is not
    derivable there, and the comparison must fall back to the stated spelling.
    """
    stated = list(view.stride or [])
    if stated:
        return stated
    shape = list(getattr(owner, "shape", []) or [])
    if not shape:
        return stated
    static_extents: list[int] = []
    for dim in shape:
        extent = _extent_int(dim)
        if extent is None:
            return None
        static_extents.append(extent)
    span = ir.Span("argument-contract", 0, 0)
    exprs = [ir.ConstInt(extent, DataType.INT64, span) for extent in static_extents]
    try:
        return list(ir.tensor_view_semantics.build_logical_strides_from_layout(exprs, view.layout))
    except ValueError:
        # A layout with no packed form at this rank (DN below rank 2): the
        # blank stride states nothing derivable, so compare spellings.
        return None


def _same_view(declared: Any, actual: Any, declared_owner: ir.Type, actual_owner: ir.Type) -> bool:
    """Whether two ``TensorView``s state the same layout contract.

    ``ir.structural_equal`` accepts IR nodes and types, not views, so the
    fields are compared explicitly: layout and pad exactly, stride and
    valid-shape extent-wise (the same literal-or-symbol rule as the shape).
    The stride comparison reads each side's *effective* strides
    (:func:`_effective_view_strides`), so a blank stride — the RFC #1300
    abbreviation of the packed canonical form — and the explicit canonical
    strides ``tensor.view`` derives state one view rather than two. A stride
    outside the canonical packing still refuses, and ``valid_shape`` keeps
    its literal comparison: only the stride field has a canonical form.
    """
    if declared.layout != actual.layout or declared.pad != actual.pad:
        return False
    declared_strides = _effective_view_strides(declared_owner, declared)
    actual_strides = _effective_view_strides(actual_owner, actual)
    if declared_strides is None or actual_strides is None:
        # No canonical derivation on one side: compare the *stated* spellings
        # on both, never a derivation against a blank — the pre-canonical
        # behaviour, so two blanks still agree whatever their shapes hold.
        declared_strides = list(declared.stride or [])
        actual_strides = list(actual.stride or [])
    if len(declared_strides) != len(actual_strides):
        return False
    if any(not _same_extent(d, a) for d, a in zip(declared_strides, actual_strides)):
        return False
    declared_valid = list(declared.valid_shape or [])
    actual_valid = list(actual.valid_shape or [])
    if len(declared_valid) != len(actual_valid):
        return False
    if any(not _same_extent(d, a) for d, a in zip(declared_valid, actual_valid)):
        return False
    return True


def _mismatch(declared: ir.Type, actual: ir.Type) -> str | None:
    """Why ``actual`` falls outside ``declared``, or ``None`` when it is in.

    Inside means: same kind, same dtype, same rank, every declared *static*
    extent equal, and the same tile memory space / declared layout. A declared
    dynamic extent accepts any concrete extent — specializing to the call
    site is exactly what the lane is for; changing the *type* is not.
    """
    if _kind(declared) is not _kind(actual):
        return "a different type kind"
    declared_dtype = getattr(declared, "dtype", None)
    actual_dtype = getattr(actual, "dtype", None)
    if declared_dtype is not None and actual_dtype is not None and declared_dtype != actual_dtype:
        return f"dtype {actual_dtype} against the declared {declared_dtype}"
    shape_problem = _shape_mismatch(declared, actual)
    if shape_problem is not None:
        return shape_problem
    if isinstance(declared, ir.TileType) and isinstance(actual, ir.TileType):
        if declared.memory_space != actual.memory_space:
            return f"memory space {actual.memory_space} against the declared {declared.memory_space}"
    declared_view = getattr(declared, "tensor_view", None)
    actual_view = getattr(actual, "tensor_view", None)
    if declared_view is not None and (
        actual_view is None or not _same_view(declared_view, actual_view, declared, actual)
    ):
        return "a different tensor layout"
    return None


def _actual_type(arg: object) -> ir.Type:
    """The IR type a call-site argument carries.

    A runtime facade wraps the staged value, so its type is the value's type. An
    *annotation-mode* facade (the ``specialize``/``lower`` sample spelling) wraps
    no expression — that is exactly what ``unwrap`` reports — and states its type
    directly.
    """
    try:
        unwrapped = arg.unwrap()  # type: ignore[attr-defined]  # is_staged guarantees the facade protocol
    except (RuntimeError, ValueError):
        # ``unwrap``'s annotation-only signal. The facades disagree on the
        # exception type (``Scalar`` raises ``RuntimeError``, ``Tensor`` a
        # ``ValueError``), so both spellings are accepted here.
        return annotation_to_type(arg)
    # ``unwrap`` is declared on the facade classes, not on the ``object`` this
    # helper receives, so its result reaches here untyped. A staged facade wraps
    # an IR expression, which is what makes the expression's type the answer.
    expr = cast("ir.Expr", unwrapped)
    return expr.type


def check_declared_argument(
    name: str,
    arg: object,
    hint: object,
    fn_name: str,
) -> None:
    """Check one staged argument against its parameter declaration.

    Host values are out of scope here (``_check_param_worlds`` owns the
    world question), and a declaration that states no concrete type admits
    any argument of its kind.
    """
    declared = _declared_type(hint)
    if declared is None or not is_staged(arg):
        return
    actual = _actual_type(arg)
    problem = _refusal(declared, actual)
    if problem is not None:
        raise TypeError(_refusal_message(name, declared, actual, problem, fn_name))


def _refusal(declared: ir.Type, actual: ir.Type) -> str | None:
    """Why ``actual`` stays refused against ``declared``, or None when it is in.

    An admitted scalar conversion is inside the declaration by construction:
    :func:`coerce_declared_arguments` emits the conversion, so it is not a
    problem to report. Everything else still answers to ``_mismatch``.
    """
    if classify_scalar_conversion(declared, actual).disposition is ArgumentDisposition.CONVERT:
        return None
    return _mismatch(declared, actual)


def _refusal_message(
    name: str,
    declared: ir.Type,
    actual: ir.Type,
    problem: str,
    fn_name: str,
) -> str:
    """The one refusal wording both layers can quote."""
    return (
        f"call to {fn_name!r}: parameter {name!r} is declared {_describe(declared)} "
        f"but the argument is {_describe(actual)} ({problem}). An integer argument "
        "converts to the declared dtype when one end is pl.INDEX; otherwise convert "
        "the value at the call site (pl.cast for a scalar dtype), or change the "
        "declaration to state the type the callee accepts."
    )


def _symbol_name(dim: object) -> str | None:
    """The declared dynamic symbol an extent names, if any."""
    if not _is_dynamic_dim(dim):
        return None
    return getattr(dim, "name_hint", None) or getattr(dim, "name", None)


def _shared_extent_mismatch(
    param_hints: dict[str, Any],
    param_names: list[str] | tuple[str, ...],
    args: tuple[Any, ...] | list[Any],
) -> str | None:
    """The first place one declared symbol receives two concrete extents.

    A symbol is a single dimension, so every occurrence of ``B`` in a
    declaration describes the same value. Checking arguments one at a time
    cannot see this: ``x: [B, 8]`` and ``y: [B, 8]`` each accept any extent
    independently, while ``x=[16, 8], y=[32, 8]`` is not a program where ``B``
    exists. Only concrete sample extents are compared — a symbolic argument
    supplies no value to contradict.
    """
    seen: dict[str, tuple[int, str, int]] = {}
    for name, arg in zip(param_names, args):
        declared = _declared_type(param_hints.get(name))
        if declared is None or not is_staged(arg):
            continue
        actual = _actual_type(arg)
        declared_shape = list(getattr(declared, "shape", []) or [])
        actual_shape = list(getattr(actual, "shape", []) or [])
        if len(declared_shape) != len(actual_shape):
            continue  # already reported by the per-argument check
        for axis, (declared_dim, actual_dim) in enumerate(zip(declared_shape, actual_shape)):
            symbol = _symbol_name(declared_dim)
            value = _extent_int(actual_dim)
            if symbol is None or value is None:
                continue
            prior = seen.get(symbol)
            if prior is None:
                seen[symbol] = (value, name, axis)
            elif prior[0] != value:
                value0, name0, axis0 = prior
                return (
                    f"the declared extent {symbol!r} is {value0} at parameter "
                    f"{name0!r} dim {axis0} but {value} at parameter {name!r} dim {axis}. "
                    "One symbol names one dimension; pass matching extents, or "
                    "declare independent symbols for dimensions that may differ."
                )
    return None


def check_declared_arguments(
    param_hints: dict[str, Any],
    param_names: list[str] | tuple[str, ...],
    args: tuple[Any, ...] | list[Any],
    fn_name: str,
) -> None:
    """Apply :func:`check_declared_argument` across one call site.

    Conversion is a no-op on the host lane, so this stays usable for a caller
    that wants the checks without the rewritten argument list.
    """
    coerce_declared_arguments(param_hints, param_names, args, fn_name)


def coerce_declared_arguments(
    param_hints: dict[str, Any],
    param_names: list[str] | tuple[str, ...],
    args: tuple[Any, ...] | list[Any],
    fn_name: str,
) -> tuple[Any, ...]:
    """Convert admitted scalar arguments, then apply the remaining checks.

    Returns the argument list to emit: a position inside the declared slot's
    integer lattice is replaced by the temporary that carries the conversion, so
    the callee's parameter and the call argument state the same dtype. A
    position outside the lattice keeps its refusal, and the world / bare-marker
    checks stay where they are (``_check_param_worlds``,
    :func:`reject_bare_data_type_annotations`).

    The conversion is emitted as its own statement rather than an inline ``Cast``
    argument: the device task-argument path reads a variable or a constant
    literal, and the temporary also makes the printed program identical to the
    spelling an author writes by hand.
    """
    frame = active_frame()
    converted = list(args)
    for index, (name, arg) in enumerate(zip(param_names, args)):
        declared = _declared_type(param_hints.get(name))
        if declared is None or not is_staged(arg):
            continue
        actual = _actual_type(arg)
        decision = classify_scalar_conversion(declared, actual)
        if decision.disposition is ArgumentDisposition.CONVERT:
            converted[index] = _convert_scalar(frame, name, arg, decision.target)
            continue
        problem = _mismatch(declared, actual)
        if problem is not None:
            raise TypeError(_refusal_message(name, declared, actual, problem, fn_name))
    problem = _shared_extent_mismatch(param_hints, param_names, converted)
    if problem is not None:
        raise TypeError(f"call to {fn_name!r}: {problem}")
    return tuple(converted)


def _convert_scalar(
    frame: object,
    name: str,
    arg: object,
    target: DataType | None,
) -> object:
    """Stage one scalar conversion, or re-type the sample on the host lane.

    A host dispatch has no traced value to convert: the argument is a sample
    whose dtype states storage, and the declaration supplies the ABI — so an
    admitted pair re-types the annotation-mode facade to the declared dtype
    (the same rule the torch INDEX bridge applies), and a lazily traced entry
    no longer leaks the sample's dtype into its signature. A facade that wraps
    a value from inside another trace keeps its expr for the caller's root
    check to refuse. Inside a trace the conversion is a real statement, so the
    call argument carries the declared dtype and the callee's signature stops
    depending on the argument.
    """
    if target is None:
        return arg
    if frame is not None:
        return frame.stage_scalar_conversion(arg, target, name=name)  # type: ignore[attr-defined]  # the Frame protocol
    try:
        arg.unwrap()  # type: ignore[attr-defined]  # is_staged guarantees the facade protocol
    except (RuntimeError, ValueError):
        # The annotation-only signal (the facades disagree on the exception
        # type exactly as ``_actual_type`` documents): a sample states storage,
        # so the declaration's dtype replaces it.
        return facade_from_ir_type(ir.ScalarType(target))
    return arg


def _dtype_marker(dtype: DataType) -> str:
    """The bare annotation marker for ``dtype`` (``pl.INDEX``).

    The ``pl`` constants are ``DataType`` re-exports under the member name, and
    ``str()`` is that name in lower case, so upper-casing recovers the spelling
    the author wrote.
    """
    return f"pl.{str(dtype).upper()}"


def reject_bare_data_type_annotations(
    param_hints: dict[str, Any],
    param_names: list[str] | tuple[str, ...],
    fn_name: str,
) -> None:
    """Refuse a parameter annotated with a bare dtype marker.

    ``vlen: pl.INDEX`` names a dtype but no world, so the parameter was a
    host knob: the value was baked at trace time and the artifact specialized
    per value (a caller that changed the scalar paid a recompile and got a
    different artifact identity) while ``pl.Scalar[pl.INDEX]`` stayed one
    runtime parameter. The failure was silent, so it is refused at decoration
    with the spelling that states the intent.
    """
    for name in param_names:
        hint = param_hints.get(name)
        if isinstance(hint, DataType):
            marker = _dtype_marker(hint)
            raise TypeError(
                f"kernel {fn_name!r}: parameter {name!r} is annotated {marker}, which "
                "states a dtype but not a parameter world — the value would be baked "
                "at trace time as a host knob, specializing the artifact per value. "
                f"Annotate pl.Scalar[{marker}] for a runtime scalar, or a plain Python "
                "type / no annotation for a host knob."
            )


def bind_declared_keywords(
    param_names: Sequence[str],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    fn_name: str,
    *,
    reserved: Collection[str] = (),
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Fold a call's keyword arguments into the declared parameter order.

    A sandbox Python call names its arguments however it likes, and the parser
    lane already binds a keyword against the callee's signature; without this
    fold the tracing lane alone refused a spelling the other lane compiles, so
    shared code could not name a parameter even when doing so was clearer. The
    binding is the parser lane's rule (``bind_call_args``) applied to live
    values: every declared parameter appears exactly once, in declaration
    order, and every lane below — the specialize lane's annotation map, the
    callee lane's variant key, the dispatch lane's bridge — reads one positional
    list, so the printed program and the artifact cannot depend on the spelling.

    ``reserved`` names stay call-site metadata: they are returned untouched for
    the detached emission (``attrs=`` always legal, ``device=`` on an
    Orchestrator callee), never bound to a parameter.

    Every malformed spelling is refused here rather than silently dropped: an
    unknown keyword, a parameter bound twice, a parameter bound nowhere, and a
    surplus positional argument would each have changed the specialized program
    with no diagnostic.

    Args:
        param_names: The callee's declared parameter order.
        args: The positional arguments, in call order.
        kwargs: The keyword arguments, keyed by name.
        fn_name: The callee's name, for diagnostics.
        reserved: Keyword names that are call-site metadata, not parameters.

    Returns:
        The arguments in declaration order, and the reserved keywords that were
        not bound.

    Raises:
        TypeError: An argument is unbound, duplicated, unknown, or surplus.
    """
    overrides = {name: kwargs[name] for name in reserved if name in kwargs}
    declared = list(param_names)
    if len(args) > len(declared):
        raise TypeError(
            f"call to {fn_name!r} passes {len(args)} positional argument(s) but the callee "
            f"declares {len(declared)}; remove the surplus argument"
        )
    bound: dict[str, Any] = dict(zip(declared, args))
    for name, value in kwargs.items():
        if name in reserved:
            continue
        if name not in declared:
            raise TypeError(
                f"call to {fn_name!r} passes no parameter {name!r}; the callee declares {declared}"
            )
        if name in bound:
            raise TypeError(
                f"call to {fn_name!r} binds parameter {name!r} both positionally and by keyword; bind it once"
            )
        bound[name] = value
    missing = [name for name in declared if name not in bound]
    if missing:
        raise TypeError(
            f"call to {fn_name!r} does not bind parameter(s) {missing}; every parameter must "
            "be supplied — a jit call specializes on all of them"
        )
    return tuple(bound[name] for name in declared), overrides
