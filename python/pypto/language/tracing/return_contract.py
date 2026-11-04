# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The traced-function return contract — comparison and pinning.

Shared tail of every root trace producer (the entrance foundation's
:class:`~pypto.language.tracing.entry.TracedCallable.invoke`, the decorator
adapter, and the program/jit envelopes): after a successful trace, a
declared flat return contract is compared against the traced return types
and pinned onto the completed ``ir.Function`` when they match. The Sema
intentionally sees no structural type for a flat multi-return signature
(the trace keeps ``return a, b`` as two ReturnStmt values), so the pin is
what makes the flat signature visible to callers and the printer.
"""

from __future__ import annotations

from collections.abc import Sequence

from pypto.ir.utils import (
    derived_return_types,  # noqa: F401 — re-exported; defined in the shared IR layer
    return_value_types,  # noqa: F401 — moved to the shared IR layer
    signature_expressible,  # noqa: F401 — re-exported; defined in the shared IR layer
)
from pypto.language.typing import annotation_to_return_slots, facade_class_for_type
from pypto.pypto_core import ir

__all__ = [
    "check_declared_return_values",
    "check_return_paths",
    "check_specializing_return_values",
    "derived_return_types",
    "pin_derived_return_types",
    "rebuild_with_return_types",
    "same_return_types",
    "return_value_types",
    "signature_expressible",
]


def same_return_types(left: Sequence[ir.Type], right: Sequence[ir.Type]) -> bool:
    """Structural return-contract comparison (auto-mapping absorbs per-trace Var identity)."""
    return len(left) == len(right) and all(
        ir.structural_equal(a, b, enable_auto_mapping=True) for a, b in zip(left, right)
    )


def rebuild_with_return_types(fn: ir.Function, return_types: Sequence[ir.Type]) -> ir.Function:
    """Rebuild ``fn`` with an explicit return contract pinned onto it."""
    params = list(zip(fn.params, fn.param_directions))
    return ir.Function(
        fn.name,
        params,
        list(return_types),
        fn.body,
        fn.span,
        fn.func_type,
        fn.level,
        fn.role,
        fn.attrs,
        fn.requires_runtime_binding,
    )


def check_return_paths(fn: ir.Function) -> list[ir.Type]:
    """The body's return types, refusing alternative paths that disagree.

    ``return_value_types`` collapses ``UnknownType`` over disagreeing paths.
    Keeping that marker would either publish an unannotated signature that
    hides the disagreement or type every caller as Unknown. The trace is the
    only place that knows the paths, so it is where the program is refused.
    """
    actual = return_value_types(fn)
    if any(isinstance(t, ir.UnknownType) for t in actual):
        raise TypeError(
            f"kernel {fn.name!r}: the traced return paths disagree, so one "
            "signature cannot describe the values the body returns. Give every "
            "path the same result type, or make the difference explicit in the "
            "declaration."
        )
    return actual


def check_declared_return_values(
    fn: ir.Function,
    declared: Sequence[ir.Type],
    actual: Sequence[ir.Type] | None = None,
) -> None:
    """Refuse a declared ``->`` that the traced values contradict.

    A declaration is checked against the values, never overwritten by them.
    A body with no value-bearing return is exempt: a jit entry may declare the
    output buffer it writes through its ``pl.Out`` parameter. Once the body
    does return a value, the declaration must describe it.
    """
    traced = list(return_value_types(fn) if actual is None else actual)
    if not traced:
        return
    # A flat multi-result declaration names the elements of one staged tuple.
    # A structural ``pl.Tuple[...]`` declaration is one value and stays so.
    if (
        len(traced) == 1
        and isinstance(traced[0], ir.TupleType)
        and not (len(declared) == 1 and isinstance(declared[0], ir.TupleType))
    ):
        traced = list(traced[0].types)
    declared_text = [str(t) for t in declared]
    actual_text = [str(t) for t in traced]
    if len(traced) != len(declared):
        raise TypeError(
            f"kernel {fn.name!r}: the declared `->` has {len(declared)} result(s) "
            f"{declared_text} but the traced body returns {len(traced)} value(s) "
            f"{actual_text}"
        )
    for index, (declared_type, actual_type) in enumerate(zip(declared, traced)):
        if not ir.structural_equal(declared_type, actual_type, enable_auto_mapping=True):
            raise TypeError(
                f"kernel {fn.name!r}: return position {index} is declared "
                f"{declared_type} but the traced body returns {actual_type} — "
                "the `->` must describe the values the body produces."
            )


def check_specializing_return_values(fn: ir.Function, return_type: object) -> None:
    """Refuse a specialized callee's declared ``->`` that its values contradict.

    The callee lane pins the variant's return types on the traced values
    (``pin_derived_return_types``), so this check reads them, never pins
    from the declaration. What it enforces is the declaration's static
    core: kind, dtype, arity and the extents the declaration states
    statically. Only a declared *dynamic* extent may differ — it
    specializes to whatever the values carry, the documented callee-lane
    contract. Without this check a nested callee whose ``->`` lied about
    dtype or static shape was silently retyped in the caller, while the
    same lie at an entry was already loud.

    A body with no value-bearing return is exempt, exactly as
    ``check_declared_return_values`` exempts it: a callee may declare the
    output buffer it writes through an ``Out`` parameter, and the derived
    pin speaks for the values.
    """
    if return_type is None:
        return
    slots = annotation_to_return_slots(return_type)
    if slots is None:
        return
    traced = return_value_types(fn)
    if not traced:
        return
    # A flat multi-result declaration names the elements of one staged
    # tuple; a structural ``pl.Tuple[...]`` declaration is one value and
    # stays so (the same normalization ``check_declared_return_values``
    # applies).
    if (
        len(traced) == 1
        and isinstance(traced[0], ir.TupleType)
        and not (len(slots) == 1 and isinstance(slots[0], ir.TupleType))
    ):
        traced = list(traced[0].types)
    if len(traced) != len(slots):
        raise TypeError(
            f"kernel {fn.name!r}: the declared `->` has {len(slots)} result(s) "
            f"{[str(s) for s in slots]} but the specialized body returns "
            f"{len(traced)} value(s) {[str(t) for t in traced]}"
        )
    for index, (slot, actual) in enumerate(zip(slots, traced)):
        if isinstance(actual, ir.UnknownType):
            continue
        if isinstance(slot, type):
            # A bare facade class states a kind, never a shape or dtype —
            # under-specified, so only the kind can be contradicted.
            facade = facade_class_for_type(actual)
            if facade is None or not issubclass(facade, slot):
                raise TypeError(
                    f"kernel {fn.name!r}: return slot {index} is declared "
                    f"{slot.__name__} (a bare facade class) but the specialized "
                    f"body returns {actual} — the declared kind does not match "
                    "the traced value"
                )
            continue
        if _specializing_slot_matches(slot, actual):
            continue
        raise TypeError(
            f"kernel {fn.name!r}: return position {index} is declared {slot} "
            f"but the specialized body returns {actual} — a callee's declared "
            "`->` must describe the values its specialized body produces; only "
            "a declared dynamic extent may specialize"
        )


def _specializing_slot_matches(declared: ir.Type, actual: ir.Type) -> bool:
    """Does the declaration's static core survive specialization to ``actual``?

    Structurally equal slots pass outright. Beyond that the comparison is
    the static core only — shape/dtype/kind for shaped types, element-wise
    for tuples, dtype for scalars — never the trace-deduced extras
    (``TileView``, memory space, storage): the annotation syntax has no word
    for those, and comparing them would flag every Vec-tile callee as a lie.
    Kinds with no statically comparable core cannot be contradicted and pass.
    """
    if ir.structural_equal(declared, actual, enable_auto_mapping=True):
        return True
    if isinstance(declared, ir.ShapedType) and isinstance(actual, ir.ShapedType):
        # A distributed tensor IS-A tensor: a declared ``-> pl.Tensor``
        # accepts a traced ``DistributedTensorType`` return (the annotation
        # syntax has no distributed flavour).
        if not isinstance(actual, type(declared)):
            return False
        if len(declared.shape) != len(actual.shape):
            return False
        if declared.dtype != actual.dtype:
            return False
        return all(_specializing_extent_matches(d, a) for d, a in zip(declared.shape, actual.shape))
    if isinstance(declared, ir.TupleType) and isinstance(actual, ir.TupleType):
        return len(declared.types) == len(actual.types) and all(
            _specializing_slot_matches(d, a) for d, a in zip(declared.types, actual.types)
        )
    if isinstance(declared, ir.ScalarType) and isinstance(actual, ir.ScalarType):
        return declared.dtype == actual.dtype
    return True


def _specializing_extent_matches(declared: ir.Expr, actual: ir.Expr) -> bool:
    """One extent of a declared slot vs the specialized value's extent.

    Two static extents must agree. Everything else passes: a declared
    dynamic extent (a symbol or an expression the annotation spelled)
    specializes to whatever the values carry, and a value-side dynamic
    extent cannot be contradicted by a declared static one loudly enough
    to refuse — the derived signature carries the value's own extent.
    """
    if isinstance(declared, ir.ConstInt) and isinstance(actual, ir.ConstInt):
        return declared.value == actual.value
    return True


def pin_derived_return_types(fn: ir.Function) -> ir.Function:
    """Pin the derived annotationless return contract onto ``fn``.

    A no-op when ``fn`` already carries return types — a declared ``->``
    *is* the signature (it was checked against the trace at build time),
    and a derived pin must never overwrite a declaration. The jit
    envelope's contract: upstream's specializer synthesized the inferred
    ``->`` into its rewritten source for every jit kernel; the traced
    counterpart materializes the recovered types into the Function
    signature, so a later ``pl.spmd_submit`` callee sees them as declared
    facts.

    A body with no value-bearing return falls back to its first
    ``Out`` / ``InOut`` parameter's type — the parser lane's convention for a
    jit entry written without ``return`` (its ``->`` is the output buffer the
    compiled program hands back). Without it the host entry loses an
    annotation the parser lane prints, and the print -> parse round-trip stops
    agreeing between lanes. A body whose return *was* traced but whose types
    cannot be pinned (paths disagree, or a body-local extent the signature
    cannot spell) stays unannotated: guessing there would put a type on the
    signature that the values do not support.
    """
    if fn.return_types:
        return fn
    derived = derived_return_types(fn)
    if derived:
        return rebuild_with_return_types(fn, derived)
    if return_value_types(fn):
        return fn
    out_type = _first_out_param_type(fn)
    if out_type is None:
        return fn
    return rebuild_with_return_types(fn, [out_type])


def _first_out_param_type(fn: ir.Function) -> ir.Type | None:
    """The first ``Out`` / ``InOut`` parameter's type — the parser lane's
    implicit single-result convention for a jit entry body without ``return``."""
    for param, direction in zip(fn.params, fn.param_directions):
        if direction in (ir.ParamDirection.Out, ir.ParamDirection.InOut):
            return param.type
    return None
