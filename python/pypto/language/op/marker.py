# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The sub-kernel ``Marker`` — the callable that emits a detached
``ir.Call``.

``pl.function`` *marks* (records the declared signature); the envelope
traces. The Marker is the mark's runtime face: a stdlib-op-shaped callable
whose ``__call__(*args)`` yields a facade wrapping a *detached*
``ir.Call(self.gvar, [a.unwrap() for a in args], deduce(...))``. The
caller's Frame bind / expression-statement hook collects it. Calls stay
rewrite-safe pass-through; the source rewriter is untouched; no
contextvar — a detached node outside a trace
GCs cleanly. The trace interpreter holds nothing;
the Markers live in the exec namespace.

The Marker is **self-sufficient**: every fact deduction needs —
param types + directions + ``->`` return types — comes from the *declared*
signature recorded at decoration, never from the callee's traced body. A
call to a helper traced later in the class body (a forward reference)
emits with the helper's declared signature, not ``UnknownType``.
"""

from __future__ import annotations

import inspect
import types
import typing
from typing import Any

from pypto.language.ambient import active_frame
from pypto.language.facade import is_facade
from pypto.language.op._validation import normalize_call_attrs
from pypto.language.protocol import is_staged, is_staged_annotation
from pypto.language.typing import (
    Scalar,
    annotation_to_return_types,
    annotation_to_type,
    wrap_expr,
)
from pypto.language.typing._host import coerce_host_scalar
from pypto.language.typing.arg_direction import param_directions, unmangle_hints, unmangle_param_name
from pypto.language.typing.tensor import NoDepMarker
from pypto.pypto_core import ir

# The function attr key for a hand-written external C++ kernel source (the
# parser frontend's EXTERNAL_SOURCE_ATTR, decorator.py): stored as an absolute
# path string on the Function attrs; the verifier skips the return-params
# check for a function carrying it (verify_return_params_explicit.cpp) and
# the backend compiles the referenced ``.cpp`` as the kernel instead of the
# traced body.
EXTERNAL_SOURCE_ATTR = "external_source"

# Call attrs whose values are vector<VarPtr> on the C++ side. Documented
# here for the kwarg-converter contract; the unwrapping itself is the
# open-world rule (``normalize_call_attrs`` — any staged value unwraps, so a
# Var-list attr normalizes the same way as any other list).
_VAR_LIST_ATTRS: tuple[str, ...] = (
    "dump_vars",
    "manual_dep_edges",
    "compiler_manual_dep_edges",
    "arg_direction_overrides_vars",
)


def _fold_external_source(attrs: dict[str, Any] | None, external_source: Any) -> dict[str, Any] | None:
    """Fold the ``external_source=`` decorator kwarg into the function attrs
    (mirroring the parser frontend's storage: ``attrs={"external_source": ...}``),
    merging with any user-supplied ``attrs={...}``."""
    if external_source is None:
        return attrs
    folded = dict(attrs) if attrs else {}
    folded[EXTERNAL_SOURCE_ATTR] = str(external_source)
    return folded


__all__ = ["Marker"]


class Marker:
    """A marked sub-kernel: callable, emits a detached ``ir.Call``.

    Data: ``name`` (the *definition* name — the parser frontend's
    ``GlobalVar`` uses ``func.name``, not the binding name), the declared
    signature (``params`` as Vars created with types, ``param_directions``,
    ``return_types``), ``gvar = GlobalVar(name)``, and the
    function-classification facts (``function_type``/``level``/``role``/
    ``attrs``, flowing to ``PtoSema.__init__`` at trace time).
    """

    def __init__(
        self,
        kernel: types.FunctionType,
        *,
        function_type: ir.FunctionType = ir.FunctionType.Opaque,
        level: ir.Level | None = None,
        role: ir.Role | None = None,
        attrs: dict[str, Any] | None = None,
        strict_ssa: bool = False,
        external_source: Any = None,
        localns: dict[str, object] | None = None,
    ) -> None:
        self.kernel = kernel
        self.name = kernel.__name__
        self.function_type = function_type
        self.level = level
        self.role = role
        self.attrs = _fold_external_source(attrs, external_source)
        self.strict_ssa = strict_ssa
        # The caller-frame names captured at decoration (the AST's
        # ``closure_vars``): PEP 563 string annotations may reference
        # enclosing-scope host names (``rows`` inside a test helper) that
        # ``get_type_hints`` cannot see otherwise.
        self._localns = localns
        self.gvar = ir.GlobalVar(self.name)

        # The function-classification facts + the derived role: an explicit
        # role wins; else the function type derives it — Orchestration →
        # Orchestrator, else SubWorker (the C++ Function's own derivation in
        # function.h). The ``device=`` dispatch kwarg on a
        # sub-kernel call is legal exactly when this effective role is
        # Orchestrator (the AST's ``_reject_unknown_call_kwargs`` checks the
        # callee's IR Function role, which the AST derives the same way).
        self._effective_role = (
            role
            if role is not None
            else (
                ir.Role.Orchestrator if function_type == ir.FunctionType.Orchestration else ir.Role.SubWorker
            )
        )

        # The declared signature: from the annotations + the source
        # (directions). The leading ``self`` of a program method is a host
        # param (passed through, never an IR param), excluded here;
        # the call site passes the method's args without it.
        hints = unmangle_hints(kernel, typing.get_type_hints(kernel, localns=localns))
        signature = inspect.signature(kernel)
        names = [
            unmangle_param_name(kernel, p.name)
            for p in signature.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if names and names[0] == "self":
            names = names[1:]
        directions = param_directions(kernel, localns=localns)
        params: list[ir.Var] = []
        param_directions_: list[ir.ParamDirection] = []
        for name in names:
            annotation = hints.get(name)
            if annotation is None or not is_staged_annotation(annotation):
                raise TypeError(
                    f"kernel {self.name!r}: parameter {name!r} needs a pl.Tensor/"
                    f"pl.Tile/pl.Scalar/pl.Array/pl.Ptr annotation for the new "
                    f"frontend, got {annotation!r}"
                )
            params.append(ir.Var(name, annotation_to_type(annotation), ir.Span.unknown()))
            param_directions_.append(directions.get(name, ir.ParamDirection.In))
        return_type = hints.get("return")
        self.param_names = names
        self.params = params
        self.param_directions = param_directions_
        # ``pl.Tuple[...]`` remains one structural TupleType return, while
        # builtin ``tuple[T1, T2]`` follows the parser's flat multi-return
        # contract. Both forms are explicit annotation syntax; a host tuple
        # returned from the body is never inferred as an annotation.
        self.return_types: list[ir.Type] = annotation_to_return_types(return_type)
        # The *declared* return types (from the ``->`` annotation only). A
        # later refinement may pin ``return_types`` with the recovered trace types for an
        # annotationless callee; the submit-tuple augmentation must stay on
        # the declared set (the AST's ``as_submit`` convention — its tuple
        # arity is governed by pl.submit, not the callee's implicit return).
        self.declared_return_types: list[ir.Type] = list(self.return_types)
        # Whether the *implicit* return set is final. A class-body Marker
        # starts unpinned (its body has not traced yet — the program decorator
        # pins it in definition order); a declared ``->`` never needs pinning.
        # Consumers that would guess from an empty return set (the submit
        # TASK_ID-only path) must reject an unpinned marker instead.
        self.implicit_returns_pinned = False

    @classmethod
    def _from_ir_function(cls, function: ir.Function) -> Marker:
        """Adapt a standalone traced function for a call inside ``pl.program``.

        Standalone ``@pl.function`` intentionally materializes an
        ``ir.Function``. A program method may capture that value in a Python
        closure; tracing replaces the closure value with this call-capable
        adapter while retaining the original Function in the completed
        Program.
        """
        marker = object.__new__(cls)
        marker._init_from_ir_function(function)
        # Assigned on the local, not inside _init_from_ir_function: a
        # ``self.kernel = None`` in a method would widen the declared
        # ``Marker.kernel`` type to ``FunctionType | None`` for every consumer.
        marker.kernel = None
        return marker

    def _init_from_ir_function(self, function: ir.Function) -> None:
        """Populate the Marker surface (the declared signature and
        classification facts) from a traced ``ir.Function`` — the shared
        body of ``_from_ir_function`` and the jit envelope's lazy global
        wrapper, which subclasses ``Marker`` and overrides only the call
        lane. The ``kernel`` slot is the caller's to fill (None for
        function-adapted markers)."""
        self.name = function.name
        self.function_type = function.func_type
        self.level = function.level
        self.role = function.role
        self.attrs = function.attrs
        self.strict_ssa = False
        self._localns = None
        self.gvar = ir.GlobalVar(function.name)
        self._effective_role = function.role
        self.param_names = [param.name_hint for param in function.params]
        self.params = list(function.params)
        self.param_directions = list(function.param_directions)
        self.return_types = list(function.return_types)
        self.declared_return_types = list(function.return_types)
        # The adaptation pins the return set only when it is a sound fact:
        # declared/derived return types, or a genuinely void body. A body
        # whose return paths disagree or whose types the signature cannot
        # spell stays unpinned — a submit referencing it stays loud instead
        # of guessing TASK_ID-only (see ``_check_submit_callee``).
        from pypto.ir.utils import return_value_types  # noqa: PLC0415

        self.implicit_returns_pinned = bool(function.return_types) or not return_value_types(function)

    def on_submit_reference(self) -> None:
        """A Submit references its callee without calling it. The base
        Marker's producer registers it eagerly (decoration, closure
        adaptation); a callee whose program-assembly registration is lazy
        (the jit envelope's global wrapper — register-at-first-call) overrides
        this to register at the reference site, so the Submit's gvar never
        dangles at program assembly."""

    def resolve_submit(self, args: tuple[Any, ...]) -> tuple[Marker, tuple[Any, ...]] | None:
        """Resolve this callee for a Submit reference with its kernel args.

        The base Marker is the frozen declaration: verbatim (``None`` — the
        submit lowers against this callee with the args unchanged). A callee
        whose references specialize (the jit envelope's global wrapper)
        overrides this to run the same per-call instantiation a call would,
        returning the specialized target and the surviving runtime args
        (positions whose values were baked into the variant's signature are
        dropped). Call and submit share one instantiation semantics; only
        the emitted instruction differs."""

    # —— the detached emission ——
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Yield a facade wrapping a detached ``ir.Call``.

        Pure construction: the caller's Frame bind / expression-statement
        hook collects the node. Guardrails, span-pointed (the parser frontend's
        call-kwarg discipline): ``attrs=`` is always legal (call-site
        metadata); ``device=`` only when the callee is an Orchestrator (the
        dispatch-device surface); every other keyword is loud — the AST
        rejects the same set (``_reject_unknown_call_kwargs``). Arg-count
        mismatch and host args are loud (``unwrap`` is a facade-only
        operation).
        """
        attrs: dict[str, Any] | None = kwargs.pop("attrs", None)
        device = kwargs.pop("device", None)
        if kwargs:
            raise TypeError(
                f"call to '{self.name}': keyword arguments are not accepted on a "
                f"sub-kernel call (parser-aligned — pass all arguments positionally); "
                f"only attrs= and (for an Orchestrator callee) device= are legal, "
                f"got {sorted(kwargs)}"
            )
        if device is not None and self._effective_role != ir.Role.Orchestrator:
            raise TypeError(
                f"call to '{self.name}': device= is only legal when the callee "
                f"is an Orchestrator (its effective role is {self._effective_role})"
            )
        if len(args) != len(self.params):
            raise TypeError(f"call to '{self.name}' expects {len(self.params)} argument(s), got {len(args)}")
        arg_exprs: list[ir.Expr] = []
        no_dep_indices: list[int] = []
        for i, (arg, param) in enumerate(zip(args, self.params)):
            # The ``pl.no_dep`` marker: unwrap the
            # ``NoDepMarker`` and record the slot for the ``arg_direction_overrides``
            # attr below — the AST parser's ``_strip_call_arg_markers``
            # threads the same per-arg override into the Call.
            if isinstance(arg, NoDepMarker):
                no_dep_indices.append(i)
            if is_staged(arg):
                arg_exprs.append(arg.unwrap())
                continue
            # The operand boundary: a host value reaching a sub-kernel
            # call is lifted to a typed constant at the call-site boundary —
            # typed by the declared param type when it is a ScalarType, else
            # by the canonical bare-literal dtype (the AST parser's
            # ``parse_constant`` rule: int→INDEX, float→DEFAULT_CONST_FLOAT,
            # bool→ConstBool). The parser frontend wrapped host constants
            # automatically; the tracing frontend must do the same before the
            # value crosses into the detached ``ir.Call``.
            arg_exprs.append(_host_call_arg(arg, param.type, self.name))

        call_attrs: dict[str, Any] | None = None
        if attrs is not None:
            if not isinstance(attrs, dict):
                raise TypeError(f"call to '{self.name}': attrs= must be a dict, got {attrs!r}")
            # The open-world rule shared with the op-call boundary: any staged
            # value unwraps (a bare expression becomes an IR expr; a list
            # element-wise), so ``attrs={"predicate": gate[0] > 0}`` carries
            # the comparison Expr the passes read.
            call_attrs = normalize_call_attrs(attrs, f"call to '{self.name}'")
        if device is not None:
            device_expr = _device_expr(device, self.name)
            call_attrs = call_attrs or {}
            call_attrs["device"] = device_expr
        # The dedicated ``pl.no_dep`` marker wins over a round-trip
        # ``attrs=`` dict (the AST's setdefault discipline: dedicated params
        # override generic attrs; the two sources are mutually exclusive in
        # practice — printed IR carries no ``pl.no_dep`` wrappers).
        if no_dep_indices:
            call_attrs = call_attrs or {}
            call_attrs["arg_direction_overrides"] = list(no_dep_indices)
        # The forward-sticky ``pl.dump_tag`` set
        # (traced): every call arg whose Var was tagged is merged into the
        # Call's ``dump_vars`` — the AST's ``_parse_kernel_call`` merge by
        # Var identity. Only whole-tensor Args are dumpable (the C++ side
        # drops what it never consumes).
        #
        # Unsafe-tier site: a sub-kernel call
        # during a trace reaches the active Frame through ``active_frame()`` to
        # read the dump-tag set — the one sanctioned non-funnel reader is
        # the callee dispatch, and this is its forward-sticky
        # merge. ``active_frame()`` is ``None`` outside a trace (the
        # no-trace guard) — a direct call (no enclosing trace)
        # builds the ``Call`` with no dump_vars.
        frame = active_frame()
        if frame is not None:
            dump_vars = [e for e in arg_exprs if isinstance(e, ir.Var) and frame.is_dump_tagged(e)]
            if dump_vars:
                call_attrs = call_attrs or {}
                call_attrs["dump_vars"] = dump_vars

        # Uniform deduction: read the *declared* signature — the
        # Marker's params + ``->`` return types — never the callee's traced
        # body. This is what makes the Marker self-sufficient.
        return_types = ir.deduce_call_return_type(self.params, arg_exprs, self.return_types)
        return _call_facade(self.gvar, arg_exprs, return_types, attrs=call_attrs)


def _host_call_arg(arg: Any, param_type: ir.Type, callee: str) -> ir.Expr:
    """Lift a host call argument to a typed constant at the operand boundary.
    Composes the B-axis funnel (``coerce_host_scalar``) for
    the bool/int/float→Const ladder and translates its error to the
    call-site message (call-site tailored logic around the common
    funnel). The dtype is the canonical bare-literal dtype (``INDEX`` /
    ``DEFAULT_CONST_FLOAT``) — int→INDEX regardless of the param: typing the
    constant by the *param annotation* would be annotation-as-typing
    — typing flows from values, not annotations. Any dtype adaptation (e.g.
    an int arg widening to an FP32 scalar param, host promotion) is the C++
    deduce layer's business, which sees both sides."""
    try:
        return coerce_host_scalar(arg, callee, f"argument {arg!r}")
    except TypeError:
        raise TypeError(
            f"call to '{callee}': argument {arg!r} is a host value of an unsupported "
            f"type ({type(arg).__name__}) — a sub-kernel call takes staged facades "
            f"(pl.Tensor/pl.Tile/pl.Scalar/...) or host bool/int/float (lifted to a "
            f"typed constant at the call-site boundary)"
        ) from None


def _device_expr(device: Any, callee: str) -> ir.Expr:
    """Lift the ``device=`` dispatch value to an ``ir.Expr`` at the op-arg
    boundary: a Scalar unwraps; a raw ``ir.Expr`` passes; a host
    int/bool literal lifts to a typed constant via the canonical rule
    (int→INDEX — matching the AST parser's literal dtype and the unroll
    pass's loop-var substitution, so a ``device=r`` loop-var round-trips;
    bool→ConstBool). The AST is permissive about the value form
    (any IR expression works — the comm-collection pass inspects the
    resolved expression)."""
    if is_facade(device):
        return device.unwrap()
    if isinstance(device, ir.Expr):
        return device
    return coerce_host_scalar(device, f"call to '{callee}'", "device=")


def _call_facade(
    gvar: ir.GlobalVar,
    arg_exprs: list[ir.Expr],
    return_types: list[ir.Type],
    *,
    attrs: dict[str, Any] | None = None,
) -> Any:
    """Wrap the detached ``ir.Call`` in the facade matching the (deduced)
    return type — so the caller's Frame bind / expression-statement hook
    recognizes it as staged and collects it. Mirrors ``_make_call_with_return_type``'s
    Call shapes (0 → untyped, 1 → typed, >1 → flat ``TupleType`` over the
    element types — the multi-return lane). An
    annotation-less callee yields an untyped Call
    wrapped in the minimal Scalar — the return-type *check* is
    where ``->`` is required, never for deduction. ``attrs`` (call-site
    metadata / the
    dispatch ``device``) lands on the Call when present."""
    span = ir.Span.unknown()
    if not return_types:
        if attrs:
            call = ir.Call(gvar, arg_exprs, {}, attrs, ir.UnknownType(), span)
        else:
            call = ir.Call(gvar, arg_exprs, span)
        return Scalar(expr=call)
    if len(return_types) == 1:
        if attrs:
            call = ir.Call(gvar, arg_exprs, {}, attrs, return_types[0], span)
        else:
            call = ir.Call(gvar, arg_exprs, return_types[0], span)
        t = return_types[0]
    else:
        # Multi-return callee call: the call's type
        # is a flat ``TupleType`` (the AST parser's ``_make_call_with_return_type``
        # shape) and ``wrap_expr`` wraps it in the ``Tuple`` facade, whose
        # ``__getitem__``/``__iter__`` share a ``_tuple_tmp`` across element
        # projections via ``_emit.py::_structural_projection`` (the Submit-style
        # split). The tuple-element binds no longer need a separate gate — the
        # machinery ``pl.submit`` uses is reused here.
        if attrs:
            call = ir.Call(gvar, arg_exprs, {}, attrs, ir.TupleType(return_types), span)
        else:
            call = ir.Call(gvar, arg_exprs, ir.TupleType(return_types), span)
        return wrap_expr(call, ir.TupleType(return_types))
    return wrap_expr(call, t)
