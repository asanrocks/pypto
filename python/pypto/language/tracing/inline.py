# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parsing-frontend bodies calling traced inline helpers — the import bridge.

A ``@pl.jit`` body is parsed from source; a ``@pl.jit_trace.inline`` helper
collects its IR by tracing. Composing them one-way lets shared leaves migrate
before their callers, without either frontend learning the other's body
language:

    parser entry -> parser helper -> traced inline -> traced inline

The callee reaches the caller's ``@pl.program`` the way a hand-written external
``ir.Function`` does: specialization folds each call site's compile-time values
from the call's own source and binds the site in the generated program's parse
namespace under a private name, and the parser — when it reaches the rewritten
call, holding the arguments as parsed IR values — materializes the helper
variant from those values' own types. ``InlineFunctions`` then splices it, so
nothing about the helper survives in the artifact — no dispatch, no ABI, no
artifact cache entry.

The specialization deliberately runs at parse time. A source-level
reconstruction of the caller's arguments (the metadata walk a parser dependency
is declared from) models tensors as shape/dtype/layout only: it cannot see an
explicit tile memory space, a view's valid region, or the distributed kind of a
window, and a body traced against that approximation is traced against the
wrong type. The parsed argument's IR type still has every field, so it — not a
reconstruction — is what specializes the helper.

Two rules keep the composition sound:

- **Inline only, transitively.** Everything the bridge imports is an Inline
  function, so the program has no traced ABI callee to lower. A traced body
  reaching an ``incore``/``opaque``/``graph`` callee is refused by name.
- **The helper's declaration is its contract.** A parser dependency is
  re-declared from its call site; a traced helper is not. Its staged parameter
  types come from the call site (that is what specializes it), and they are
  checked against the declaration by the tracer's own argument contract. A
  contradiction the parser lane would silently absorb is loud here.

Compile-time (baked) parameters are folded from the call site's source, exactly
as a parser dependency's ``pl.constexpr`` arguments are. Their folded text is
the variant's identity, so two call sites with different constants get two
helpers and the caller's cache key covers both.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pypto.jit import _ast_analysis, _call_binding, _tensor_meta
from pypto.jit.specializer import (
    TRACED_INLINE_BINDING_PREFIX,
    DynDim,
    TensorMeta,
    constant_source,
    free_name_source,
    func_name_lookup,
)
from pypto.pypto_core import DataType, ir

__all__ = [
    "TracedInlineBinding",
    "TracedInlineBuilder",
    "TracedInlineCallSite",
    "TracedInlineImport",
    "baked_call_identity",
    "discover_traced_inline",
    "is_traced_inline",
]


@dataclass(frozen=True)
class TracedInlineBinding:
    """One traced-inline helper a caller's source reaches by ``call_name``."""

    call_name: str
    helper: Any


@dataclass(frozen=True)
class TracedInlineCallSite:
    """One call site of a traced-inline helper, as specialization folded it.

    ``param_names`` is the helper's declared parameter order, which a keyword
    argument is normalized against; ``baked`` are the parameters the trace
    consumes (compile-time values), so the generated call does not pass them.
    ``baked_values`` are those parameters' folded host values — the call's own
    source is the only statement of them, and the parse-time resolution re-feeds
    them to the trace.
    """

    call_name: str
    helper: Any
    param_names: tuple[str, ...]
    baked: frozenset[str]
    baked_values: Mapping[str, Any]


class TracedInlineImport:
    """The parse-namespace binding for one traced-inline call site.

    The generated program rewrites each call site to this binding's name, with
    its compile-time arguments dropped. When the parser reaches that call the
    arguments are parsed IR values carrying their full types — the authoritative
    types, which no source-level reconstruction produced — and ``materialize``
    specializes the helper from exactly those.
    """

    # The parser's bare-name call dispatch recognizes this marker and nothing
    # else about this class, so the parser layer stays free of any jit import:
    # the binding owns the site's specialization, the parser only hands it the
    # parsed arguments.
    _pto_traced_inline = True

    def __init__(self, binding_name: str, site: TracedInlineCallSite, builder: TracedInlineBuilder) -> None:
        self.binding_name = binding_name
        self._site = site
        self._builder = builder

    @property
    def runtime_params(self) -> tuple[str, ...]:
        """The helper's staged parameters, in declaration order."""
        return tuple(param for param in self._site.param_names if param not in self._site.baked)

    def materialize(self, args: Sequence[ir.Expr]) -> ir.Function:
        """Specialize this site's helper from the parsed argument IR types."""
        return self._builder.materialize_call(self._site, args)

    def coerce_declared_scalar_args(
        self, args: Sequence[ir.Expr], builder: Any, span: ir.Span
    ) -> list[ir.Expr]:
        """Convert an in-lattice scalar argument to its declared dtype at the call.

        The helper's declared scalar dtype is the ABI (the shared admission
        rule, :mod:`pypto.language.scalar_conversion`), but the specialized
        helper is typed from the parsed argument — so an argument whose dtype
        differs yet lies inside the declared slot's integer lattice is
        converted *before* ``materialize``, bound to the same temporary name
        both other lanes give it. One signature then serves every integer
        caller. Everything outside the lattice keeps the refusal
        ``_check_declared`` raises at the same boundary, and non-scalar
        parameters pass through untouched.
        """
        from pypto.language.scalar_conversion import (  # noqa: PLC0415
            ArgumentDisposition,
            classify_scalar_conversion,
            conversion_temp_name,
        )
        from pypto.language.tracing.argument_contract import _declared_type  # noqa: PLC0415

        hints = _declared_hints(self._site.helper)
        converted: list[ir.Expr] = []
        for param, arg in zip(self.runtime_params, args):
            declared = _declared_type(hints.get(param))
            decision = classify_scalar_conversion(declared, arg.type) if declared is not None else None
            if (
                decision is not None
                and decision.disposition is ArgumentDisposition.CONVERT
                and decision.target is not None
            ):
                temp_name = conversion_temp_name(param, decision.target)
                converted.append(builder.let(temp_name, ir.cast(arg, decision.target, span=span), span=span))
                continue
            converted.append(arg)
        return converted


def is_traced_inline(value: Any) -> bool:
    """Whether ``value`` is a ``@pl.jit_trace.inline`` helper object.

    Both the eager and the lazy traced lane are accepted: a helper whose
    parameters are all facade-annotated has an eager record, one with an
    unannotated parameter re-traces per call. Either way the call-site
    specialization below is the lane that runs, because the helper is being
    called from inside another body — the same rule the tracer applies to a
    traced callee.
    """
    from pypto.language.tracing.decorators.jit_trace import _JitSpecializer  # noqa: PLC0415

    return isinstance(value, _JitSpecializer) and value._function_type == ir.FunctionType.Inline


def discover_traced_inline(func: Any) -> list[TracedInlineBinding]:
    """Traced-inline helpers ``func``'s body calls, in first-appearance order.

    Resolution is the parser dependency rule: a bare name looked up through the
    function's globals with its closure bindings taking precedence, which is how
    an aliased import or a factory-captured helper resolves.

    One shadowing rule guards it: a name the body binds *locally* — a
    parameter, an assignment target, a comprehension binder, a nested
    ``def``/``class``/``import as`` — never reaches a global binding in the
    function's own Python, so a local named like a traced helper is refused
    loudly here instead of being silently rewritten to the helper (a
    miscompile: the compiled program would call what the Python never calls).
    A module-level alias (``h = helper``) is not a local, so the aliased-import
    path stays open.
    """
    func_def = _ast_analysis.get_func_def(func)
    called = _ast_analysis.collect_all_called_names(func_def)
    local_names = _ast_analysis.body_local_names(func)
    namespace = func_name_lookup(func)
    found: list[TracedInlineBinding] = []
    seen: set[str] = set()
    for name in called:
        if name in seen:
            continue
        helper = namespace.get(name)
        if is_traced_inline(helper):
            seen.add(name)
            if name in local_names:
                raise _shadowed_call_name_error(func, func_def, name)
            found.append(TracedInlineBinding(call_name=name, helper=helper))
    return found


def _shadowed_call_name_error(func: Any, func_def: ast.FunctionDef, name: str) -> TypeError:
    """The loud refusal for a locally-bound name that also resolves to a helper."""
    call = next(
        (
            node
            for node in ast.walk(func_def)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        ),
        None,
    )
    where = f" (first at line {call.lineno})" if call is not None else ""
    return TypeError(
        f"@pl.jit function '{getattr(func, '__qualname__', func)}' binds the name "
        f"{name!r} locally{where}, so a call spelled {name}(...) runs that local "
        "binding in Python — but the traced-inline import resolves the same name "
        f"to the module-level traced helper {name!r}, which would compile a call "
        "the function never makes. Rename the local binding or the helper."
    )


def _declared_hints(helper: Any) -> dict[str, Any]:
    """The helper's resolved parameter annotations, by parameter name.

    ``_param_hints`` is the same map the tracer's own argument contract reads,
    so a declaration means here exactly what it means there.
    """
    return dict(helper._param_hints)


def _staged(helper: Any, name: str) -> bool:
    """Whether the helper declares ``name`` as a staged (runtime) parameter."""
    from pypto.language.tracing.decorators.jit_trace import _staged_declared  # noqa: PLC0415

    return _staged_declared(_declared_hints(helper).get(name))


def _scalar_declaration(helper: Any, name: str) -> DataType | None:
    """The dtype the helper declares for a staged *scalar* parameter.

    A staged scalar's dtype is settled by its declaration: the call site supplies
    a runtime value, not a type, so the declaration is the only statement of
    what the helper reads. Tensor parameters return ``None`` — their type comes
    from the call site, which is what specializes them.
    """
    from pypto.language.typing.scalar import Scalar  # noqa: PLC0415

    hint = _declared_hints(helper).get(name)
    return hint.dtype if isinstance(hint, Scalar) else None


def _baked_names(helper: Any) -> frozenset[str]:
    """Helper parameters the trace consumes, so the emitted call drops them.

    Everything that is not staged: a ``pl.constexpr`` marker and an unannotated
    host knob are both values the tracer bakes into the collected IR, exactly as
    it does for a traced caller.
    """
    return frozenset(name for name in helper.param_names if not _staged(helper, name))


def _call_arg_nodes(param_names: Sequence[str], node: ast.Call, call_name: str) -> dict[str, ast.expr]:
    """The call's argument *expressions* by parameter name, fully validated.

    Not ``_call_binding.build_param_mapping``: that answers with a caller-side
    *reference* (a name, a subscript) for metadata resolution, and returns
    ``None`` for everything else. A compile-time argument is usually a literal,
    which has an expression but no reference, and the expression is exactly what
    has to be folded.

    The bind must be complete and unambiguous: an argument this function drops
    is one the generated call never passes, so a surplus, unknown, duplicated or
    missing argument is refused here instead of changing the program silently.
    """
    return _call_binding.bind_call_args(param_names, node, call_name)


class _TracedDep:
    """The ``JITFunction``-shaped view metadata resolution needs.

    ``resolve_dep_call_metadata`` reads only the callee's declaration surface —
    its parameter names, its Python function (for annotation-derived dtypes and
    layouts) and its display name. A traced helper has all three; this adapter
    keeps the shared resolver from growing a second, subtly different copy for
    the tracing lane.
    """

    def __init__(self, call_name: str, helper: Any) -> None:
        self.__name__ = call_name
        self._func_type = "inline"
        self._helper = helper

    def _param_names(self) -> list[str]:
        return list(self._helper.param_names)

    @property
    def _func(self) -> Any:
        return self._helper.kernel


def _walk_facade(
    helper: Any,
    name: str,
    meta: TensorMeta | None,
    dtype: DataType | None,
    call_name: str,
) -> Any:
    """The *provisional* facade for one staged parameter, from walk metadata.

    Only the metadata walk's model — tensor shape/dtype/layout and scalar
    dtypes — is available before the caller is parsed, so this is a lossy
    approximation used solely to answer a traced call's return metadata for
    parser dependency signatures (``return_meta``). It never types the imported
    variant: the parse-time resolution does that, from the parsed argument's
    full IR type. A parameter the walk cannot model raises, and the caller
    declines the provisional answer instead of guessing.
    """
    from pypto.language.typing.dynamic import DynVar  # noqa: PLC0415
    from pypto.language.typing.scalar import Scalar  # noqa: PLC0415
    from pypto.language.typing.tensor import Tensor  # noqa: PLC0415

    if meta is not None:
        # A dynamic extent must stay a symbol, which the facade types as a plain
        # extent slot rather than an int.
        shape: Any = [dim if isinstance(dim, int) else DynVar(dim.name) for dim in meta.shape]
        return Tensor(shape, meta.dtype, layout=meta.layout, _annotation_only=True)
    if dtype is not None:
        return Scalar(dtype, _annotation_only=True)
    declared = _scalar_declaration(helper, name)
    if declared is not None:
        # The declaration states the dtype, but the walk could not resolve the
        # argument's. Tracing with the declaration would retype the caller's
        # value at the ABI boundary, so this declines the provisional answer.
        raise TypeError(
            f"call to traced helper {call_name!r}: parameter {name!r} is a declared "
            f"pl.Scalar[{declared}] whose argument type the metadata walk could not "
            "resolve"
        )
    raise TypeError(
        f"call to traced helper {call_name!r}: parameter {name!r} is staged but the "
        "metadata walk cannot type it here"
    )


def _fold_baked(
    arg: ast.expr,
    caller_scope: _tensor_meta._StaticScope,
    caller_constexpr: Mapping[str, str],
) -> tuple[Any, str] | None:
    """The compile-time value and its folded text for one baked argument.

    The call's own source is the only statement of the value, so it is evaluated
    the way a parser dependency's ``pl.constexpr`` argument is: in the caller's
    namespace with its locals removed (a local shadows a same-named global) and
    builtins stripped. ``constant_source`` then renders the value by the rule the
    generated source would use, and that text — not the value — is the variant's
    identity, so a cache key can never stand still while the compiled body moves.

    ``None`` means the argument has no compile-time value. Callers differ in what
    that means: specialization reports it against the parameter, while cache-key
    composition skips it (the build will report it soon enough).
    """
    namespace = {k: v for k, v in caller_scope.namespace.items() if k not in caller_scope.shadowed}
    namespace["__builtins__"] = {}
    for forwarded_name, text in caller_constexpr.items():
        try:
            namespace[forwarded_name] = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            continue
    try:
        value = eval(ast.unparse(arg), namespace)  # noqa: S307 - the user's own source, own namespace
    except Exception:  # noqa: BLE001 - every failure means "no compile-time value"
        return None
    text = constant_source(value)
    return None if text is None else (value, text)


def _baked_value(
    call_name: str,
    param: str,
    arg: ast.expr,
    caller_scope: _tensor_meta._StaticScope,
    caller_constexpr: Mapping[str, str],
) -> tuple[Any, str]:
    """``_fold_baked`` for specialization: no compile-time value is a loud error."""
    folded = _fold_baked(arg, caller_scope, caller_constexpr)
    if folded is not None:
        return folded
    raise TypeError(
        f"call to traced helper {call_name!r}: parameter {param!r} is bound to "
        f"'{ast.unparse(arg)}', which has no compile-time value at this call site. "
        "Pass a literal, one of the caller's own pl.constexpr parameters, or a "
        "module-level/closure constant."
    )


def baked_call_identity(caller_func: Any, caller_constexpr: Mapping[str, str]) -> tuple[tuple[Any, ...], ...]:
    """Identity records for the compile-time values this body bakes into helpers.

    Keyed by helper, call name, ordinal and parameter, exactly like the parser
    lane's ``pl.constexpr`` records, and for the same reason: the *folded text*
    is what the compiled body contains, and it is the only form that catches a
    value reached through an attribute chain (``cfg.BLOCK``) — a name the
    source hash cannot render. A call site that cannot fold is skipped here; the
    specialization reports it with the parameter it was bound to.
    """
    scope = _tensor_meta._StaticScope.of(caller_func, caller_constexpr)
    records: list[tuple[Any, ...]] = []
    for binding in discover_traced_inline(caller_func):
        baked = _baked_names(binding.helper)
        if not baked:
            continue
        nodes = _call_binding._dep_call_nodes(caller_func, binding.call_name)
        for ordinal, node in enumerate(nodes):
            args = _call_arg_nodes(list(binding.helper.param_names), node, binding.call_name)
            for param in sorted(baked):
                folded = _fold_baked(args[param], scope, caller_constexpr)
                if folded is None:
                    continue
                records.append((binding.call_name, ordinal, param, folded[1]))
    return tuple(records)


@dataclass
class TracedInlineBuilder:
    """Materializes the traced helpers one ``@pl.jit`` program build imports.

    One builder per build: it owns the assembly that both allocates the imported
    functions' names and collects the callees they reach, so a helper called at
    two sites with one type list is one function, and a name is never handed out
    twice.

    Specialization itself happens at parse time (``materialize_call``, reached
    through each site's ``TracedInlineImport``): the call's arguments are parsed
    IR values then, and their own types — with layout, view, memory space,
    storage and kind — are what the helper is traced against. The pre-parse
    metadata walk never types helper arguments; its one remaining job is
    answering a traced call's *return* metadata for parser dependency
    signatures, and that answer is provisional (``return_meta``): traced into a
    scratch assembly that never joins the program, memoized per lossy key, and
    declined on any disagreement — the parse-time resolution is authoritative
    and raises the real error.
    """

    used_names: set[str]
    # Identifiers the program's caller bodies bind locally (parameters,
    # assignment targets, comprehension binders). A minted binding name must
    # skip them: a caller local of that name does not shadow the parse-
    # namespace binding for a bare-name call, so a collision would run the
    # helper at a call site the caller's own Python gives to its local.
    reserved_locals: frozenset[str] = frozenset()
    assembly: Any = None
    # Specialized variants by identity: helper, then one part per parameter
    # (staged type or folded constant text).
    _by_key: dict[tuple[Any, ...], ir.Function] = field(default_factory=dict[tuple[Any, ...], ir.Function])
    # Provisional return-metadata answers, keyed by the same key shape but on
    # the lossy walk facades — a different value space, so a provisional answer
    # can never alias a real variant.
    _provisional: dict[tuple[Any, ...], tuple[list[TensorMeta] | None, list[ir.Type]]] = field(
        default_factory=dict[tuple[Any, ...], tuple[list[TensorMeta] | None, list[ir.Type]]]
    )
    # Scratch assembly for provisional traces: functions registered here never
    # join the program, and neither do the callees a provisional body reaches.
    _scratch: Any = None
    # Registered call sites by their parse-namespace binding name.
    _sites: dict[str, TracedInlineCallSite] = field(default_factory=dict[str, TracedInlineCallSite])
    _binding_count: int = 0

    def __post_init__(self) -> None:
        if self.assembly is None:
            from pypto.language.tracing.decorators.jit_trace import _ProgramAssembly  # noqa: PLC0415

            self.assembly = _ProgramAssembly()
        # The caller's program already owns names (its generated methods). Taking
        # them out of circulation first is what lets a helper share a name with a
        # parser dependency without the program rejecting the duplicate.
        self.assembly.reserve(self.used_names)

    # —— reads the program build consumes ——

    @property
    def functions(self) -> list[ir.Function]:
        """Every function the imports introduced, in registration order."""
        return list(self.assembly.functions)

    # —— the parse-time bindings ——

    def add_site(self, site: TracedInlineCallSite) -> str:
        """Register one call site and mint its parse-namespace binding name.

        The generated source names each site's rewritten call after this
        binding; one name per site, because the site's folded compile-time
        values are its own. The prefix keeps the name clear of every identifier
        the generated source can contain: folded free names are replaced by
        literals, generated methods are reached through ``self.``, and dynamic
        symbols are declared under the names the user wrote.

        A user-written local of that exact name is the one form the prefix
        cannot stay clear of, so the mint skips it: the parser's bare-name call
        dispatch resolves the parse-namespace binding even where the caller's
        own local of that name should win, so a collision would silently run
        the helper at the caller's call to its local. The skip set is the
        program's caller bodies' local names plus the reserved and already
        minted names, and it is self-healing — the counter simply moves past
        every collision.
        """
        while True:
            self._binding_count += 1
            name = f"{TRACED_INLINE_BINDING_PREFIX}{self._binding_count}"
            if name not in self.reserved_locals and name not in self.used_names and name not in self._sites:
                break
        self._sites[name] = site
        return name

    def imports(self) -> dict[str, TracedInlineImport]:
        """The parse-namespace bindings for every registered call site."""
        return {name: TracedInlineImport(name, site, self) for name, site in self._sites.items()}

    # —— parse-time specialization (authoritative) ——

    def materialize_call(self, site: TracedInlineCallSite, args: Sequence[ir.Expr]) -> ir.Function:
        """Specialize one call site's helper from its parsed argument IR types.

        Each staged parameter takes the *parsed* argument's own type — the full
        ``ir.Type``, with layout, view, memory space, storage and kind — so the
        traced body is collected against exactly the value the call passes.
        Baked parameters take the values folded from the call site's source at
        specialization time. The materialized function is program-unique and
        registered with the build's assembly, so a second call site at the same
        types and constants reuses it instead of tracing again.
        """
        from pypto.language.tracing.decorators.jit_trace import _variant_key_part  # noqa: PLC0415
        from pypto.language.typing._ir_facade import facade_from_ir_type  # noqa: PLC0415

        helper = site.helper
        hints = _declared_hints(helper)
        runtime = [param for param in site.param_names if param not in site.baked]
        if len(args) != len(runtime):
            # The rewrite drops exactly the baked parameters, so a mismatch here
            # means the generated call and the binding disagree — an internal
            # error, not a user diagnostic.
            raise TypeError(
                f"call to traced helper {site.call_name!r} passes {len(args)} "
                f"argument(s) but {len(runtime)} runtime parameter(s) survived the "
                "rewrite"
            )
        annotations: dict[str, Any] = {}
        key_parts: list[Any] = [id(helper)]
        # Declared order, not runtime-then-baked: ``bind_parameters`` binds the
        # instrumented kernel positionally in annotation order, so any other
        # insertion order scrambles an interspersed baked/staged helper's
        # arguments (the staged value after the knob lands on the knob) — and
        # the annotation map's order is also what the shared variant key and
        # the in-flight cycle guard canonicalize by name to survive.
        runtime_args = iter(args)
        for param in site.param_names:
            if param in site.baked:
                value = site.baked_values[param]
                annotations[param] = value
                key_parts.append(("const", constant_source(value)))
                continue
            facade = facade_from_ir_type(next(runtime_args).type)
            _check_declared(site.call_name, param, facade, hints)
            annotations[param] = facade
            key_parts.append(_variant_key_part(facade))
        key = tuple(key_parts)
        cached = self._by_key.get(key)
        if cached is not None:
            return cached
        function = self._trace(site.call_name, helper, annotations, self.assembly)
        self._by_key[key] = function
        return function

    def _trace(
        self, call_name: str, helper: Any, annotations: Mapping[str, Any], assembly: Any
    ) -> ir.Function:
        """Trace ``helper`` against ``annotations`` and register the result in ``assembly``.

        The trace runs under ``assembly``, so any helper the body calls is
        specialized and registered alongside it — that is what makes the
        inline-only rule checkable on the *reachable* set rather than on the
        call's own spelling.
        """
        from pypto.language.tracing.decorators.jit_trace import (  # noqa: PLC0415
            _ACTIVE_ASSEMBLY,
            _renamed_function,
        )

        before = len(assembly.functions)
        token = _ACTIVE_ASSEMBLY.set(assembly)
        try:
            function = helper._trace_with(dict(annotations), derive_return=True)
        finally:
            _ACTIVE_ASSEMBLY.reset(token)
        _reject_non_inline_reachable(assembly.functions[before:], call_name)
        name = assembly.unique_name(function.name)
        if name != function.name:
            function = _renamed_function(function, name)
        assembly.register(function)
        return function

    def _scratch_assembly(self) -> Any:
        """The throwaway assembly provisional traces register into."""
        if self._scratch is None:
            from pypto.language.tracing.decorators.jit_trace import _ProgramAssembly  # noqa: PLC0415

            self._scratch = _ProgramAssembly()
        return self._scratch

    # —— metadata for the caller's later calls ——

    def resolver_factory(
        self, discover_deps: _call_binding.DiscoverDeps
    ) -> _tensor_meta.TracedInlineResolverFactory:
        """Build reply maps for whichever body the metadata descent is walking.

        A resolver closes over the function it was built for, so a parser
        dependency's body cannot be typed with its caller's map: the callee's
        traced calls would resolve against the wrong source and locals. The
        metadata walk asks this factory for the map of the body it is about to
        scan, so a nested body always gets its own.
        """

        def factory(func: Any, func_type: str, caller_constexpr: Mapping[str, str]) -> Mapping[str, Any]:
            return self.resolvers(func, func_type, caller_constexpr, discover_deps=discover_deps)

        return factory

    def resolvers(
        self,
        caller_func: Any,
        caller_func_type: str,
        caller_constexpr: Mapping[str, str],
        *,
        discover_deps: _call_binding.DiscoverDeps,
    ) -> dict[str, Any]:
        """One metadata answer per traced helper this caller's body reaches.

        Metadata inference runs source-ordered and may reach a traced call
        before the call sites are specialized, so it is handed this map: a
        call name resolves to the helper's materialized result types, which is
        the only statement of what the call produces (there is no parser source
        to descend into).
        """
        answers: dict[str, Any] = {}
        for binding in discover_traced_inline(caller_func):

            def answer(
                call: ast.Call,
                targets: Sequence[ast.expr],
                local: Mapping[str, TensorMeta],
                _binding: TracedInlineBinding = binding,
            ) -> list[Any]:
                return self.return_targets(
                    _binding,
                    call,
                    targets,
                    caller_func,
                    caller_func_type,
                    local,
                    caller_constexpr,
                    discover_deps=discover_deps,
                    resolvers=answers,
                )

            answers[binding.call_name] = answer
        return answers

    def return_meta(
        self,
        binding: TracedInlineBinding,
        call_node: ast.Call,
        caller_func: Any,
        caller_func_type: str,
        locals_meta: Mapping[str, TensorMeta],
        caller_constexpr: Mapping[str, str],
        *,
        discover_deps: _call_binding.DiscoverDeps,
        resolvers: Mapping[str, Any] | None = None,
    ) -> tuple[list[TensorMeta] | None, list[ir.Type]] | None:
        """The helper's *provisional* result metadata at one call site, or ``None``.

        A caller that feeds a helper's result into a later parser dependency
        needs that result's shape and dtype to declare the dependency's
        signature, but the caller has not been parsed yet, so the helper can
        only be traced against the metadata walk's lossy view of the arguments.
        The answer types the dependency declaration and nothing else: it is
        traced into a scratch assembly that never joins the program, memoized
        per lossy key, and declined (``None``) on any disagreement. The
        authoritative specialization happens at parse time, from the parsed
        argument's own IR type, and raises the real error there.
        """
        helper = binding.helper
        call_name = binding.call_name
        try:
            hints = _declared_hints(helper)
            baked = _baked_names(helper)
            adapter = _TracedDep(call_name, helper)
            # Only the staged arguments need metadata here; scalar parameters
            # keep their declaration's dtype, which is the contract their call
            # must meet.
            scalar_hints = {
                param: dtype
                for param in helper.param_names
                if param not in baked and (dtype := _scalar_declaration(helper, param)) is not None
            }
            meta, _ = _tensor_meta.resolve_dep_call_metadata(
                cast("Any", adapter),
                caller_func,
                dict(locals_meta),
                scalar_hints,
                {},
                caller_func_type,
                dep_call_name=call_name,
                call_node=call_node,
                caller_constexpr=caller_constexpr,
                discover_deps=discover_deps,
                traced_inline=resolvers,
                traced_inline_for=self.resolver_factory(discover_deps),
            )
            scope = _tensor_meta._StaticScope.of(caller_func, caller_constexpr)
            call_args = _call_arg_nodes(list(helper.param_names), call_node, call_name)
            from pypto.language.tracing.decorators.jit_trace import _variant_key_part  # noqa: PLC0415

            annotations: dict[str, Any] = {}
            key_parts: list[Any] = [id(helper)]
            for param in helper.param_names:
                if param in baked:
                    value, _text = _baked_value(call_name, param, call_args[param], scope, caller_constexpr)
                    annotations[param] = value
                    key_parts.append(("const", constant_source(value)))
                    continue
                facade = _walk_facade(helper, param, meta.get(param), scalar_hints.get(param), call_name)
                _check_declared(call_name, param, facade, hints)
                annotations[param] = facade
                key_parts.append(_variant_key_part(facade))
        except Exception:  # noqa: BLE001 - provisional only: the parse-time resolution is authoritative
            return None
        key = tuple(key_parts)
        if key in self._provisional:
            return self._provisional[key]
        function = self._trace(call_name, helper, annotations, self._scratch_assembly())
        answer = _return_metas(function, meta, call_name)
        self._provisional[key] = answer
        return answer

    def return_targets(
        self,
        binding: TracedInlineBinding,
        call_node: ast.Call,
        targets: Sequence[ast.expr],
        caller_func: Any,
        caller_func_type: str,
        locals_meta: Mapping[str, TensorMeta],
        caller_constexpr: Mapping[str, str],
        *,
        discover_deps: _call_binding.DiscoverDeps,
        resolvers: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        """The caller's assignment targets, typed from the helper's results.

        One entry per *statement* target, because the metadata walk applies the
        answer positionally against ``stmt.targets``: a tuple target
        (``qa, qb = helper(...)``) is one entry carrying both names, exactly as
        the parser-dependency rules already do. A result the helper's return
        list does not cover — or a provisional answer that declined — leaves
        whatever the caller knew about that name standing rather than dropping
        it.
        """
        declined = _tensor_meta._DEP_RETURN_DECLINED
        answer = self.return_meta(
            binding,
            call_node,
            caller_func,
            caller_func_type,
            locals_meta,
            caller_constexpr,
            discover_deps=discover_deps,
            resolvers=resolvers,
        )
        if answer is None:
            return [declined for _ in targets]
        metas, _types = answer
        answers: list[Any] = []
        cursor = 0
        for target in targets:
            elements = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
            payload: dict[str, TensorMeta] = {}
            for element in elements:
                meta = metas[cursor] if metas is not None and cursor < len(metas) else None
                cursor += 1
                if isinstance(element, ast.Name) and meta is not None:
                    payload[element.id] = meta
            answers.append(declined if not payload else _tensor_meta._DepReturn(payload, frozenset()))
        return answers


def _check_declared(call_name: str, param: str, facade: Any, hints: Mapping[str, Any]) -> None:
    """Hold a staged call-site type to the helper's own declaration.

    The parser lane re-declares a dependency from its call site; a traced helper
    is not re-declared. Refusing here keeps a caller from silently changing the
    helper's ABI, and it is the tracer's own rule — the same check its callee
    lane runs.
    """
    from pypto.language.tracing.argument_contract import check_declared_argument  # noqa: PLC0415

    check_declared_argument(param, facade, hints.get(param), call_name)


def _return_metas(
    function: ir.Function, args_meta: Mapping[str, TensorMeta], call_name: str
) -> tuple[list[TensorMeta] | None, list[ir.Type]]:
    """The helper's results as caller-visible metadata, or ``None`` if not tensors.

    A returned tensor dimension that is a symbol is the caller's own dynamic
    extent whenever the symbol came from an argument (the common ``return x``
    and ``create_tensor`` shapes); otherwise it is a fresh runtime extent and is
    recorded as a synthesized dimension, which is what the parser lane does for
    any extent it cannot resolve statically.
    """
    bounds: dict[str, DynDim] = {}
    for meta in args_meta.values():
        for dim in meta.shape:
            if isinstance(dim, DynDim):
                bounds.setdefault(dim.name, dim)
    metas: list[TensorMeta] = []
    for index, ty in enumerate(function.return_types):
        if not isinstance(ty, ir.TensorType):
            return None, list(function.return_types)
        shape: list[Any] = []
        for axis, dim in enumerate(ty.shape):
            # A static extent is ``ir.ConstInt`` in IR shape, not a Python int:
            # reading only ``int`` turned every returned ``[64, 64]`` into two
            # synthesized dynamic symbols, which then contradicted any callee
            # that declares the same static shape.
            literal = dim.value if isinstance(dim, ir.ConstInt) else (dim if isinstance(dim, int) else None)
            if literal is not None:
                shape.append(literal)
                continue
            name = getattr(dim, "name_hint", None)
            known = bounds.get(name) if isinstance(name, str) else None
            if known is not None:
                shape.append(known)
            elif isinstance(name, str):
                shape.append(DynDim(name=name, literal=name, static_bound=1))
            else:
                shape.append(_tensor_meta.synthesized_dyn_dim(function.name, f"ret{index}", axis))
        view = getattr(ty, "tensor_view", None)
        metas.append(TensorMeta(shape=tuple(shape), dtype=ty.dtype, layout=getattr(view, "layout", None)))
    return metas, list(function.return_types)


def _reject_non_inline_reachable(reached: Sequence[ir.Function], call_name: str) -> None:
    """Refuse a traced helper whose collected closure leaves the inline lane.

    The bridge imports Inline functions so the program has nothing but inline
    calls to eliminate. A helper that reaches an ``incore``/``opaque``/``graph``
    callee would put a traced ABI function into a parser program — the
    unsupported direction — so it is named here, at the helper that reached it.
    """
    for function in reached:
        if function.func_type == ir.FunctionType.Inline:
            continue
        raise TypeError(
            f"traced inline helper {call_name!r} reaches {function.func_type.name} "
            f"function {function.name!r}. Composing a parser caller with a traced "
            "helper requires every import in the helper's own closure to be inline: "
            "migrate that callee to @pl.jit_trace.inline too, or keep this helper on "
            "the parsing frontend."
        )


def _source_identity(func: Any) -> tuple[Any, ...]:
    """A stable identity for one traced helper's source and body language.

    Read for the caller's cache key only, so it must be deterministic across
    processes: the defining module and qualified name, the function's first
    line, and its own source text. A helper whose body changes while the file is
    edited produces a different record even when the caller's source is
    untouched.
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = ""
    return (
        getattr(func, "__module__", ""),
        getattr(func, "__qualname__", ""),
        getattr(getattr(func, "__code__", None), "co_firstlineno", -1),
        source,
    )


def _declared_kind(value: Any) -> str:
    """A configuration enum's stable spelling (``None`` stays ``"None"``)."""
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else ("None" if value is None else str(value))


def _config_identity(value: Any) -> Any:
    """A deterministic, JSON-safe identity for one helper configuration value.

    ``attrs`` is user-supplied, so its values are normalized explicitly —
    scalars as themselves, enums by module/qualname/name, sequences and
    string-keyed mappings recursively. Anything else is refused rather than
    stringified: a ``repr`` or a bare ``str`` would let two different
    configurations share one cache entry, and this value only ever feeds the
    caller's identity.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, (list, tuple)):
        items = cast("Sequence[object]", value)
        return tuple(_config_identity(item) for item in items)
    if isinstance(value, Mapping):
        pairs: list[tuple[str, Any]] = []
        for key, item in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                raise TypeError(f"helper attrs must be string-keyed, got {key!r}")
            pairs.append((key, _config_identity(item)))
        return tuple(sorted(pairs))
    if isinstance(getattr(value, "name", None), str):
        return ("enum", type(value).__module__, type(value).__qualname__, value.name)
    raise TypeError(
        f"traced inline helper attribute of type {type(value).__qualname__} has no "
        "cache identity; use a scalar, an enum, or a list/mapping of them"
    )


def _helper_identity(helper: Any) -> tuple[Any, ...]:
    """A stable identity for one traced helper instantiation.

    Source alone does not identify a helper: two instances produced by one
    factory share their module, qualname, first line, *and* source text, and
    differ only in the closure values they capture (``factory(2.0)`` vs
    ``factory(3.0)``). Those values are exactly what ``free_name_source``
    renders for the constants the body folds, so they are part of the identity.

    The helper's own materialization configuration is part of it too. The same
    kernel decorated ``auto_scope=False`` and ``auto_scope=True`` compiles to
    different IR, and a cache key that ignored the difference would keep serving
    the first decoration's artifact.
    """
    kernel = helper.kernel
    namespace = func_name_lookup(kernel)
    folded = tuple(
        (name, text)
        for name in _ast_analysis.constant_dependency_names(kernel)
        if (text := free_name_source(name, namespace)) is not None
    )
    config = (
        _declared_kind(getattr(helper, "_function_type", None)),
        _declared_kind(getattr(helper, "_level", None)),
        _declared_kind(getattr(helper, "_role", None)),
        _config_identity(dict(getattr(helper, "_attrs", None) or {})),
    )
    return ("traced-inline", *_source_identity(kernel), folded, config)


def _nested_baked_records(caller_func: Any, binding: TracedInlineBinding) -> list[tuple[Any, ...]]:
    """Identity records for the compile-time values a traced helper's body bakes.

    A helper body reaches another traced helper through the tracer's own callee
    lane, so the value it bakes at that call is invisible to the plan-level
    records (``baked_call_identity`` walks the parser functions only) and to the
    helper's own record (a free *name* folds only when the source renders it —
    an attribute chain like ``cfg.KNOB`` renders nothing). Folding the nested
    call sites from the helper's own source mirrors the plan-level mechanism,
    which is what keeps the caller's key from standing still while the compiled
    body moves.

    An argument that folds to no compile-time value — one bound to the helper's
    own staged or baked parameter — contributes nothing here: its value
    arrives from the caller's call site, where the plan-level records or an
    outer layer of this same recursion already carry it.
    """
    baked = _baked_names(binding.helper)
    if not baked:
        return []
    scope = _tensor_meta._StaticScope.of(caller_func, {})
    records: list[tuple[Any, ...]] = []
    for ordinal, node in enumerate(_call_binding._dep_call_nodes(caller_func, binding.call_name)):
        args = _call_arg_nodes(list(binding.helper.param_names), node, binding.call_name)
        for param in sorted(baked):
            folded = _fold_baked(args[param], scope, {})
            if folded is None:
                continue
            records.append(
                (
                    "traced-inline-baked",
                    *_source_identity(caller_func)[:3],
                    binding.call_name,
                    ordinal,
                    param,
                    folded[1],
                )
            )
    return records


def traced_source_records(functions: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    """Identity records for every traced helper ``functions`` import.

    The caller's cache key is composed *before* the program is built, so this
    walk has to answer "what do the imported helpers contribute?" without
    tracing anything: each helper's own source, and the module-level or closure
    constants its body folds in. Both are exactly what the parser lane records
    for one of its own functions, so a traced helper is identified by the same
    evidence a parser dependency is.

    A helper record alone is not enough — the *wiring* is part of the compiled
    program too. Two call names can reach two different factory instances, and
    rebinding one name to the other changes which helper a call site reaches
    while leaving the set of reached helpers, and therefore the record list,
    untouched. So each caller also contributes one record per binding it
    resolves, naming the caller, the call name, and the helper it reaches.

    Traversal follows bare-name calls into other traced helpers, which is the
    same rule that finds them at a call site. A helper reached only through a
    non-callable indirection is invisible here — the same boundary the parser
    lane's dependency discovery has. A nested helper body also contributes the
    compile-time values its own traced calls bake
    (:func:`_nested_baked_records`), because the trace bakes them from the
    helper's namespace at materialization time and no other record carries
    them.
    """
    records: list[tuple[Any, ...]] = []
    seen_helpers: set[tuple[Any, ...]] = set()
    seen_callers: set[int] = set()

    def visit(func: Any, *, nested: bool = False) -> None:
        # A helper that calls itself (or a cycle among helpers) must terminate.
        if id(func) in seen_callers:
            return
        seen_callers.add(id(func))
        caller = _source_identity(func)[:3]
        for binding in discover_traced_inline(func):
            helper = binding.helper
            identity = _helper_identity(helper)
            records.append(("traced-inline-binding", *caller, binding.call_name, *identity))
            if nested:
                records.extend(_nested_baked_records(func, binding))
            if identity not in seen_helpers:
                seen_helpers.add(identity)
                records.append(identity)
            visit(helper.kernel, nested=True)

    for func in functions:
        visit(func)
    return tuple(records)


def call_sites_by_name(
    bindings: Sequence[TracedInlineBinding], caller_func: Any
) -> dict[str, list[ast.Call]]:
    """Every call node of each traced-inline binding, in source order.

    The ordinal is the call's position in this list, which is how the generated
    source and the specializer agree on *which* call site a rewritten call
    names: both walk the same dedented source.
    """
    return {
        binding.call_name: _call_binding._dep_call_nodes(caller_func, binding.call_name)
        for binding in bindings
    }
