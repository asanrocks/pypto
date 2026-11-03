# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Read-only source analysis shared by JIT dependency and metadata resolution.

Cached ASTs describe function source, which is *not* immutable in one way that
matters: a module file can be edited while a function object from it survives
(the file is fixed, the object is not reloaded). Every cache below is therefore
keyed on the parsed tree — a fresh parse produces a fresh key — and the parse
itself is memoised on the source text. Mutable name bindings are always
resolved through the per-request namespace snapshot, never cached in an AST."""

import ast
import functools
import inspect
import textwrap
import weakref
from collections.abc import MutableMapping
from typing import Any

from .specializer import (
    DynDim,
    _collect_annotation_dynamic_dims,
    _collect_dynvar_names,
)

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def get_func_def(func: Any) -> ast.FunctionDef:
    """Parse func source and return its FunctionDef node.

    ``inspect.getsource`` reads the file live, so the parse is revalidated
    against it on every call: a body edited while its function object survives
    (a module edited between two compiles) must not keep serving the stale
    tree, or the discovery and local-name scans below would silently disagree
    with every other reader of the same source — including the identity
    records the cache key hashes. The expensive part, ``ast.parse``, stays memoised on the
    source text. The returned node is shared across callers, so callers MUST
    treat it as read-only — every in-tree consumer only walks/reads it.

    Raises:
        OSError: If the source code cannot be retrieved (e.g. interactive REPL,
            Jupyter notebook, or exec/eval-generated functions).
    """
    try:
        src = textwrap.dedent(inspect.getsource(func))
    except OSError as e:
        raise OSError(
            f"@pl.jit cannot retrieve source code for '{func.__name__}'. "
            "Source code must be available on disk. "
            "Interactive shells, Jupyter notebooks, and exec/eval-generated "
            f"functions are not supported. (Original error: {e})"
        ) from e
    try:
        cached = _func_defs.get(func)
    except TypeError:
        # Not weak-referenceable (not a plain function object): parse without
        # the per-function cache; the text-keyed parse memo still applies.
        cached = None
    if cached is not None and cached[0] == src:
        return cached[1]
    func_def = _parse_func_def(src, func.__name__)
    try:
        _func_defs[func] = (src, func_def)
    except TypeError:
        pass
    return func_def


# Parsed bodies by function object, paired with the source they were parsed
# from so an edit under a live object is detected on the next call. Weak keys:
# the cache must not keep functions alive.
_func_defs: MutableMapping[Any, tuple[str, ast.FunctionDef]] = weakref.WeakKeyDictionary()


@functools.lru_cache(maxsize=512)
def _parse_func_def(source: str, name: str) -> ast.FunctionDef:
    """Parse ``source`` and locate the ``name`` FunctionDef (parse memo)."""
    tree = ast.parse(source)
    func_def = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )
    if func_def is None:
        raise OSError(
            f"@pl.jit could not locate function definition '{name}' "
            "in its own source file. This may happen with heavily wrapped functions."
        )
    return func_def


@functools.lru_cache(maxsize=512)
def collect_all_called_names(func_def: ast.FunctionDef) -> tuple[str, ...]:
    """Cache call names from the immutable AST; resolve their bindings per call."""
    names: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in seen:
                names.append(name)
                seen.add(name)
    return tuple(names)


def _collect_bind_dynamic_bindings(
    func_def: ast.FunctionDef,
    param_names: set[str],
) -> dict[tuple[str, int], str]:
    """Scan ``param.bind_dynamic(dim, dynvar_var)`` calls.

    Returns ``(param_name, dim_idx) → dynvar_variable_name``.
    """
    result: dict[tuple[str, int], str] = {}
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        if not (
            isinstance(fn, ast.Attribute)
            and fn.attr == "bind_dynamic"
            and isinstance(fn.value, ast.Name)
            and fn.value.id in param_names
        ):
            continue
        if len(call.args) < 2:
            continue
        dim_node, dv_node = call.args[0], call.args[1]
        if (
            isinstance(dim_node, ast.Constant)
            and isinstance(dim_node.value, int)
            and isinstance(dv_node, ast.Name)
        ):
            result[(fn.value.id, dim_node.value)] = dv_node.id
    return result


def build_dyndim_map_for_func(
    func: Any,
    param_names: tuple[str, ...],
) -> dict[str, dict[int, DynDim]]:
    """For each tensor param of ``func``, return ``dim_idx → DynDim``.

    Unions the three sources of "this dim is dynamic" declarations:

    1. ``param.bind_dynamic(dim, dynvar_var)`` — gives both ``(param, dim)``
       and the dynvar Python variable name.
    2. Annotation-embedded ``pl.dynamic()`` (``pl.Tensor[[M, …], …]``) —
       gives the same info; bind_dynamic takes precedence on overlap.
    3. ``M = pl.dynamic("M_literal")`` body assignment — maps each dynvar
       variable name to its ``pl.dynamic()`` string literal (these usually
       match but can differ, e.g. ``rows = pl.dynamic("M")``).

    ``DynDim.static_bound`` is filled with ``0`` here as a placeholder; the
    real per-call extent is injected by ``extract_tensor_meta`` from the
    actual ``torch.Tensor`` argument.
    """
    func_def = get_func_def(func)
    return _build_dyndim_map(func_def, func, tuple(param_names))


@functools.lru_cache(maxsize=512)
def _build_dyndim_map(
    func_def: ast.FunctionDef,
    func: Any,
    param_names: tuple[str, ...],
) -> dict[str, dict[int, DynDim]]:
    """The body of :func:`build_dyndim_map_for_func`, keyed on the parsed tree.

    Keying on the node — not the function object — is what makes the map
    follow a source edit: a fresh parse is a fresh cache key, while the
    signature-derived annotation half stays memoised on the function (a live
    function object's annotations cannot change under it).
    """
    pset = set(param_names)
    bd_bindings = _collect_bind_dynamic_bindings(func_def, pset)
    _, ann_bindings, ann_literals = _collect_annotation_dynamic_dims(func, pset)
    dyn_literals = _collect_dynvar_names(func_def)

    out: dict[str, dict[int, DynDim]] = {}

    def _literal_for(dv_name: str) -> str:
        return dyn_literals.get(dv_name) or ann_literals.get(dv_name, dv_name)

    # bind_dynamic source (authoritative when present)
    for (p, i), dv_name in bd_bindings.items():
        out.setdefault(p, {})[i] = DynDim(name=dv_name, literal=_literal_for(dv_name), static_bound=0)

    # annotation source fills dims not covered by bind_dynamic
    for key, dv_name in ann_bindings.items():
        p, idx_str = key.rsplit("__", 1)
        i = int(idx_str)
        per_param = out.setdefault(p, {})
        if i not in per_param:
            per_param[i] = DynDim(name=dv_name, literal=_literal_for(dv_name), static_bound=0)
    return out


def scan_dynamic_dims(func: Any, param_names: list[str]) -> set[tuple[str, int]]:
    """Return dynamic ``(param, dim)`` pairs declared in ``func`` (union of all sources)."""
    dyn_map = build_dyndim_map_for_func(func, tuple(param_names))
    return {(p, i) for p, dims in dyn_map.items() for i in dims}


def body_local_names(func: Any) -> set[str]:
    """Names a body binds locally: its parameters and every name its statements bind.

    A local shadows a same-named module or closure binding, so anything that
    folds a free name must consult this first. The specializer applies the same
    rule through ``BodyTransformer._used_names``; keeping one definition here
    is what stops a folded value and the emitted source from disagreeing.
    """
    return set(_body_local_names(get_func_def(func)))


@functools.lru_cache(maxsize=512)
def _body_local_names(func_def: ast.FunctionDef) -> frozenset[str]:
    """Every name ``func_def``'s body binds locally, keyed on the parsed tree.

    Parameters plus every store-context ``Name`` (assignment targets,
    comprehension binders, loop variables, walrus targets), and the binding
    forms that carry no ``Name`` node of their own: nested ``def``/``class``
    statements, ``import ... as``, and ``except ... as``. All of them shadow a
    same-named module or closure binding exactly the same way, so a resolver
    that ignores one can resolve a call the function's own Python never makes.
    """
    local_names = {arg.arg for arg in ast.walk(func_def.args) if isinstance(arg, ast.arg)}
    for node in ast.walk(func_def):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local_names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not func_def:
            # The tree's root is the function under analysis; only definitions
            # *nested in its body* bind a local name.
            local_names.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            local_names.add(node.name)
        elif isinstance(node, ast.alias):
            local_names.add(node.asname or node.name.split(".")[0])
    return frozenset(local_names)


def constant_dependency_names(func: Any) -> tuple[str, ...]:
    """Find names that can supply folded constants, excluding body locals.

    Match the specializer's parameter/Store-target shadowing rules. Annotation
    names resolve in the defining namespace independently of body locals.
    Decorators and defaults are already evaluated when the function is defined.
    """
    definition = get_func_def(func)
    return _constant_dependency_names(definition, _body_local_names(definition))


@functools.lru_cache(maxsize=512)
def _constant_dependency_names(func_def: ast.FunctionDef, local_names: frozenset[str]) -> tuple[str, ...]:
    names = {
        node.id
        for statement in func_def.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in local_names
    }
    annotations = [arg.annotation for arg in ast.walk(func_def.args) if isinstance(arg, ast.arg)]
    annotations.append(func_def.returns)
    for annotation in annotations:
        if annotation is None:
            continue
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                annotation = ast.parse(annotation.value, mode="eval")
            except SyntaxError:
                continue
        names.update(node.id for node in ast.walk(annotation) if isinstance(node, ast.Name))
    return tuple(sorted(names))
