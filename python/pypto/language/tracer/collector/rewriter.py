# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type-blind stdlib-``ast`` source rewriter for instrumented kernels.

The instrumentor: it transforms kernel syntax into calls on the trace
interpreter without inspecting value types. Local reads and writes are
redirected through ``rt.vars``; the trace interpreter and Sema determine
whether the resulting values and control-flow sites follow host execution or
staged execution.

History: this module replaced an earlier libcst-based implementation. The
stdlib tree carries statement structure, native line positions, and
expression routing directly, and the emitted source goes straight back to
``ast.parse`` + ``compile`` — nothing downstream consumes libcst's
whitespace-preserving round-trip, which cost ~7x per kernel. The two
implementations were held to emitted-source and behavioral parity by a
differential oracle over the snippet corpus before the libcst one was
retired; the corpus itself remains as the construct contract
(``test_rewriter_snippets.py``).

The walker operates on tree nodes rather than source text, so string
contents cannot be rewritten accidentally. Source comments — dropped by
``ast.parse`` — are recovered from the token stream (one pass yields both
the comment table and the header-colon lines the tree does not record) and
attached to the logical statement that owns them as explicit metadata
events. Statement and expression rewriters preserve native evaluation
order, short-circuiting, grouping, and exception behavior while adding the
interpreter operations needed for collection.

Names bound in the kernel function become instrumented locals. Binders
belonging to nested functions, classes, lambdas, and comprehensions remain
native to those scopes, while reads of outer kernel locals continue to
redirect. Nested definitions and classes are otherwise preserved as
host-language constructs and synchronize their bindings into the
instrumented locals.

Native ``break``/``continue`` is the one construct that changes the shape of
enclosing instrumented constructs: a native transfer skips a plain trailing
``exit_construct()`` call, so every construct whose body holds a transfer
keyword bound to an enclosing loop is wrapped in the unwind scope guard
``with rt.unwind_guard():`` instead. The keyword's own unwind runs the guard
exit at the construct's exact lexical boundary — after inner source context
managers and ``finally`` bodies — so active-construct cleanup preserves
Python's unwind order. ``with`` constructs need no guard: their generated
context manager already runs during native unwinding.
"""

from __future__ import annotations

import ast
import bisect
import io
import tokenize
from collections.abc import Sequence
from typing import TypeAlias, cast

_AnyStmt: TypeAlias = ast.stmt
_AnyPattern: TypeAlias = ast.pattern

_SIMPLE_STMT_TYPES = (
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Expr,
    ast.Return,
    ast.Delete,
    ast.Raise,
    ast.Assert,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
)
_COMPOUND_STMT_TYPES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


# ---------------------------------------------------------------- errors ---


class RewriteError(SyntaxError):
    """A span-pointed rejection of a construct outside the rewrite subset.

    The span is filled from the node's native source position when the input
    node came from parsed source.
    """

    def __init__(self, message: str, pos: tuple[int, int] | None = None):
        super().__init__(message)
        self.message = message
        self.pos = pos
        if pos is not None:
            self.lineno = pos[0]  # source lines are 1-based
            self.offset = pos[1] + 1  # columns are 0-based → SyntaxError 1-based


def choose_prefix(names: frozenset[str]) -> str:
    """Choose the synthesized ``rt`` interpreter handle's name so it cannot collide with a kernel name."""
    if "rt" not in names:
        return "rt"
    i = 0
    while f"rt_{i}" in names:
        i += 1
    return f"rt_{i}"


# ----------------------------------------------------- construct model ---


class _Ctx:
    """Per-rewrite state for names, scopes, generated names, and policy.

    ``policy`` enables rewrite-subset checks in the kernel body and disables
    them inside nested helper definitions, which execute as host plumbing.
    ``comments`` carries the source's line-indexed comments for statement
    attachment.
    """

    __slots__ = (
        "prefix",
        "instrumented",
        "shadow",
        "names",
        "_temp",
        "policy",
        "comments",
        "future_annotations",
        "line_offset",
        "_escapes_memo",
    )

    def __init__(
        self,
        prefix: str,
        instrumented: frozenset[str],
        names: frozenset[str],
        policy: bool = True,
        comments: _Comments | None = None,
        future_annotations: bool = False,
        line_offset: int = 0,
    ):
        self.prefix = prefix
        self.instrumented = instrumented
        self.shadow: frozenset[str] = frozenset()
        self.names = names  # every name in the kernel (temp/prefix avoidance)
        self._temp = 0
        self.policy = policy
        self.comments = comments
        self.future_annotations = future_annotations
        # The dedented source's line 1 is the caller's ``line_offset + 1``:
        # span events carry the ORIGINAL file line.
        self.line_offset = line_offset
        # Per-pass memo for the transfer-escape walk (see _escapes_transfer):
        # the rewrite consults the same subtrees many times, so each
        # (node, kinds, depth) answer is computed once.
        self._escapes_memo: dict[tuple[int, int, int], bool] = {}

    def with_shadow(self, extra: frozenset[str]) -> _Ctx:
        child = _Ctx(
            self.prefix,
            self.instrumented,
            self.names,
            self.policy,
            self.comments,
            self.future_annotations,
            self.line_offset,
        )
        child.shadow = self.shadow | extra
        return child

    def span(self, node: ast.AST) -> tuple[int, int] | None:
        """The node's (line, column) — 1-based line, 0-based column, native."""
        line = getattr(node, "lineno", None)
        col = getattr(node, "col_offset", None)
        if line is not None and col is not None:
            return line, col
        return None

    def fresh_temp(self, hint: str) -> str:
        while True:
            name = f"__pto_{hint}{self._temp}"
            self._temp += 1
            if name not in self.names:
                return name

    def end_line(self, node: ast.AST) -> int | None:
        """The node's end line, or None when unavailable (synthetic nodes)."""
        return getattr(node, "end_lineno", None)

    def drain_comments(self, lo: int, hi: int) -> list[str]:
        """Comments on source lines ``(lo, hi]``, in line order."""
        if self.comments is None:
            return []
        return self.comments.drain(lo, hi)

    def trailing_comment(self, line: int, col: int) -> list[str]:
        """The statement's trailing comment on its own end line."""
        if self.comments is None:
            return []
        return self.comments.trailing_after(line, col)


class _Comments:
    """Line- and column-indexed source comments plus header-colon lines.

    The ast tree drops comments and does not record where a compound
    statement's header (its ``:``) ends; both are recovered from the token
    stream once. Depth-0 colons are the header colons — annotation, walrus,
    and lambda colons inside brackets or headers resolve through the
    paren-depth filter (a bare walrus in an ``if``/``while`` test shares the
    header's first line, so the first-colon rule stays correct there).
    """

    __slots__ = ("_by_line", "_colons")

    def __init__(self, source: str):
        self._by_line, self._colons = _tokenize_source(source)

    def drain(self, lo: int, hi: int) -> list[str]:
        """Comment texts on lines ``(lo, hi]``, in line order."""
        out: list[str] = []
        for line in range(lo + 1, hi + 1):
            out.extend(text for _col, text in self._by_line.get(line, ()))
        return out

    def drain_tagged(self, lo: int, hi: int) -> list[tuple[int, str]]:
        """(line, text) pairs for comments on lines ``(lo, hi]``."""
        out: list[tuple[int, str]] = []
        for line in range(lo + 1, hi + 1):
            out.extend((line, text) for _col, text in self._by_line.get(line, ()))
        return out

    def on_line(self, line: int) -> list[str]:
        """Every comment on one line (a header line holds at most one, after
        the colon)."""
        return [text for _col, text in self._by_line.get(line, ())]

    def trailing_after(self, line: int, col: int) -> list[str]:
        """Comments on ``line`` starting at or after column ``col`` — the
        statement's own trailing comment, never one inside its parentheses."""
        return [text for c, text in self._by_line.get(line, ()) if c >= col]

    def first_colon_at_or_after(self, line: int) -> int | None:
        """The first depth-0 header-colon line at or after ``line``."""
        i = bisect.bisect_left(self._colons, line)
        return self._colons[i] if i < len(self._colons) else None

    def first_colon_after(self, line: int) -> int | None:
        """The first depth-0 header-colon line strictly after ``line``."""
        i = bisect.bisect_right(self._colons, line)
        return self._colons[i] if i < len(self._colons) else None

    def last_colon_in(self, start: int, end: int) -> int | None:
        """The last depth-0 header-colon line in ``[start, end)``."""
        i = bisect.bisect_left(self._colons, end) - 1
        if i >= 0 and self._colons[i] >= start:
            return self._colons[i]
        return None


def _tokenize_source(source: str) -> tuple[dict[int, list[tuple[int, str]]], list[int]]:
    """(comments by line as (col, text), sorted depth-0 colon lines) — or
    empties on malformed source (the caller's ast.parse surfaces the real
    error)."""
    comments: dict[int, list[tuple[int, str]]] = {}
    colons: set[int] = set()
    depth = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.OP:
                if tok.string in "([{":
                    depth += 1
                elif tok.string in ")]}":
                    depth -= 1
                elif tok.string == ":" and depth == 0:
                    colons.add(tok.start[0])
            elif tok.type == tokenize.COMMENT:
                text = tok.string[1:] if tok.string.startswith("#") else tok.string
                if text.startswith(" "):
                    text = text[1:]
                comments.setdefault(tok.start[0], []).append((tok.start[1], text))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return comments, sorted(colons)
    return comments, sorted(colons)


def _names_in_target(target: ast.expr) -> frozenset[str]:
    # Accepted target: ``a, *rest``; collected names: ``a``, ``rest``.
    names: set[str] = set()
    stack: list[ast.expr] = [target]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            stack.extend(node.elts)
        elif isinstance(node, ast.Starred):
            stack.append(node.value)
    return frozenset(names)


def _param_names(args: ast.arguments) -> list[str]:
    # Accepted signature: ``def f(a, /, b, *args, c, **kwargs)``.
    names: list[str] = [p.arg for p in args.posonlyargs]
    names.extend(p.arg for p in args.args)
    if args.vararg is not None:
        names.append(args.vararg.arg)
    names.extend(p.arg for p in args.kwonlyargs)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _copy_type_params(source: ast.AST, target: ast.AST) -> None:
    """Carry PEP 695 type parameters across a rebuild.

    The stdlib constructor stubs do not expose ``type_params`` (the runtime
    accepts it), so the field is set directly on the rebuilt node; ``None``
    (pre-3.12 sources and the stdlib default) is skipped.
    """
    tp = getattr(source, "type_params", None)
    if tp:
        setattr(target, "type_params", list(tp))


def _empty_args() -> ast.arguments:
    return ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )


def _import_local_name(alias: ast.alias) -> str:
    # Accepted imports: ``import a.b as c`` -> ``c``; ``from a import b`` -> ``b``.
    if alias.asname is not None:
        return alias.asname
    return alias.name.split(".")[0]


def _add_target(target: ast.expr, instrumented: set[str]) -> None:
    # Accepted target: ``a, *rest``; route both names.
    if isinstance(target, ast.Name):
        instrumented.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for el in target.elts:
            _add_target(el, instrumented)
    elif isinstance(target, ast.Starred):
        _add_target(target.value, instrumented)
    # Accepted target: ``a.b`` or ``a[i]``; no local name is bound.


def _collect_expr(e: ast.AST, instrumented: set[str]) -> None:
    """Collect kernel-scope walrus targets from an expression.

    Lambda and comprehension binders are skipped — they belong to those
    scopes, not the kernel. A walrus inside a comprehension binds the
    enclosing kernel scope, so it is collected and later redirected through
    ``rt.bind`` like any kernel-local walrus.
    """
    # Accepted expression: ``(x := value)``; collect ``x`` and recurse into ``value``.
    if isinstance(e, ast.NamedExpr):
        _add_target(e.target, instrumented)
        _collect_expr(e.value, instrumented)
    # Accepted expression: ``lambda x: x``; do not collect its local binder.
    elif isinstance(e, ast.Lambda):
        return
    # Accepted expressions: ``[x for x in values]``, ``(x for x in values)``,
    # ``{x for x in values}``; collect the element and generators.
    elif isinstance(e, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        _collect_expr(e.elt, instrumented)
        _collect_generators(e.generators, instrumented)
    # Accepted expression: ``{k: v for k, v in values}``.
    elif isinstance(e, ast.DictComp):
        _collect_expr(e.key, instrumented)
        _collect_expr(e.value, instrumented)
        _collect_generators(e.generators, instrumented)
    # Other expressions are traversed recursively, for example ``a + b``.
    else:
        for child in ast.iter_child_nodes(e):
            _collect_expr(child, instrumented)


def _collect_generators(generators: Sequence[ast.comprehension], instrumented: set[str]) -> None:
    for gen in generators:
        _collect_expr(gen.iter, instrumented)
        for if_ in gen.ifs:
            _collect_expr(if_, instrumented)


def _collect_pattern(pattern: _AnyPattern, instrumented: set[str]) -> None:
    """Collect match binders and names used by pattern value expressions."""

    for name in _pattern_binders(pattern):
        instrumented.add(name)
    if isinstance(pattern, ast.MatchValue):
        _collect_expr(pattern.value, instrumented)
    elif isinstance(pattern, ast.MatchClass):
        _collect_expr(pattern.cls, instrumented)
        for p in pattern.patterns:
            _collect_pattern(p, instrumented)
        for p in pattern.kwd_patterns:
            _collect_pattern(p, instrumented)
    elif isinstance(pattern, ast.MatchMapping):
        for key in pattern.keys:
            _collect_expr(key, instrumented)
        for p in pattern.patterns:
            _collect_pattern(p, instrumented)
    elif isinstance(pattern, ast.MatchSequence):
        for p in pattern.patterns:
            _collect_pattern(p, instrumented)
    elif isinstance(pattern, ast.MatchOr):
        for p in pattern.patterns:
            _collect_pattern(p, instrumented)
    elif isinstance(pattern, ast.MatchAs) and pattern.pattern is not None:
        _collect_pattern(pattern.pattern, instrumented)


def _pattern_binders(pattern: _AnyPattern) -> list[str]:
    """Return match binders in pattern order.

    The host ``match`` binds these names in the kernel scope. Case bodies
    synchronize them into the instrumented locals, and the bindings remain
    visible after the statement.
    """

    names: list[str] = []
    if isinstance(pattern, ast.MatchAs):
        # both fields are optional and independent — MatchAs can bind a name
        # and a sub-pattern at once; a bare ``_`` is a no-op here
        if pattern.name is not None:
            names.append(pattern.name)
        if pattern.pattern is not None:
            names.extend(_pattern_binders(pattern.pattern))
    elif isinstance(pattern, ast.MatchStar):
        if pattern.name is not None:
            names.append(pattern.name)
    elif isinstance(pattern, ast.MatchSequence):
        for p in pattern.patterns:
            names.extend(_pattern_binders(p))
    elif isinstance(pattern, ast.MatchOr):
        for p in pattern.patterns:
            names.extend(_pattern_binders(p))
    elif isinstance(pattern, ast.MatchMapping):
        for p in pattern.patterns:
            names.extend(_pattern_binders(p))
        if pattern.rest is not None:
            names.append(pattern.rest)
    elif isinstance(pattern, ast.MatchClass):
        for p in pattern.patterns:
            names.extend(_pattern_binders(p))
        for p in pattern.kwd_patterns:
            names.extend(_pattern_binders(p))
    return names


def _collect_stmt(stmt: _AnyStmt, instrumented: set[str]) -> None:  # noqa: PLR0912 - one branch per construct
    # Accepted statements include ``x = value`` and ``x += value``.
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            _add_target(t, instrumented)
        _collect_expr(stmt.value, instrumented)
    elif isinstance(stmt, ast.AugAssign):
        _add_target(stmt.target, instrumented)
        _collect_expr(stmt.value, instrumented)
    # Accepted annotation assignment: ``x: Type = value``.
    elif isinstance(stmt, ast.AnnAssign):
        _add_target(stmt.target, instrumented)
        if stmt.value is not None:
            _collect_expr(stmt.value, instrumented)
    # Accepted loops: ``for x in values: ...`` (sync and async alike).
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        _add_target(stmt.target, instrumented)
        _collect_expr(stmt.iter, instrumented)
        _collect(stmt.body, instrumented)
        _collect(stmt.orelse, instrumented)
    # Accepted loop: ``while predicate: ...``.
    elif isinstance(stmt, ast.While):
        _collect_expr(stmt.test, instrumented)
        _collect(stmt.body, instrumented)
        _collect(stmt.orelse, instrumented)
    # Accepted context managers: ``with manager as value: ...`` (sync/async).
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            _collect_expr(item.context_expr, instrumented)
            if item.optional_vars is not None:
                _add_target(item.optional_vars, instrumented)
        _collect(stmt.body, instrumented)
    # Accepted pattern statement: ``match value: case pattern: ...``.
    elif isinstance(stmt, ast.Match):
        _collect_expr(stmt.subject, instrumented)
        for case in stmt.cases:
            _collect_pattern(case.pattern, instrumented)
            if case.guard is not None:
                _collect_expr(case.guard, instrumented)
            _collect(case.body, instrumented)
    # Accepted conditional: ``if predicate: ...`` (elif chains nest).
    elif isinstance(stmt, ast.If):
        _collect_expr(stmt.test, instrumented)
        _collect(stmt.body, instrumented)
        _collect(stmt.orelse, instrumented)
    # Accepted protected block: ``try: ... finally: ...``.
    elif isinstance(stmt, ast.Try):
        _collect(stmt.body, instrumented)
        for h in stmt.handlers:
            _collect(h.body, instrumented)
        _collect(stmt.orelse, instrumented)
        _collect(stmt.finalbody, instrumented)
    # Accepted native binders: ``def helper(): ...`` and ``class Helper: ...``.
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        instrumented.add(stmt.name)
    # Accepted import: ``import module as name``.
    elif isinstance(stmt, ast.Import):
        for alias in stmt.names:
            instrumented.add(_import_local_name(alias))
    # Accepted import: ``from module import name``.
    elif isinstance(stmt, ast.ImportFrom):
        if any(alias.name == "*" for alias in stmt.names):
            raise RewriteError("wildcard imports are not supported in collector kernels")
        for alias in stmt.names:
            instrumented.add(_import_local_name(alias))
    # Accepted expression statement: ``use(value)`` or ``(x := value)``.
    elif isinstance(stmt, ast.Expr):
        _collect_expr(stmt.value, instrumented)
    # Accepted return: ``return value``.
    elif isinstance(stmt, ast.Return):
        if stmt.value is not None:
            _collect_expr(stmt.value, instrumented)


def _collect(stmts: Sequence[_AnyStmt], instrumented: set[str]) -> None:
    for stmt in stmts:
        _collect_stmt(stmt, instrumented)


def _all_names(  # noqa: PLR0912 - one branch per named-identifier site
    fn_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Every identifier the source spells — the pool the event prefix and
    internal temps must avoid.

    The stdlib tree carries most identifiers as plain strings (param names,
    definition names, attribute names, call keywords, import components,
    exception and match binders) rather than Name nodes, so each spelling
    site is collected explicitly.
    """
    names: set[str] = {fn_def.name}
    for node in ast.walk(fn_def):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword):
            if node.arg is not None:
                names.add(node.arg)
        elif isinstance(node, ast.alias):
            if node.asname is not None:
                names.add(node.asname)
            else:
                names.update(node.name.split("."))
        elif isinstance(node, ast.ExceptHandler):
            if node.name is not None:
                names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            if node.name is not None:
                names.add(node.name)
        elif isinstance(node, ast.MatchClass):
            names.update(node.kwd_attrs)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # PEP 695 type parameters carry names the stdlib stubs do not
            # expose; collect them through the generic attribute.
            for tp in getattr(node, "type_params", None) or ():
                tp_name = getattr(tp, "name", None)
                if isinstance(tp_name, str):
                    names.add(tp_name)
    return frozenset(names)


def collect_locals(
    fn_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[frozenset[str], frozenset[str]]:
    """(instrumented, names) for the kernel.

    ``instrumented``: params + every kernel-scope binder (assignments, walrus,
    for/with targets, def/class/import names). ``names``: every name in the
    tree — the pool the event prefix and internal temps must avoid.
    """
    instrumented: set[str] = set(_param_names(fn_def.args))
    _collect(fn_def.body, instrumented)
    return frozenset(instrumented), _all_names(fn_def)


# --------------------------------------------------------------- builder ---


def _n(id: str) -> ast.Name:
    return ast.Name(id=id, ctx=ast.Load())


def _n_store(id: str) -> ast.Name:
    return ast.Name(id=id, ctx=ast.Store())


def _call(
    func: ast.expr, args: list[ast.expr] | None = None, keywords: list[ast.keyword] | None = None
) -> ast.Call:
    return ast.Call(func=func, args=args or [], keywords=keywords or [])


def _kw(arg: str, value: ast.expr) -> ast.keyword:
    return ast.keyword(arg=arg, value=value)


def _expr_stmt(value: ast.expr) -> ast.Expr:
    return ast.Expr(value=value)


def _assign(target: ast.expr, value: ast.expr) -> ast.Assign:
    return ast.Assign(targets=[target], value=value)


def _rt_attr(ctx: _Ctx, attr: str) -> ast.Attribute:
    return ast.Attribute(value=_n(ctx.prefix), attr=attr, ctx=ast.Load())


def _rt_vars(ctx: _Ctx) -> ast.Attribute:
    return ast.Attribute(value=_n(ctx.prefix), attr="vars", ctx=ast.Load())


def _route_read(ctx: _Ctx, name: str) -> ast.expr:
    # Input: ``a``; output for an instrumented name: ``rt.vars.a``.
    if name in ctx.instrumented and name not in ctx.shadow:
        return ast.Attribute(value=_rt_vars(ctx), attr=name, ctx=ast.Load())
    # Input: a shadowed/local ``x``; output: ``x``.
    return _n(name)


def _route_target(ctx: _Ctx, target: ast.expr) -> ast.expr:
    """Rewrite an assignment target and route reads inside it.

    Store contexts are set on every rebuilt node — the stdlib tree carries
    the context on each target node.
    """
    # Input: ``x = value``; output target: ``rt.vars.x``.
    if isinstance(target, ast.Name):
        if target.id in ctx.instrumented and target.id not in ctx.shadow:
            return ast.Attribute(value=_rt_vars(ctx), attr=target.id, ctx=ast.Store())
        return _n_store(target.id)
    # Input: ``a, *rest = values``; output: ``rt.vars.a, *rt.vars.rest``.
    if isinstance(target, (ast.Tuple, ast.List)):
        cls = type(target)
        return cls(elts=[_route_target(ctx, el) for el in target.elts], ctx=ast.Store())
    # Input: ``*rest = values``; output: ``*rt.vars.rest``.
    if isinstance(target, ast.Starred):
        return ast.Starred(value=_route_target(ctx, target.value), ctx=ast.Store())
    # Input: ``obj.x = value``; output target: ``rt.vars.obj.x``.
    if isinstance(target, ast.Attribute):
        return ast.Attribute(value=_route_expr(ctx, target.value), attr=target.attr, ctx=ast.Store())
    # Input: ``obj[i] = value``; output target: ``rt.vars.obj[rt.vars.i]``.
    if isinstance(target, ast.Subscript):
        return ast.Subscript(
            value=_route_expr(ctx, target.value), slice=_route_slice(ctx, target.slice), ctx=ast.Store()
        )
    return target


def _rewrite_comp_for(ctx: _Ctx, gen: ast.comprehension) -> ast.comprehension:
    # Input: ``for x in values if x > limit``; output: ``for x in rt.vars.values if x > rt.vars.limit``.
    return ast.comprehension(
        target=gen.target,
        iter=_route_expr(ctx, gen.iter),
        ifs=[_route_expr(ctx, i) for i in gen.ifs],
        is_async=gen.is_async,
    )


def _rewrite_params(ctx: _Ctx, args: ast.arguments) -> ast.arguments:
    """Route reads in defaults and eagerly evaluated annotations.

    Defaults and annotations are evaluated in the enclosing scope. A kernel
    local referenced there is therefore an outer instrumented read, not a local of
    the nested definition or lambda. Future annotations remain unchanged.
    """

    def route_arg(p: ast.arg) -> ast.arg:
        annotation = (
            _route_expr(ctx, p.annotation)
            if p.annotation is not None and not ctx.future_annotations
            else p.annotation
        )
        return ast.arg(arg=p.arg, annotation=annotation, type_comment=p.type_comment)

    return ast.arguments(
        posonlyargs=[route_arg(p) for p in args.posonlyargs],
        args=[route_arg(p) for p in args.args],
        vararg=route_arg(args.vararg) if args.vararg is not None else None,
        kwonlyargs=[route_arg(p) for p in args.kwonlyargs],
        kw_defaults=[_route_expr(ctx, d) if d is not None else None for d in args.kw_defaults],
        kwarg=route_arg(args.kwarg) if args.kwarg is not None else None,
        defaults=[_route_expr(ctx, d) for d in args.defaults],
    )


def _rewrite_decorators(ctx: _Ctx, decorators: Sequence[ast.expr]) -> list[ast.expr]:
    """Route reads in decorators, which evaluate in the enclosing scope."""
    # Input: ``@decorate(a)``; output: ``@decorate(rt.vars.a)``.
    return [_route_expr(ctx, d) for d in decorators]


def _comp_shadow(generators: Sequence[ast.comprehension]) -> frozenset[str]:
    names: set[str] = set()
    for gen in generators:
        names |= _names_in_target(gen.target)
    return frozenset(names)


def _route_expr(ctx: _Ctx, e: ast.expr) -> ast.expr:  # noqa: PLR0912 - one branch per construct
    """Route local reads inside an expression; the construct itself stays."""
    # Input: ``a``; output: ``rt.vars.a`` when ``a`` is instrumented.
    if isinstance(e, ast.Name):
        return _route_read(ctx, e.id)
    # Input: ``a.value``; output: ``rt.vars.a.value``.
    if isinstance(e, ast.Attribute):
        return ast.Attribute(value=_route_expr(ctx, e.value), attr=e.attr, ctx=ast.Load())
    # Input: ``a[i]``; output: ``rt.vars.a[rt.vars.i]``.
    if isinstance(e, ast.Subscript):
        return ast.Subscript(
            value=_route_expr(ctx, e.value), slice=_route_slice(ctx, e.slice), ctx=ast.Load()
        )
    # Input: ``use(a, *args)``; output: ``use(rt.vars.a, *rt.vars.args)``.
    if isinstance(e, ast.Call):
        if ctx.policy:
            _check_forbidden_call(ctx, e)
        return ast.Call(
            func=_route_expr(ctx, e.func),
            args=[_route_expr(ctx, a) for a in e.args],
            keywords=[ast.keyword(arg=k.arg, value=_route_expr(ctx, k.value)) for k in e.keywords],
        )
    # Input: ``(x := a + 1)``; output: ``rt.bind("x", rt.vars.a + 1)``.
    if isinstance(e, ast.NamedExpr):
        # Route a kernel-scope walrus through the same bind path as an assignment.
        name = e.target.id
        if name in ctx.instrumented and name not in ctx.shadow:
            return _call(
                _rt_attr(ctx, "bind"),
                [ast.Constant(value=name), _route_expr(ctx, e.value)],
            )
        return e
    # Input: ``lambda x: x + a``; output: ``lambda x: x + rt.vars.a``.
    if isinstance(e, ast.Lambda):
        shadow = frozenset(_param_names(e.args))
        return ast.Lambda(
            args=_rewrite_params(ctx, e.args), body=_route_expr(ctx.with_shadow(shadow), e.body)
        )
    # Comprehensions: ``[x + a for x in values if p]`` — the first iterable
    # evaluates in the enclosing scope; targets, conditions, inner iterables,
    # and the element see the comprehension's own binders as native.
    if isinstance(e, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        child = ctx.with_shadow(_comp_shadow(e.generators))
        first = ast.comprehension(
            target=e.generators[0].target,
            iter=_route_expr(ctx, e.generators[0].iter),  # first iterable: enclosing scope
            ifs=[_route_expr(child, i) for i in e.generators[0].ifs],
            is_async=e.generators[0].is_async,
        )
        generators = [first] + [_rewrite_comp_for(child, g) for g in e.generators[1:]]
        elt = _route_expr(child, e.elt)
        if isinstance(e, ast.ListComp):
            return ast.ListComp(elt=elt, generators=generators)
        if isinstance(e, ast.SetComp):
            return ast.SetComp(elt=elt, generators=generators)
        return ast.GeneratorExp(elt=elt, generators=generators)
    # Input: ``{x: a for x in values}``; route keys, values, and generators.
    if isinstance(e, ast.DictComp):
        child = ctx.with_shadow(_comp_shadow(e.generators))
        first = ast.comprehension(
            target=e.generators[0].target,
            iter=_route_expr(ctx, e.generators[0].iter),
            ifs=[_route_expr(child, i) for i in e.generators[0].ifs],
            is_async=e.generators[0].is_async,
        )
        generators = [first] + [_rewrite_comp_for(child, g) for g in e.generators[1:]]
        return ast.DictComp(
            key=_route_expr(child, e.key), value=_route_expr(child, e.value), generators=generators
        )
    # Input: ``(a, *values)``; output: ``(rt.vars.a, *rt.vars.values)``.
    # A set literal carries no load/store context in the stdlib tree.
    if isinstance(e, (ast.Tuple, ast.List)):
        cls = type(e)
        return cls(elts=[_route_expr(ctx, el) for el in e.elts], ctx=ast.Load())
    if isinstance(e, ast.Set):
        return ast.Set(elts=[_route_expr(ctx, el) for el in e.elts])
    # Input: ``{a: value, **mapping}``; route keys, values, and mapping.
    if isinstance(e, ast.Dict):
        return ast.Dict(
            keys=[_route_expr(ctx, k) if k is not None else None for k in e.keys],
            values=[_route_expr(ctx, v) for v in e.values],
        )
    # Input: ``*values``; output: ``*rt.vars.values``.
    if isinstance(e, ast.Starred):
        return ast.Starred(value=_route_expr(ctx, e.value), ctx=ast.Load())
    # Input: ``a if predicate else b``; route all three expressions.
    if isinstance(e, ast.IfExp):
        return ast.IfExp(
            body=_route_expr(ctx, e.body), test=_route_expr(ctx, e.test), orelse=_route_expr(ctx, e.orelse)
        )
    # Input: ``a and b`` or ``a or b``; output: a lazy trace-interpreter resolver call.
    # The stdlib tree flattens boolean chains (``a and b and c``); fold from
    # the left so each value's thunk wraps the accumulated left side.
    if isinstance(e, ast.BoolOp):
        op = "eval_and" if isinstance(e.op, ast.And) else "eval_or"
        acc = _route_expr(ctx, e.values[0])
        for value in e.values[1:]:
            # Keep the RHS lazy so host short-circuit evaluation is preserved.
            thunk = ast.Lambda(args=_empty_args(), body=_route_expr(ctx, value))
            acc = _call(_rt_attr(ctx, op), [acc, thunk])
        return acc
    # Input: ``a + b``; output: ``rt.vars.a + rt.vars.b``.
    if isinstance(e, ast.BinOp):
        return type(e)(left=_route_expr(ctx, e.left), op=e.op, right=_route_expr(ctx, e.right))
    # Input: ``not a``; output: ``rt.eval_not(rt.vars.a)``.
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
        return _call(_rt_attr(ctx, "eval_not"), [_route_expr(ctx, e.operand)])
    # Input: ``-a``; output: ``-rt.vars.a``.
    if isinstance(e, ast.UnaryOp):
        return ast.UnaryOp(op=e.op, operand=_route_expr(ctx, e.operand))
    # Input: ``a < b``; output: ``rt.vars.a < rt.vars.b``.
    if isinstance(e, ast.Compare):
        return ast.Compare(
            left=_route_expr(ctx, e.left),
            ops=list(e.ops),
            comparators=[_route_expr(ctx, c) for c in e.comparators],
        )
    # Input: ``f"{a}"``; output: ``f"{rt.vars.a}"``.
    if isinstance(e, ast.JoinedStr):
        values: list[ast.expr] = []
        for part in e.values:
            if isinstance(part, ast.FormattedValue):
                # Format specs are not routed: their inner expressions stay
                # native (a deliberate gap inherited from the first
                # implementation; nothing in the corpus exercises it).
                values.append(
                    ast.FormattedValue(
                        value=_route_expr(ctx, part.value),
                        conversion=part.conversion,
                        format_spec=part.format_spec,
                    )
                )
            else:
                values.append(part)
        return ast.JoinedStr(values=values)
    # Input: ``await a``; output: ``await rt.vars.a``.
    if isinstance(e, ast.Await):
        return ast.Await(value=_route_expr(ctx, e.value))
    # Input: ``yield a``; output: ``yield rt.vars.a`` outside the kernel policy.
    if isinstance(e, ast.Yield):
        if ctx.policy:
            raise _policy_error(ctx, e, "yield")
        return ast.Yield(value=_route_expr(ctx, e.value) if e.value is not None else None)
    # Input: ``yield from a``; the same policy rule as ``yield``.
    if isinstance(e, ast.YieldFrom):
        if ctx.policy:
            raise _policy_error(ctx, e, "yield")
        return ast.YieldFrom(value=_route_expr(ctx, e.value))
    return e


def _route_slice(ctx: _Ctx, s: ast.expr) -> ast.expr:
    # Input slice: ``a[start:stop:step]``; route each bound expression.
    if isinstance(s, ast.Slice):
        return ast.Slice(
            lower=_route_expr(ctx, s.lower) if s.lower is not None else None,
            upper=_route_expr(ctx, s.upper) if s.upper is not None else None,
            step=_route_expr(ctx, s.step) if s.step is not None else None,
        )
    # Input slice: ``a[i:j, k]`` — the stdlib tree nests a tuple of slices
    # and plain indices under one Subscript; route each element as a slice.
    if isinstance(s, ast.Tuple):
        return ast.Tuple(elts=[_route_slice(ctx, el) for el in s.elts], ctx=ast.Load())
    # Input slice: ``a[i]`` (any index expression).
    return _route_expr(ctx, s)


# ----------------------------------------------------------- statement rows ---


def _docstring_text(stmt: _AnyStmt) -> str | None:
    """The statement's docstring text, when it is a bare string expression
    (the parser treats it as metadata on the next emitted IR statement)."""
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        text = stmt.value.value
        if isinstance(text, str):
            return text
    return None


def _comment_event(ctx: _Ctx, comments: Sequence[str]) -> ast.Expr:
    """Build a no-op interpreter event carrying source comments to the Sema."""
    values = ast.List(elts=[ast.Constant(value=comment) for comment in comments], ctx=ast.Load())
    return _expr_stmt(_call(_rt_attr(ctx, "_leading_comments"), [values]))


def _header_end(ctx: _Ctx, stmt: _AnyStmt) -> int:
    """The header's last line (the colon line) of a compound statement.

    The colon line comes from the token stream (the tree does not record
    it): the first depth-0 colon at or after the statement's start line is
    the statement's own header colon — decorators, annotations, defaults,
    and bracketed subexpressions cannot carry one earlier. A one-line suite
    (body on the colon line) excludes that line: its comments belong to the
    body's statements.
    """
    body = _stmt_body(stmt)
    colon = ctx.comments.first_colon_at_or_after(stmt.lineno) if ctx.comments is not None else None
    if colon is None:
        return (body[0].lineno - 1) if body else (getattr(stmt, "end_lineno", None) or stmt.lineno)
    if body and body[0].lineno == colon:
        return colon - 1
    return colon


def _stmt_body(stmt: _AnyStmt) -> list[_AnyStmt]:
    """The directly-owned suite statements of a compound statement."""
    if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
        return stmt.body
    if isinstance(stmt, ast.Try):
        return stmt.body
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return stmt.body
    if isinstance(stmt, ast.Match):
        return []  # a match owns cases, not statements
    return []


def _comments_on_node(ctx: _Ctx, stmt: _AnyStmt, lo: int) -> list[str]:
    """Comments attached to one logical statement.

    A statement owns the comment lines before it, the trailing comment on
    its own last line, and — for compound statements — the header comment on
    the colon line. Inline comments inside parentheses or multi-line headers
    belong to the expressions' whitespace and are not statement comments, so
    they are excluded through the column- and line-precise rules. Comments on
    decorator lines are dropped as well.
    """
    first = stmt.lineno
    leading: list[str] | None = None
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and stmt.decorator_list:
        # Comments before the first decorator are leading; decorator-line
        # comments attach to the Decorator node and are dropped.
        first_dec = min(d.lineno for d in stmt.decorator_list)
        leading = [text for line, text in _drain_tagged(ctx, lo, first - 1) if line < first_dec]
    else:
        leading = ctx.drain_comments(lo, first - 1)
    if isinstance(stmt, _COMPOUND_STMT_TYPES):
        header: list[str] = []
        colon = ctx.comments.first_colon_at_or_after(first) if ctx.comments is not None else None
        body = _stmt_body(stmt)
        if colon is not None and (not body or body[0].lineno != colon):
            # The header comment on the colon line — unless the suite shares
            # that line (a one-line suite's comment belongs to its statement).
            header = ctx.comments.on_line(colon) if ctx.comments is not None else []
        return [*leading, *header]
    end_line = getattr(stmt, "end_lineno", None) or first
    end_col = getattr(stmt, "end_col_offset", 0)
    return [*leading, *ctx.trailing_comment(end_line, end_col)]


def _drain_tagged(ctx: _Ctx, lo: int, hi: int) -> list[tuple[int, str]]:
    if ctx.comments is None:
        return []
    return ctx.comments.drain_tagged(lo, hi)


def _comments_on_else(ctx: _Ctx, stmt: ast.If) -> list[str]:
    """Comments belonging to an ``else`` clause: the lines between the
    then-branch's last statement and the else colon — the reference's Else
    leading/header comments."""
    if not stmt.orelse:
        return []
    then_end = (stmt.body[-1].end_lineno or _header_end(ctx, stmt)) if stmt.body else _header_end(ctx, stmt)
    colon = ctx.comments.first_colon_after(then_end) if ctx.comments is not None else None
    if colon is None:
        return ctx.drain_comments(then_end, stmt.orelse[0].lineno - 1)
    if stmt.orelse[0].lineno == colon:
        # A one-line else suite: the colon line's comment belongs to the
        # suite's own statement.
        return ctx.drain_comments(then_end, colon - 1)
    return ctx.drain_comments(then_end, colon)


def _prepend_comment_event(ctx: _Ctx, rows: list[_AnyStmt], comments: Sequence[str]) -> list[_AnyStmt]:
    """Place a comment event before the rows for one logical statement."""
    if not comments:
        return rows
    return [_comment_event(ctx, comments), *rows]


def _prepend_comments_to_block(ctx: _Ctx, body: list[_AnyStmt], comments: Sequence[str]) -> list[_AnyStmt]:
    """Place an ``else`` comment event inside its generated branch body."""
    if not comments:
        return body
    return [_comment_event(ctx, comments), *body]


def _rewrite_commented_row(ctx: _Ctx, stmt: _AnyStmt, lo: int) -> list[_AnyStmt]:
    """Rewrite a source row, pin its original source position, and preserve
    its logical comment attachment."""
    comments = _comments_on_node(ctx, stmt, lo)
    rows = _rewrite_row(ctx, stmt)
    docstring = _docstring_text(stmt)
    if docstring is not None:
        # A docstring stays the function's first statement (its text is
        # metadata on the next emitted IR statement) and creates no IR, so
        # it needs no span pin.
        comments = [*comments, *(docstring.splitlines() or [""])]
        return [*rows, _comment_event(ctx, comments)] if comments else rows
    # Pin the row's original position before its rows execute: the op and
    # builder layers read the pin instead of walking the Python stack (the
    # emitted code's line numbers are shifted text, not source lines).
    span_row = _expr_stmt(
        _call(
            _rt_attr(ctx, "_span"),
            [
                ast.Constant(value=stmt.lineno + ctx.line_offset),
                ast.Constant(value=stmt.col_offset),
            ],
        )
    )
    rows = [span_row, *rows]
    return _prepend_comment_event(ctx, rows, comments)


def _rewrite_body(ctx: _Ctx, stmts: Sequence[_AnyStmt], lo: int) -> list[_AnyStmt]:
    """Rewrite each statement into zero or more emitted statements.

    ``lo`` is the exclusive lower line bound for comment attachment: the
    previous sibling's end line, or the enclosing construct's header line.
    """
    out: list[_AnyStmt] = []
    for stmt in stmts:
        out.extend(_rewrite_commented_row(ctx, stmt, lo))
        lo = getattr(stmt, "end_lineno", lo)
    return out


def _rewrite_row(ctx: _Ctx, stmt: _AnyStmt) -> list[_AnyStmt]:  # noqa: PLR0912 - one branch per construct
    """Rewrite one statement into its interpreter-call statement rows."""
    # Input: ``use(a)``; output: ``rt._emit(use(rt.vars.a))``.
    if isinstance(stmt, ast.Expr):
        value = _route_expr(ctx, stmt.value)
        if isinstance(stmt.value, ast.Call):
            # Notify the trace interpreter after the call has evaluated.
            return [_expr_stmt(_call(_rt_attr(ctx, "_emit"), [value]))]
        return [_expr_stmt(value)]
    # Input: ``x = a``; output: ``rt.vars.x = rt.vars.a``.
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            if isinstance(t, ast.Name) and t.id == "__class__":
                raise _policy_error(ctx, t, "__class__ reassignment")
        targets = [_route_target(ctx, t) for t in stmt.targets]
        return [_assign_multi(targets, _route_expr(ctx, stmt.value))]
    # Input: ``x += a``; output: ``rt.vars.x += rt.vars.a``.
    if isinstance(stmt, ast.AugAssign):
        # The rewrite preserves assignment-target validity (see the cast in
        # _route_assign_target); the constructor stub cannot express it.
        row: list[_AnyStmt] = [
            ast.AugAssign(
                target=cast("ast.Name | ast.Attribute | ast.Subscript", _route_target(ctx, stmt.target)),
                op=stmt.op,
                value=_route_expr(ctx, stmt.value),
            )
        ]
        return row
    # Input: ``x: Type = a``; output: ``rt.vars.x = rt.vars.a``.
    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is None:
            return []  # bare annotation: name is local-but-unbound; reads error host-faithfully
        return [_assign(_route_target(ctx, stmt.target), _route_expr(ctx, stmt.value))]
    # Input: ``return a``; output: a temp, ``rt.return_``, and a host guard.
    if isinstance(stmt, ast.Return):
        return _return_row(ctx, stmt)
    # Input: ``if a: ...``; output: ``eval_if``, guarded bodies, and close.
    if isinstance(stmt, ast.If):
        return _if_row(ctx, stmt)
    # Input: ``for x in values: ...``; output: a temp-targeted instrumented loop.
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return _for_row(ctx, stmt)
    # Input: ``while a: ...``; output: a hoisted predicate thunk and instrumented loop.
    if isinstance(stmt, ast.While):
        return _while_row(ctx, stmt)
    # Input: ``try: ... finally: ...``; output: the same host structure with instrumented bodies.
    if isinstance(stmt, ast.Try):
        if stmt.handlers:
            raise _policy_error(ctx, stmt, "try/except")
        body_end = (
            (stmt.body[-1].end_lineno or _header_end(ctx, stmt)) if stmt.body else _header_end(ctx, stmt)
        )
        orelse_colon = ctx.comments.first_colon_after(body_end) if ctx.comments is not None else None
        orelse_lo = (
            orelse_colon
            if orelse_colon is not None
            else (stmt.orelse[0].lineno - 1 if stmt.orelse else body_end)
        )
        final_clause = stmt.orelse or stmt.body
        final_start = (final_clause[-1].end_lineno or body_end) if final_clause else body_end
        final_colon = ctx.comments.first_colon_after(final_start) if ctx.comments is not None else None
        final_lo = (
            final_colon
            if final_colon is not None
            else (stmt.finalbody[0].lineno - 1 if stmt.finalbody else final_start)
        )
        return [
            ast.Try(
                body=_rewrite_body(ctx, stmt.body, _header_end(ctx, stmt)),
                handlers=[],
                orelse=_rewrite_body(ctx, stmt.orelse, orelse_lo) if stmt.orelse else [],
                finalbody=_rewrite_body(ctx, stmt.finalbody, final_lo) if stmt.finalbody else [],
            )
        ]
    # Input: ``def helper(): ...``; output: the nested definition plus a binding sync.
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _nested_def_row(ctx, stmt)
    # Input: ``class Helper: ...``; output: the class plus a binding sync.
    if isinstance(stmt, ast.ClassDef):
        if stmt.keywords:
            raise _policy_error(ctx, stmt, "metaclass")
        return _classdef_row(ctx, stmt)
    # Input: ``import module as name``; output: the import plus ``rt.vars.name = name``.
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return _native_binder_row(ctx, stmt, [stmt])
    # Input: ``with manager as value: ...``; output: a generated context manager.
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return _with_row(ctx, stmt)
    # Input: ``match value: case pattern: ...``; output: instrumented subject, guards, and bind syncs.
    if isinstance(stmt, ast.Match):
        return _match_row(ctx, stmt)
    # Input: ``break``; output: ``if rt.break_(): [exit_construct() ...] break``.
    if isinstance(stmt, ast.Break):
        return _break_row(ctx)
    # Input: ``continue``; output: ``if rt.continue_(): [exit_construct() ...] continue``.
    if isinstance(stmt, ast.Continue):
        return _continue_row(ctx)
    # Input: ``del obj.value``; output: ``del rt.vars.obj.value``.
    if isinstance(stmt, ast.Delete):
        return _del_row(ctx, stmt)
    # Input: ``global name`` or ``nonlocal name``; output: a rewrite error.
    if isinstance(stmt, (ast.Global, ast.Nonlocal)):
        raise _policy_error(ctx, stmt, "global/nonlocal")
    # Input: ``raise Error(a)``; output: ``raise Error(rt.vars.a)``.
    if isinstance(stmt, ast.Raise):
        return [
            ast.Raise(
                exc=_route_expr(ctx, stmt.exc) if stmt.exc is not None else None,
                cause=_route_expr(ctx, stmt.cause) if stmt.cause is not None else None,
            )
        ]
    # Input: ``assert a, message``; output: instrumented test and message expressions.
    if isinstance(stmt, ast.Assert):
        return [
            ast.Assert(
                test=_route_expr(ctx, stmt.test),
                msg=_route_expr(ctx, stmt.msg) if stmt.msg is not None else None,
            )
        ]
    # Input: an unsupported-but-host-safe statement (``pass``, ...); unchanged.
    return [stmt]


def _assign_multi(targets: list[ast.expr], value: ast.expr) -> ast.Assign:
    return ast.Assign(targets=targets, value=value)


def _if_construct_stmts(stmt: ast.If) -> list[_AnyStmt]:
    """The statements collected inside this if's scope: both branch
    bodies, and (an elif) the nested if's whole subtree."""
    out = list(stmt.body)
    if stmt.orelse:
        if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
            out.extend(_if_construct_stmts(stmt.orelse[0]))
        else:
            out.extend(stmt.orelse)
    return out


def _if_row(ctx: _Ctx, stmt: ast.If) -> list[_AnyStmt]:
    """Rewrite an ``if`` into predicate, branch-guard, and close operations.

    The predicate is evaluated once. An ``elif`` is represented as an
    else-guard containing a nested rewritten ``if`` (the stdlib tree spells
    ``else:``-wrapped single ``if`` identically, and both produce these same
    rows). When a native transfer inside the branches is bound to an
    enclosing loop, the branch rows are wrapped in the unwind scope guard
    (its exit closes the if at the Construct's lexical boundary during
    keyword unwinding) instead of a trailing close.
    """
    rows: list[_AnyStmt] = [_expr_stmt(_call(_rt_attr(ctx, "eval_if"), [_route_expr(ctx, stmt.test)]))]
    inner: list[_AnyStmt] = [
        ast.If(
            test=_call(_rt_attr(ctx, "eval_then")),
            body=_rewrite_body(ctx, stmt.body, _header_end(ctx, stmt)),
            orelse=[],
        )
    ]
    orelse = stmt.orelse
    if orelse:
        # ``else``-clause line range for comment attachment: the then-branch's
        # last statement through the else body's first statement's header.
        then_end = (
            (stmt.body[-1].end_lineno or _header_end(ctx, stmt)) if stmt.body else _header_end(ctx, stmt)
        )
        inner.append(_expr_stmt(_call(_rt_attr(ctx, "eval_else"))))
        if len(orelse) == 1 and isinstance(orelse[0], ast.If):
            # elif: the nested if's rows under the else-active guard.
            nested = _rewrite_commented_row(ctx, orelse[0], then_end)
            inner.append(ast.If(test=_call(_rt_attr(ctx, "eval_else_active")), body=nested, orelse=[]))
        else:
            else_colon = ctx.comments.first_colon_after(then_end) if ctx.comments is not None else None
            else_lo = else_colon if else_colon is not None else orelse[0].lineno - 1
            else_body = _rewrite_body(ctx, orelse, else_lo)
            else_body = _prepend_comments_to_block(ctx, else_body, _comments_on_else(ctx, stmt))
            inner.append(ast.If(test=_call(_rt_attr(ctx, "eval_else_active")), body=else_body, orelse=[]))
    if _unbound_transfer(_if_construct_stmts(stmt), (ast.Break, ast.Continue), ctx._escapes_memo):
        # The predicate evaluates inside the guard: the guard's entry
        # marker is the enclosing Construct, and the if Construct it installs is
        # the one its exit closes.
        return [_lifetime_with(ctx, [rows[0], *inner])]
    inner.append(_exit_construct_stmt(ctx))
    rows.extend(inner)
    return rows


def _exit_construct_stmt(ctx: _Ctx, else_follows: bool = False) -> ast.Expr:
    """Emit the Construct-close operation.

    ``else_follows`` is the loop-``else`` static ``has_else`` fact: a
    loop-``else`` closes its Construct from inside the else suite, so the
    trace interpreter can reject a staged loop there."""
    keywords = [_kw("else_follows", ast.Constant(value=True))] if else_follows else []
    return _expr_stmt(_call(_rt_attr(ctx, "exit_construct"), [], keywords))


def _lifetime_with(ctx: _Ctx, rows: Sequence[_AnyStmt]) -> ast.With:
    """Wrap transfer-affected rows in the unwind scope guard.

    The guard replaces the trailing close that a native transfer would skip."""
    return ast.With(
        items=[ast.withitem(context_expr=_call(_rt_attr(ctx, "unwind_guard")), optional_vars=None)],
        body=list(rows),
    )


def _escapes_transfer(
    node: ast.AST,
    kinds: tuple[type, ...],
    loop_depth: int,
    _memo: dict[tuple[int, int, int], bool] | None = None,
) -> bool:
    """Does a native transfer of ``kinds`` in ``node``'s subtree escape
    it — i.e. is the keyword's nearest enclosing loop outside the subtree?

    ``loop_depth`` counts loops within the subtree: a keyword bound to an
    inner loop (depth >= 1) does not cross the subtree's constructs. A
    loop's ``else`` suite runs *after* that loop completes, so a transfer
    there binds to the next enclosing loop outside it — the else suite
    traverses at the original depth, only the body at ``loop_depth + 1``
    (a top-level else-suite transfer is a SyntaxError upstream, so the
    enclosing loop always exists). Transfers cannot cross
    function/comprehension boundaries, so a plain subtree walk is exact.
    """
    key = (id(node), len(kinds), loop_depth)
    if _memo is not None and key in _memo:
        return _memo[key]
    result = _escapes_transfer_impl(node, kinds, loop_depth, _memo)
    if _memo is not None:
        _memo[key] = result
    return result


def _escapes_transfer_impl(
    node: ast.AST, kinds: tuple[type, ...], loop_depth: int, _memo: dict[tuple[int, int, int], bool] | None
) -> bool:
    if isinstance(node, (ast.Break, ast.Continue)):
        return loop_depth == 0 and isinstance(node, kinds)
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        if any(_escapes_transfer(s, kinds, loop_depth + 1, _memo) for s in node.body):
            return True
        if node.orelse:
            return any(_escapes_transfer(s, kinds, loop_depth, _memo) for s in node.orelse)
        return False
    return any(_escapes_transfer(child, kinds, loop_depth, _memo) for child in ast.iter_child_nodes(node))


def _unbound_transfer(
    body: Sequence[_AnyStmt],
    kinds: tuple[type, ...],
    memo: dict[tuple[int, int, int], bool] | None = None,
) -> bool:
    """Does the suite hold a native transfer of ``kinds`` whose target
    loop is outside the suite?

    - For an enclosing construct (if): any unbound ``break`` or
      ``continue`` crosses its close.
    - For a loop: only ``break`` does — ``continue`` targets the loop
      itself and never skips its close.
    """
    return any(_escapes_transfer(stmt, kinds, 0, memo) for stmt in body)


def _native_transfer_row(ctx: _Ctx, op: str, keyword: _AnyStmt) -> list[_AnyStmt]:
    """Rewrite native ``break``/``continue`` under the transfer guard.

    The return value decides whether the keyword runs. Cleanup is emitted by
    the unwind scope guard around transfer-affected constructs; its exit runs
    during keyword unwinding in Python's order, after inner source context
    managers and ``finally`` bodies.
    """
    return [ast.If(test=_call(_rt_attr(ctx, op)), body=[keyword], orelse=[])]


def _break_row(ctx: _Ctx) -> list[_AnyStmt]:
    return _native_transfer_row(ctx, "break_", ast.Break())


def _continue_row(ctx: _Ctx) -> list[_AnyStmt]:
    return _native_transfer_row(ctx, "continue_", ast.Continue())


def _del_row(ctx: _Ctx, stmt: ast.Delete) -> list[_AnyStmt]:
    """Rewrite ``del`` while rejecting local-name deletion.

    Deleting a local would leave the instrumented locals holding a stale
    value; attribute and subscript deletion do not create that mismatch.
    """
    leaves: list[ast.expr] = []
    for target in stmt.targets:
        leaves.extend(_del_leaf_targets(target))
    if any(isinstance(t, ast.Name) for t in leaves):
        raise _policy_error(ctx, stmt, "del <name>")
    return [ast.Delete(targets=[_route_target(ctx, t) for t in stmt.targets])]


def _del_leaf_targets(target: ast.expr) -> list[ast.expr]:
    """Flatten a ``del`` target to identify local-name deletions.

    Accepted input: ``del (a, obj.value)``; leaves: ``a``, ``obj.value``.
    """
    if isinstance(target, ast.Tuple):
        out: list[ast.expr] = []
        for el in target.elts:
            out.extend(_del_leaf_targets(el))
        return out
    return [target]


def _for_row(ctx: _Ctx, stmt: ast.For | ast.AsyncFor) -> list[_AnyStmt]:
    """Rewrite a ``for`` with an instrumented target and a balanced Construct entry.

    The iterable is evaluated once by ``begin_for``. A fresh target temp
    preserves host tuple unpacking while redirecting each bound name. A
    ``for``-``else`` keeps its native completion behavior and closes the
    Construct from inside the else branch (``else_follows`` — the trace
    interpreter rejects the close of a staged loop there). A ``break`` in the
    body skips both close sites, so the loop rows are wrapped in the unwind
    scope guard instead (its exit closes the loop Construct after keyword
    unwinding); a ``continue`` never skips the loop's own close and needs no
    guard for the loop itself.
    """
    temp = ctx.fresh_temp("it")
    body: list[_AnyStmt] = [_assign(_route_target(ctx, stmt.target), _n(temp))]
    body.extend(_rewrite_body(ctx, stmt.body, _header_end(ctx, stmt)))
    # The reference rebuilds the loop as a plain ``for`` — an ``async for``
    # is de-asynced by the rewrite (mirrored here; type-preserving rebuilds
    # would diverge from the reference's emitted source).
    new_for = ast.For(
        target=_n_store(temp),
        iter=_call(_rt_attr(ctx, "begin_for"), [_route_expr(ctx, stmt.iter)]),
        body=body,
        orelse=[],
    )
    rows: list[_AnyStmt]
    if stmt.orelse:
        body_end = (
            (stmt.body[-1].end_lineno or _header_end(ctx, stmt)) if stmt.body else _header_end(ctx, stmt)
        )
        colon = ctx.comments.first_colon_after(body_end) if ctx.comments is not None else None
        else_lo = colon if colon is not None else stmt.orelse[0].lineno - 1
        close = [_exit_construct_stmt(ctx, else_follows=True)]
        else_body = close + _rewrite_body(ctx, stmt.orelse, else_lo)
        rows = [ast.For(target=new_for.target, iter=new_for.iter, body=new_for.body, orelse=else_body)]
    else:
        rows = [new_for, _exit_construct_stmt(ctx)]
    if _unbound_transfer(stmt.body, (ast.Break,), ctx._escapes_memo):
        # The guard replaces the trailing close (the pre-else close inside
        # the else suite stays: it pops the loop Construct, making the guard
        # exit a no-op).
        return [_lifetime_with(ctx, rows[:1])]
    return rows


def _while_row(ctx: _Ctx, stmt: ast.While) -> list[_AnyStmt]:
    """Rewrite a ``while`` with a stable predicate thunk and balanced scope.

    The thunk is hoisted beside the loop, so its identity names the site
    for one entry into the enclosing block: the host ``while`` re-evaluates
    the same thunk until the loop exits. Host execution retains native
    re-evaluation; staged execution traces one representative body
    iteration. A ``while``-``else`` closes the Construct from inside the else
    branch (``else_follows`` — the trace interpreter rejects the close of a
    staged loop there). A ``break`` in the body skips both close sites, so
    the loop rows are wrapped in the unwind scope guard instead.
    """
    temp = ctx.fresh_temp("w")
    hoist = _assign(_n_store(temp), ast.Lambda(args=_empty_args(), body=_route_expr(ctx, stmt.test)))
    rewritten_body = _rewrite_body(ctx, stmt.body, _header_end(ctx, stmt))
    new_while = ast.While(
        test=_call(_rt_attr(ctx, "eval_bool"), [_n(temp)]),
        body=rewritten_body,
        orelse=[],
    )
    rows: list[_AnyStmt]
    if stmt.orelse:
        body_end = (
            (stmt.body[-1].end_lineno or _header_end(ctx, stmt)) if stmt.body else _header_end(ctx, stmt)
        )
        colon = ctx.comments.first_colon_after(body_end) if ctx.comments is not None else None
        else_lo = colon if colon is not None else stmt.orelse[0].lineno - 1
        close = [_exit_construct_stmt(ctx, else_follows=True)]
        else_body = close + _rewrite_body(ctx, stmt.orelse, else_lo)
        rows = [ast.While(test=new_while.test, body=new_while.body, orelse=else_body)]
    else:
        rows = [new_while, _exit_construct_stmt(ctx)]
    if _unbound_transfer(stmt.body, (ast.Break,), ctx._escapes_memo):
        # The guard replaces the trailing close (as in _for_row).
        return [hoist, _lifetime_with(ctx, rows[:1])]
    return [hoist, *rows]


def _single_call_stmts(stmts: Sequence[_AnyStmt]) -> bool:
    """The with-suite is exactly one call-valued simple statement.

    One ``Assign`` / ``AnnAssign`` / ``Expr`` whose value is a ``Call``
    (``x = f()``, ``x: T = f()``, or a bare ``f()``). Semicolon suites,
    compound statements, and empty suites are not single-call. Read on the
    user's original suite — the rewritten body routes every statement
    through ``rt.*`` calls, which would make any one-statement body look
    call-valued.
    """
    if len(stmts) != 1:
        return False
    one = stmts[0]
    return isinstance(one, (ast.Assign, ast.AnnAssign, ast.Expr)) and isinstance(one.value, ast.Call)


def _with_row(ctx: _Ctx, stmt: ast.With | ast.AsyncWith) -> list[_AnyStmt]:
    """Rewrite ``with`` through an exception-safe generated context manager.

    The host context-manager protocol remains responsible for enter and exit
    order. ``as`` targets use fresh temps so each target is synchronized
    immediately after its own ``__enter__``. Multiple items are nested to
    preserve left-to-right entry, right-to-left exit, and exception safety. A
    native transfer out of the body needs no additional guard: the generated
    context manager itself runs during keyword unwinding at the exact lexical
    boundary.
    """
    inner: list[_AnyStmt] = _rewrite_body(ctx, stmt.body, _header_end(ctx, stmt))
    single_call = _single_call_stmts(stmt.body)
    # The reference rebuilds ``with`` as a plain ``with`` — an ``async with``
    # is de-asynced by the rewrite (mirrored).
    for index, item in enumerate(reversed(stmt.items)):
        as_target = item.optional_vars
        call_args = [_route_expr(ctx, item.context_expr)]
        keywords: list[ast.keyword] = []
        if isinstance(as_target, ast.Name):
            # Pass a single-name target to the Sema; tuple targets are
            # synchronized element by element instead.
            keywords.append(_kw("name", ast.Constant(value=as_target.id)))
        # The static single-call fact for the innermost item only (an outer
        # item's effective body is the nested with, never a single call) —
        # the rewriter's mirror of the parser's body-shape rule, threaded to
        # the construct through the interpreter (``as_name`` rides the same
        # channel).
        keywords.append(_kw("single_call_body", ast.Constant(value=index == 0 and single_call)))
        wrapped = _call(_rt_attr(ctx, "on_begin_with"), call_args, keywords)
        if as_target is not None:
            temp = ctx.fresh_temp("with")
            sync = _assign(_route_target(ctx, as_target), _n(temp))
            this = ast.With(
                items=[ast.withitem(context_expr=wrapped, optional_vars=_n_store(temp))],
                body=[sync, *inner],
            )
        else:
            this = ast.With(items=[ast.withitem(context_expr=wrapped, optional_vars=None)], body=inner)
        inner = [this]
    return inner


def _route_pattern(ctx: _Ctx, pattern: _AnyPattern) -> _AnyPattern:
    """Route pattern value expressions while leaving binders native."""

    # Input: ``case rt_source.kind``; output: ``case rt.vars.rt_source.kind``.
    if isinstance(pattern, ast.MatchValue):
        return ast.MatchValue(value=_route_expr(ctx, pattern.value))
    # Input: ``case Point(x=constants.VALUE)``; route class/value expressions.
    if isinstance(pattern, ast.MatchClass):
        return ast.MatchClass(
            cls=_route_expr(ctx, pattern.cls),
            patterns=[_route_pattern(ctx, p) for p in pattern.patterns],
            kwd_attrs=list(pattern.kwd_attrs),
            kwd_patterns=[_route_pattern(ctx, p) for p in pattern.kwd_patterns],
        )
    # Input: ``case {constants.KEY: value}``; route the mapping key expression.
    if isinstance(pattern, ast.MatchMapping):
        return ast.MatchMapping(
            keys=[_route_expr(ctx, k) for k in pattern.keys],
            patterns=[_route_pattern(ctx, p) for p in pattern.patterns],
            rest=pattern.rest,
        )
    # Input: ``case [constants.VALUE, value]``; route value expressions, not binders.
    if isinstance(pattern, ast.MatchSequence):
        return ast.MatchSequence(patterns=[_route_pattern(ctx, p) for p in pattern.patterns])
    # Input: ``case constants.A | constants.B``; route each alternative expression.
    if isinstance(pattern, ast.MatchOr):
        return ast.MatchOr(patterns=[_route_pattern(ctx, p) for p in pattern.patterns])
    # Input: ``case constants.VALUE as name``; route only the nested value pattern.
    if isinstance(pattern, ast.MatchAs) and pattern.pattern is not None:
        return ast.MatchAs(pattern=_route_pattern(ctx, pattern.pattern), name=pattern.name)
    return pattern


def _match_row(ctx: _Ctx, stmt: ast.Match) -> list[_AnyStmt]:
    """Rewrite a ``match`` while preserving native binder behavior.

    The subject is evaluated once. Pattern values and guards route outer
    reads, while binders remain native until synchronized into ``rt.vars``.
    Guards run before that synchronization because host matching binds names
    before evaluating a guard. A guarded-away or never-executed binder is
    synchronized defensively so an unread local remains unbound.
    """
    all_binders: set[str] = set()
    cases: list[ast.match_case] = []
    for case in stmt.cases:
        binders_list = _pattern_binders(case.pattern)
        binders = frozenset(binders_list)
        all_binders |= binders
        body: list[_AnyStmt] = [_assign(_route_target(ctx, _n_store(n)), _n(n)) for n in binders_list]
        body_first = case.body[0].lineno if case.body else None
        case_colon = (
            ctx.comments.last_colon_in(getattr(case, "lineno", stmt.lineno), body_first)
            if ctx.comments is not None and body_first is not None
            else None
        )
        case_lo = (
            case_colon if case_colon is not None else (body_first - 1 if body_first else stmt.lineno - 1)
        )
        body.extend(_rewrite_body(ctx, case.body, case_lo))
        cases.append(
            ast.match_case(
                pattern=_route_pattern(ctx, case.pattern),
                guard=_route_expr(ctx.with_shadow(binders), case.guard) if case.guard is not None else None,
                body=body,
            )
        )
    rows: list[_AnyStmt] = [ast.Match(subject=_route_expr(ctx, stmt.subject), cases=cases)]
    for n in sorted(all_binders):
        sync = _assign(_route_target(ctx, _n_store(n)), _n(n))
        rows.append(
            ast.Try(
                body=[sync],
                handlers=[ast.ExceptHandler(type=_n("UnboundLocalError"), name=None, body=[ast.Pass()])],
                orelse=[],
                finalbody=[],
            )
        )
    return rows


def _policy_error(ctx: _Ctx, node: ast.AST, what: str) -> RewriteError:
    """Create a span-pointed error for a construct outside the rewrite subset."""
    return RewriteError(f"rewrite-unsafe construct ({what}) in the kernel body", ctx.span(node))


def _check_forbidden_call(ctx: _Ctx, call: ast.Call) -> None:
    """Reject dynamic operations that make collection non-deterministic."""
    func = call.func
    if isinstance(func, ast.Name) and func.id in ("eval", "exec", "compile", "__import__"):
        raise _policy_error(ctx, func, f"{func.id}()")
    if isinstance(func, ast.Name) and func.id in ("getattr", "setattr", "hasattr", "delattr"):
        if len(call.args) < 2 or not (
            isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)
        ):
            raise _policy_error(ctx, func, f"{func.id}() with a non-literal name")


def _return_row(ctx: _Ctx, stmt: ast.Return) -> list[_AnyStmt]:
    """Rewrite ``return`` with one evaluation and a host-execution exit guard.

    The trace interpreter records the value before the generated function
    returns. A native return is emitted only while execution remains on the
    host; staged returns are handled by the Sema.
    """
    if stmt.value is None:
        return [
            _expr_stmt(_call(_rt_attr(ctx, "return_"))),
            ast.If(
                test=_call(_rt_attr(ctx, "in_host_mode")),
                body=[ast.Return(value=None)],
                orelse=[],
            ),
        ]
    temp = ctx.fresh_temp("ret")
    value = _route_expr(ctx, stmt.value)
    return [
        _assign(_n_store(temp), value),
        _expr_stmt(_call(_rt_attr(ctx, "return_"), [_n(temp)])),
        ast.If(
            test=_call(_rt_attr(ctx, "in_host_mode")),
            body=[ast.Return(value=_n(temp))],
            orelse=[],
        ),
    ]


def _reads_only_body(ctx: _Ctx, stmts: Sequence[_AnyStmt]) -> list[_AnyStmt]:
    """Rewrite a nested function's body for reads only.

    The nested function owns its binders and statement execution. Reads of
    outer kernel locals still read the outer function's instrumented locals.
    """
    return [_reads_only_stmt(ctx, s) for s in stmts]


def _reads_only_assign_target(ctx: _Ctx, target: ast.expr) -> ast.expr:
    """Route reads inside nested-scope assignment targets; binders stay native."""
    if isinstance(target, ast.Attribute):
        return ast.Attribute(value=_route_expr(ctx, target.value), attr=target.attr, ctx=ast.Store())
    if isinstance(target, ast.Subscript):
        return ast.Subscript(
            value=_route_expr(ctx, target.value), slice=_route_slice(ctx, target.slice), ctx=ast.Store()
        )
    if isinstance(target, (ast.Tuple, ast.List)):
        cls = type(target)
        return cls(elts=[_reads_only_assign_target(ctx, el) for el in target.elts], ctx=ast.Store())
    if isinstance(target, ast.Starred):
        return ast.Starred(value=_reads_only_assign_target(ctx, target.value), ctx=ast.Store())
    return target


def _reads_only_stmt(ctx: _Ctx, stmt: _AnyStmt) -> _AnyStmt:  # noqa: PLR0912 - one branch per construct
    # Input: nested ``use(a)``; output: ``use(rt.vars.a)``.
    if isinstance(stmt, ast.Expr):
        return ast.Expr(value=_route_expr(ctx, stmt.value))
    # Input: nested ``x = a``; output: ``x = rt.vars.a`` (native binder stays local).
    if isinstance(stmt, ast.Assign):
        return ast.Assign(targets=list(stmt.targets), value=_route_expr(ctx, stmt.value))
    # Input: nested ``obj.a += value``; output: ``rt.vars.obj.a += value``.
    if isinstance(stmt, ast.AugAssign):
        return ast.AugAssign(
            target=cast(
                "ast.Name | ast.Attribute | ast.Subscript",
                _reads_only_assign_target(ctx, stmt.target),
            ),
            op=stmt.op,
            value=_route_expr(ctx, stmt.value),
        )
    # Input: nested ``x: Type = a``; route annotation and value reads only.
    if isinstance(stmt, ast.AnnAssign):
        # Eager annotations resolve outer names through the outer function's
        # instrumented locals; future annotations are strings and must remain
        # verbatim.
        return ast.AnnAssign(
            target=cast(
                "ast.Name | ast.Attribute | ast.Subscript",
                _reads_only_assign_target(ctx, stmt.target),
            ),
            annotation=stmt.annotation if ctx.future_annotations else _route_expr(ctx, stmt.annotation),
            value=_route_expr(ctx, stmt.value) if stmt.value is not None else None,
            simple=stmt.simple,
        )
    # Input: nested ``return a``; output: ``return rt.vars.a``.
    if isinstance(stmt, ast.Return):
        return ast.Return(value=_route_expr(ctx, stmt.value) if stmt.value is not None else None)
    # Input: nested ``if a: use(a)``; output: same control flow with instrumented reads.
    if isinstance(stmt, ast.If):
        return _reads_only_if(ctx, stmt)
    # Input: nested ``for x in a: use(x)``; output: instrumented iterable, native target.
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return ast.For(
            target=stmt.target,
            iter=_route_expr(ctx, stmt.iter),
            body=_reads_only_body(ctx, stmt.body),
            orelse=_reads_only_body(ctx, stmt.orelse),
        )
    # Input: nested ``while a: use(a)``; output: instrumented predicate and body reads.
    if isinstance(stmt, ast.While):
        return ast.While(
            test=_route_expr(ctx, stmt.test),
            body=_reads_only_body(ctx, stmt.body),
            orelse=_reads_only_body(ctx, stmt.orelse),
        )
    # Input: nested ``with a as value: use(value)``; output: instrumented manager only.
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return ast.With(
            items=[
                ast.withitem(context_expr=_route_expr(ctx, i.context_expr), optional_vars=i.optional_vars)
                for i in stmt.items
            ],
            body=_reads_only_body(ctx, stmt.body),
        )
    # Input: nested ``match a: case value: use(value)``; route outer reads only.
    if isinstance(stmt, ast.Match):
        # Pattern binders belong to the nested scope; route only outer reads.
        cases: list[ast.match_case] = []
        for c in stmt.cases:
            binders = frozenset(_pattern_binders(c.pattern))
            cases.append(
                ast.match_case(
                    pattern=_route_pattern(ctx, c.pattern),
                    guard=(_route_expr(ctx.with_shadow(binders), c.guard) if c.guard is not None else None),
                    body=_reads_only_body(ctx.with_shadow(binders), c.body),
                )
            )
        return ast.Match(subject=_route_expr(ctx, stmt.subject), cases=cases)
    # Input: nested ``try: use(a) finally: close(a)``; preserve host structure.
    if isinstance(stmt, ast.Try):
        return ast.Try(
            body=_reads_only_body(ctx, stmt.body),
            handlers=[
                ast.ExceptHandler(type=h.type, name=h.name, body=_reads_only_body(ctx, h.body))
                for h in stmt.handlers
            ],
            orelse=_reads_only_body(ctx, stmt.orelse),
            finalbody=_reads_only_body(ctx, stmt.finalbody),
        )
    # Input: nested ``def helper(): return a``; output: a helper with instrumented reads.
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _nested_def_row(ctx, stmt)[0]
    # Input: nested ``class Helper: value = a``; output: a class body with instrumented reads.
    if isinstance(stmt, ast.ClassDef):
        return _classdef_row(ctx, stmt)[0]
    return stmt


def _reads_only_if(ctx: _Ctx, stmt: ast.If) -> ast.If:
    """Rewrite an ``if`` for reads only; elif chains stay elif chains."""
    return ast.If(
        test=_route_expr(ctx, stmt.test),
        body=_reads_only_body(ctx, stmt.body),
        orelse=_reads_only_body(ctx, stmt.orelse),
    )


def _nested_def_row(ctx: _Ctx, stmt: ast.FunctionDef | ast.AsyncFunctionDef) -> list[_AnyStmt]:
    """Preserve a nested function while redirecting outer reads and its binding."""
    locals_ = frozenset(_param_names(stmt.args))
    inner = _Ctx(
        ctx.prefix, ctx.instrumented, ctx.names, policy=False, future_annotations=ctx.future_annotations
    )
    inner.shadow = ctx.shadow | locals_
    collected: set[str] = set()
    for s in stmt.body:
        _collect_stmt(s, collected)
    inner.shadow |= frozenset(collected)
    fn_cls = type(stmt)
    new_fn = fn_cls(
        name=stmt.name,
        args=_rewrite_params(ctx, stmt.args),
        body=_reads_only_body(inner, stmt.body),
        decorator_list=_rewrite_decorators(ctx, stmt.decorator_list),
        returns=(
            stmt.returns if ctx.future_annotations or stmt.returns is None else _route_expr(ctx, stmt.returns)
        ),
        type_comment=stmt.type_comment,
    )
    _copy_type_params(stmt, new_fn)
    return _native_binder_row(ctx, stmt, [new_fn])


def _classdef_row(ctx: _Ctx, stmt: ast.ClassDef) -> list[_AnyStmt]:
    """Preserve a nested class while redirecting outer reads and its binding.

    Class bodies resolve enclosing function locals as native free variables,
    so reads of outer kernel locals must still route through ``rt.vars``.
    Class bodies are host plumbing and are not subject to kernel subset checks.
    """
    collected: set[str] = set()
    for s in stmt.body:
        _collect_stmt(s, collected)
    inner = _Ctx(
        ctx.prefix, ctx.instrumented, ctx.names, policy=False, future_annotations=ctx.future_annotations
    )
    inner.shadow = ctx.shadow | frozenset(collected)
    new_cls = ast.ClassDef(
        name=stmt.name,
        bases=[_route_expr(ctx, b) for b in stmt.bases],
        keywords=list(stmt.keywords),
        body=_reads_only_body(inner, stmt.body),
        decorator_list=_rewrite_decorators(ctx, stmt.decorator_list),
    )
    _copy_type_params(stmt, new_cls)
    return _native_binder_row(ctx, stmt, [new_cls])


def _native_binder_row(ctx: _Ctx, original: _AnyStmt, rewritten: Sequence[_AnyStmt]) -> list[_AnyStmt]:
    """Emit a native binder and synchronize instrumented names into ``rt.vars``."""
    # Input: ``import module as name``; output: import followed by ``rt.vars.name = name``.
    names: list[str] = []
    # Accepted binders: nested definitions, classes, and imports.
    if isinstance(original, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.append(original.name)
    elif isinstance(original, ast.Import):
        names.extend(_import_local_name(a) for a in original.names)
    elif isinstance(original, ast.ImportFrom):
        if any(a.name == "*" for a in original.names):
            raise RewriteError("wildcard imports are not supported in collector kernels")
        names.extend(_import_local_name(a) for a in original.names)
    out: list[_AnyStmt] = list(rewritten)
    for name in names:
        if name in ctx.instrumented and name not in ctx.shadow:
            out.append(_assign(ast.Attribute(value=_rt_vars(ctx), attr=name, ctx=ast.Store()), _n(name)))
    return out


# ------------------------------------------------------------ public entries ---


def _is_docstring(stmt: _AnyStmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def emit_kernel(
    source: str,
    name: str,
    freevars: tuple[str, ...] = (),
    future_annotations: bool = False,
    line_offset: int = 0,
) -> tuple[str, str]:
    """The driver-facing entry: instrument one top-level function definition.

    Rewrite the body, add the frame-entry
    prologue after a leading docstring, and — for a kernel with free
    variables — wrap the definition in the synthetic environment that binds
    them before compilation (a compilation scaffold, never called).
    """
    tree = ast.parse(source)
    fn_def = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            fn_def = node
            break
    if fn_def is None:
        raise RewriteError(f"cannot find function {name!r} in the given source")

    instrumented, names = collect_locals(fn_def)
    prefix = choose_prefix(names)
    ctx = _Ctx(
        prefix=prefix,
        instrumented=instrumented,
        names=names,
        comments=_Comments(source),
        future_annotations=future_annotations,
        line_offset=line_offset,
    )
    body = _rewrite_body(ctx, fn_def.body, _header_end(ctx, fn_def))
    params = _param_names(fn_def.args)
    # Generates ``rt.enter_frame(a=a, ...)`` immediately after a source docstring.
    prologue = _expr_stmt(_call(_rt_attr(ctx, "enter_frame"), [], [_kw(p, _n(p)) for p in params]))
    if body and _is_docstring(body[0]):
        body = [body[0], prologue, *body[1:]]
    else:
        body = [prologue, *body]
    # The reference rebuilds the wrapper as a plain ``def`` — an ``async
    # def`` kernel is de-asynced and its decorators dropped (the instrumented
    # clone must not re-run them; nested helpers keep theirs).
    new_fn = ast.FunctionDef(
        name=fn_def.name,
        args=fn_def.args,
        body=body,
        decorator_list=[],
        returns=fn_def.returns,
        type_comment=fn_def.type_comment,
    )
    _copy_type_params(fn_def, new_fn)
    # Input with free variable ``limit``; output includes an uncalled synthetic
    # environment binding ``limit`` before the rewritten function.
    if freevars:
        env_body: list[_AnyStmt] = [_assign(_n_store(fv), ast.Constant(value=None)) for fv in freevars]
        env_body.append(new_fn)
        env = ast.FunctionDef(
            name=f"__collector_env_{name}",
            args=_empty_args(),
            body=env_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        module = ast.Module([env], [])
    else:
        module = ast.Module([new_fn], [])
    # Manually built nodes carry no locations; unparse requires them present.
    ast.fix_missing_locations(module)
    return ast.unparse(module), prefix


def instrument_function_source(
    source: str, name: str, future_annotations: bool = False
) -> tuple[str, str, list[str]]:
    """Instrument a named function and return its source, generated name, and parameters.

    The test-facing entry point (mirrors ``rewriter.instrument_function_source``):
    nested functions are found too, and the returned source form lets tests
    inspect generated statements.
    """
    tree = ast.parse(source)
    fn_def = _find_function(tree, name)
    if fn_def is None:
        raise RewriteError(f"cannot find function {name!r} in the given source")
    instrumented, names = collect_locals(fn_def)
    prefix = choose_prefix(names)
    ctx = _Ctx(
        prefix=prefix,
        instrumented=instrumented,
        names=names,
        comments=_Comments(source),
        future_annotations=future_annotations,
    )
    body = _rewrite_body(ctx, fn_def.body, _header_end(ctx, fn_def))
    new_fn = ast.FunctionDef(
        name=fn_def.name,
        args=fn_def.args,
        body=body,
        decorator_list=[],
        returns=fn_def.returns,
        type_comment=fn_def.type_comment,
    )
    _copy_type_params(fn_def, new_fn)
    module = ast.Module([new_fn], [])
    ast.fix_missing_locations(module)
    return ast.unparse(module), prefix, _param_names(fn_def.args)


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


__all__ = [
    "RewriteError",
    "collect_locals",
    "emit_kernel",
    "instrument_function_source",
]
