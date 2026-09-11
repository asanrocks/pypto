#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Migrate legacy ``init_values`` / ``pl.yield_()`` test bodies to the
``pl.intrin`` assembly dialect, and flip their decorators to tracing.

Adapted from arch-r8's ``migrate_intrin.py`` for r9-dev-2's separation
model (06 §14, 07 §3.2): the printer still emits ``init_values=`` /
``pl.yield_()`` (parser-owned round-trip), and ``pl.intrin.*`` is a
tracing-only surface. So a rewritten body MUST run under the tracing
frontend — ``@pl.function`` / ``@pl.program`` — because the AST parser
does not learn ``pl.intrin.*`` semantics.

Covers (the common cases; not exhaustive — manual tweaks for the rest):

- ``for i, (a, b) in pl.range(..., init_values=(x, y)): ... a, b = pl.yield_(...)``
  →  ``(a, b) = pl.intrin.carry(x, y); for i in pl.range(...): ... pl.intrin.yield_(...);``
     ``(r,) = pl.intrin.leak()``
- ``if c: r = pl.yield_(x) else: r = pl.yield_(y)`` → bare ``pl.intrin.yield_`` per branch
  + ``(r,) = pl.intrin.leak()``

The ``while`` path IS ported: ``for cnt, x in pl.while_(init_values=...)``
with a ``pl.cond(...)`` first-body statement rewrites to ``pl.intrin.carry``
+ a bare ``while`` + ``pl.intrin.yield_`` + ``pl.intrin.leak``. ``try:/finally:``
blocks (backend-setup fixtures) are recursed into so kernels inside them
migrate too. String-embedded kernels (subprocess / expected-output
triple-quoted literals) are not reachable by libcst and are left for a
manual tweak.

Decorator flip: a function/program whose body ``transform_block`` actually
rewrote gets ``@pl.function`` / ``@pl.program``; kwargs are preserved.
Change detection is identity-based (a node that was not rewritten passes
through as the same object), so untouched subtrees — and their comments —
are preserved exactly and the script is idempotent.

Usage::

    python scripts/migrate_intrin.py tests/ut/path/to/test_x.py ...
"""

import pathlib
import sys
from typing import cast

import libcst as cst
from typing_extensions import TypeIs

_DECORATOR_FLIP = {
    "function_parser": "function",
    "program_parser": "program",
}


def code_for(node: cst.CSTNode) -> str:
    return cst.Module(body=[]).code_for_node(node).strip()


# ---------------------------------------------------------------------------
# Legacy-marker detection
# ---------------------------------------------------------------------------


def is_pl_yield_call(node: cst.CSTNode) -> TypeIs[cst.Call]:
    if not isinstance(node, cst.Call):
        return False
    func = node.func
    return (
        isinstance(func, cst.Attribute)
        and isinstance(func.value, cst.Name)
        and func.value.value == "pl"
        and func.attr.value == "yield_"
    )


def is_range_like_call(
    node: cst.CSTNode, names=("range", "parallel", "pipeline", "unroll")
) -> TypeIs[cst.Call]:
    if not isinstance(node, cst.Call):
        return False
    func = node.func
    return (
        isinstance(func, cst.Attribute)
        and isinstance(func.value, cst.Name)
        and func.value.value == "pl"
        and func.attr.value in names
    )


def is_while_call(node: cst.CSTNode) -> TypeIs[cst.Call]:
    if not isinstance(node, cst.Call):
        return False
    func = node.func
    return (
        isinstance(func, cst.Attribute)
        and isinstance(func.value, cst.Name)
        and func.value.value == "pl"
        and func.attr.value == "while_"
    )


def is_cond_call(node: cst.CSTNode) -> TypeIs[cst.Call]:
    """A bare ``pl.cond(<expr>)`` statement — the while-loop condition in
    the parser dialect."""
    call: cst.Call | None = None
    if isinstance(node, cst.Expr):
        call = node.value
    elif (
        isinstance(node, cst.SimpleStatementLine)
        and len(node.body) == 1
        and isinstance(node.body[0], cst.Expr)
    ):
        call = node.body[0].value
    else:
        return False
    if not isinstance(call, cst.Call):
        return False
    func = call.func
    return (
        isinstance(func, cst.Attribute)
        and isinstance(func.value, cst.Name)
        and func.value.value == "pl"
        and func.attr.value == "cond"
    )


def get_cond_arg_from_stmt(stmt: cst.CSTNode) -> str | None:
    call: cst.Call | None = None
    if (
        isinstance(stmt, cst.SimpleStatementLine)
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], cst.Expr)
    ):
        call = stmt.body[0].value
    elif isinstance(stmt, cst.Expr):
        call = stmt.value
    if isinstance(call, cst.Call) and call.args:
        return code_for(call.args[0].value)
    return None


def extract_while_carry_info(
    for_node: cst.For,
) -> tuple[list[str], list[str], str | None, list[cst.BaseStatement]] | None:
    """``for cnt, x in pl.while_(init_values=(n, x_0)): pl.cond(c); ...`` →
    (carry_vars, init_exprs, cond_code, body_after_cond)."""
    call = for_node.iter
    if not is_while_call(call):
        return None
    init_val = get_init_values_kwarg(call)
    if init_val is None:
        return None
    init_codes: list[str] = []
    if isinstance(init_val, cst.Tuple):
        for el in init_val.elements:
            init_codes.append(code_for(el.value))
    else:
        init_codes.append(code_for(init_val))
    target = for_node.target
    target_names: list[str] = []
    if isinstance(target, cst.Tuple):
        for el in target.elements:
            if isinstance(el.value, cst.Tuple):
                for sub in el.value.elements:
                    target_names.append(code_for(sub.value))
            else:
                target_names.append(code_for(el.value))
    elif isinstance(target, cst.Name):
        target_names.append(code_for(target))
    else:
        target_names.append(code_for(target))
    body_stmts: list[cst.BaseStatement] = (
        list(for_node.body.body) if isinstance(for_node.body, cst.IndentedBlock) else list(for_node.body.body)
    )
    cond_code: str | None = None
    remaining = body_stmts
    if body_stmts and is_cond_call(body_stmts[0]):
        cond_code = get_cond_arg_from_stmt(body_stmts[0])
        remaining = body_stmts[1:]
    return (target_names, init_codes, cond_code, remaining)


def get_init_values_kwarg(call_node: cst.Call) -> cst.CSTNode | None:
    for arg in call_node.args:
        if arg.keyword and arg.keyword.value == "init_values":
            return arg.value
    return None


def get_yield_assign_info(simple_line: cst.CSTNode) -> tuple[str, str, cst.CSTNode] | None:
    """``r = pl.yield_(x)`` / ``a, b = pl.yield_(...)`` / ``r: T = pl.yield_(s)``."""
    if not isinstance(simple_line, cst.SimpleStatementLine) or len(simple_line.body) != 1:
        return None
    stmt = simple_line.body[0]
    if isinstance(stmt, cst.Assign):
        if len(stmt.targets) != 1 or not is_pl_yield_call(stmt.value):
            return None
        target_code = code_for(stmt.targets[0].target)
        args_code = ", ".join(code_for(a.value) for a in stmt.value.args)
        return (target_code, args_code, stmt.targets[0].target)
    if isinstance(stmt, cst.AnnAssign):
        if stmt.value is None or not is_pl_yield_call(stmt.value):
            return None
        target_code = code_for(stmt.target)
        args_code = ", ".join(code_for(a.value) for a in stmt.value.args)
        return (target_code, args_code, stmt.target)
    return None


def convert_yield_assign_to_bare(simple_line: cst.SimpleStatementLine) -> cst.BaseStatement:
    info = get_yield_assign_info(simple_line)
    if not info:
        return simple_line
    _, args_code, _ = info
    return cst.parse_statement(f"pl.intrin.yield_({args_code})")


# ---------------------------------------------------------------------------
# For-with-init_values → carry + bare yield + leak
# ---------------------------------------------------------------------------


def extract_carry_info(for_node: cst.For) -> tuple[str, list[str], list[str], str] | None:  # noqa: PLR0912
    """Return (loop_index_code, carry_vars, init_exprs, new_iter_code) or None."""
    call = for_node.iter
    if not is_range_like_call(call):
        return None
    init_val = get_init_values_kwarg(call)
    if init_val is None:
        return None
    init_codes: list[str] = []
    if isinstance(init_val, cst.Tuple):
        for el in init_val.elements:
            init_codes.append(code_for(el.value))
    else:
        init_codes.append(code_for(init_val))
    target = for_node.target
    loop_index_code = ""
    carry_vars: list[str] = []
    if isinstance(target, cst.Tuple):
        if len(target.elements) == 2:
            loop_index_code = code_for(target.elements[0].value)
            inner = target.elements[1].value
            if isinstance(inner, cst.Tuple):
                for el in inner.elements:
                    carry_vars.append(code_for(el.value))
            else:
                carry_vars.append(code_for(inner))
        elif len(target.elements) == 1:
            inner = target.elements[0].value
            if isinstance(inner, cst.Tuple):
                for el in inner.elements:
                    carry_vars.append(code_for(el.value))
            else:
                carry_vars.append(code_for(inner))
        else:
            for el in target.elements:
                carry_vars.append(code_for(el.value))
    else:
        carry_vars.append(code_for(target))
    filtered = [a for a in call.args if not (a.keyword and a.keyword.value == "init_values")]
    if filtered:
        filtered[-1] = filtered[-1].with_changes(comma=None)
        for idx in range(len(filtered) - 1):
            if filtered[idx].comma is None:
                filtered[idx] = filtered[idx].with_changes(
                    comma=cst.Comma(
                        whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace(" ")
                    )
                )
    new_iter_code = code_for(call.with_changes(args=filtered))
    return (loop_index_code, carry_vars, init_codes, new_iter_code)


def make_carry_stmt(carry_vars: list[str], init_exprs: list[str]) -> cst.BaseStatement:
    lhs = f"({carry_vars[0]},)" if len(carry_vars) == 1 else f"({', '.join(carry_vars)})"
    return cst.parse_statement(f"{lhs} = pl.intrin.carry({', '.join(init_exprs)})")


def make_leak_stmt(target_code: str) -> cst.BaseStatement:
    """Build ``(v,) = pl.intrin.leak()`` (1-tuple) or ``(a, b) = pl.intrin.leak()``
    (multi). The 1-tuple trailing comma is *semantically required* —
    ``(out)`` is a parenthesized single, not a tuple, and would bind one name
    to the seed tuple (the staged-adjacency guard then rejects it as an
    ordinary bind → EndFunction cascade)."""
    t = target_code.strip()
    if "," not in t:
        return cst.parse_statement(f"({t},) = pl.intrin.leak()")
    inner = t
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()  # peel one paren layer
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    if len(parts) == 1:
        lhs = f"({parts[0]},)"  # 1-tuple: keep the trailing comma
    else:
        lhs = "(" + ", ".join(parts) + ")"  # multi: no trailing comma
    return cst.parse_statement(f"{lhs} = pl.intrin.leak()")


# ---------------------------------------------------------------------------
# If-with-yield-branches → bare yields + leak
# ---------------------------------------------------------------------------


def _block_has_yield_assign(block: cst.BaseSuite) -> bool:
    for stmt in block.body:
        if isinstance(stmt, cst.SimpleStatementLine) and get_yield_assign_info(stmt):
            return True
    return False


def is_if_with_yield_branches(if_node: cst.If) -> bool:
    if not isinstance(if_node.body, cst.IndentedBlock) or not _block_has_yield_assign(if_node.body):
        return False
    orelse = if_node.orelse
    if not isinstance(orelse, cst.Else) or not isinstance(orelse.body, cst.IndentedBlock):
        return False
    return _block_has_yield_assign(orelse.body)


def extract_if_leak_target(if_node: cst.If) -> str | None:
    for s in if_node.body.body:
        if isinstance(s, cst.SimpleStatementLine):
            info = get_yield_assign_info(s)
            if info:
                return info[0]
    return None


# ---------------------------------------------------------------------------
# Decorator flip
# ---------------------------------------------------------------------------


def _flip_decorators(decorators: list[cst.Decorator]) -> list[cst.Decorator]:
    new: list[cst.Decorator] = []
    for deco in decorators:
        expr = deco.decorator
        # @pl.<name>(...)  or  @pl.<name>
        call: cst.Call | None = expr if isinstance(expr, cst.Call) else None
        attr = (
            call.func
            if call is not None and isinstance(call.func, cst.Attribute)
            else (expr if isinstance(expr, cst.Attribute) else None)
        )
        name = (
            attr.attr.value
            if attr is not None and isinstance(attr.value, cst.Name) and attr.value.value == "pl"
            else None
        )
        is_pl_fn = name in _DECORATOR_FLIP or name in ("function", "program")
        if not is_pl_fn or attr is None:
            new.append(deco)
            continue
        # Rename _parser → unsuffixed; the assembly dialect is tracing.
        renamed = (
            attr.with_changes(attr=cst.Name(_DECORATOR_FLIP[attr.attr.value]))
            if attr.attr.value in _DECORATOR_FLIP
            else attr
        )
        # Drop ``strict_ssa`` — the round-trip dialect's no-rebind guard does
        # not apply to the tracing assembly dialect (``pl.intrin`` binding
        # proxies register already-created IR variables without ``LetStmt``
        # or ``AssignStmt``, and the body's ordinary rebinds emit
        # ``AssignStmt`` updates that SSA promotion later resolves).
        if call is not None:
            kept = [a for a in call.args if not (a.keyword and a.keyword.value == "strict_ssa")]
            if kept:
                kept[-1] = kept[-1].with_changes(comma=None)
            new_expr = call.with_changes(func=renamed, args=kept)
        else:
            new_expr = renamed
        new.append(deco.with_changes(decorator=new_expr))
    return new


# ---------------------------------------------------------------------------
# The transform (identity-based change detection → lossless, idempotent)
# ---------------------------------------------------------------------------


def transform_block(  # noqa: PLR0912, PLR0915
    stmts: list[cst.BaseStatement],
) -> tuple[list[cst.BaseStatement], bool]:
    """Rewrite legacy init_values/yield_ → pl.intrin.* and flip decorators.

    Returns ``(new_stmts, changed)``. ``changed`` is identity-based: a node
    that was not rewritten passes through as the *same object*, so libcst
    preserves its comments/format exactly and the script is idempotent and
    lossless for untouched subtrees (a code_for-based comparison would
    reflow comments and over-flip IR-builder tests that only *mention*
    ``init_values``).
    """
    new_stmts: list[cst.BaseStatement] = []
    changed = False
    for stmt in stmts:
        # For over pl.while_(init_values=) + pl.cond → carry + while + leak
        if isinstance(stmt, cst.For) and is_while_call(stmt.iter):
            winfo = extract_while_carry_info(stmt)
            if winfo:
                target_names, init_codes, cond_code, remaining_body = winfo
                if cond_code is not None:
                    new_stmts.append(make_carry_stmt(target_names, init_codes))
                    transformed_body, _ = transform_block(remaining_body)
                    final_body: list[cst.BaseStatement] = []
                    wleak_target: str | None = None
                    for bstmt in transformed_body:
                        if isinstance(bstmt, cst.SimpleStatementLine):
                            yi = get_yield_assign_info(bstmt)
                            if yi:
                                wleak_target = yi[0]
                                final_body.append(convert_yield_assign_to_bare(bstmt))
                                continue
                        final_body.append(bstmt)
                    if wleak_target is None:
                        for orig_s in remaining_body:
                            if isinstance(orig_s, cst.SimpleStatementLine):
                                yi = get_yield_assign_info(orig_s)
                                if yi:
                                    wleak_target = yi[0]
                                    break
                    while_node = cst.parse_statement(f"while {cond_code}:\n    pass")
                    new_while = while_node.with_changes(
                        body=cst.IndentedBlock(body=final_body)
                        if final_body
                        else cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])])
                    )
                    new_stmts.append(new_while)
                    if wleak_target:
                        new_stmts.append(make_leak_stmt(wleak_target))
                    changed = True
                    continue
            # pl.while_ without cond/init_values: fall through to generic For
        # For with init_values → carry + bare yield + leak
        if isinstance(stmt, cst.For):
            info = extract_carry_info(stmt)
            if info:
                loop_index_code, carry_vars, init_exprs, new_iter_code = info
                new_stmts.append(make_carry_stmt(carry_vars, init_exprs))
                body_stmts = cast(
                    "list[cst.BaseStatement]",
                    list(stmt.body.body)
                    if isinstance(stmt.body, cst.IndentedBlock)
                    else list(stmt.body.body),
                )
                transformed_body, _ = transform_block(body_stmts)
                final_body: list[cst.BaseStatement] = []
                leak_target: str | None = None

                def _convert_yield_in_stmt(s: cst.BaseStatement) -> cst.BaseStatement:
                    nonlocal leak_target
                    if isinstance(s, cst.SimpleStatementLine):
                        yi = get_yield_assign_info(s)
                        if yi:
                            leak_target = yi[0]
                            return convert_yield_assign_to_bare(s)
                    if isinstance(s, cst.With) and isinstance(s.body, cst.IndentedBlock):
                        return s.with_changes(
                            body=cst.IndentedBlock(body=[_convert_yield_in_stmt(sub) for sub in s.body.body])
                        )
                    return s

                for bstmt in transformed_body:
                    final_body.append(_convert_yield_in_stmt(bstmt))
                if leak_target is None:
                    for orig_s in body_stmts:
                        if isinstance(orig_s, cst.SimpleStatementLine):
                            yi = get_yield_assign_info(orig_s)
                            if yi:
                                leak_target = yi[0]
                                break
                new_target = cst.parse_expression(loop_index_code) if loop_index_code else cst.Name("_")
                new_for = stmt.with_changes(
                    target=new_target,
                    iter=cst.parse_expression(new_iter_code),
                    body=cst.IndentedBlock(body=final_body) if final_body else stmt.body,
                )
                new_stmts.append(new_for)
                if leak_target:
                    new_stmts.append(make_leak_stmt(leak_target))
                changed = True
                continue
            # For without init_values — recurse, rebuild only if changed
            if isinstance(stmt.body, cst.IndentedBlock):
                new_body_stmts, body_changed = transform_block(
                    cast("list[cst.BaseStatement]", list(stmt.body.body))
                )
                if body_changed:
                    new_stmts.append(stmt.with_changes(body=cst.IndentedBlock(body=new_body_stmts)))
                    changed = True
                else:
                    new_stmts.append(stmt)
            else:
                new_stmts.append(stmt)
            continue
        # If-with-yield-branches → bare yields + leak. Recurse through each
        # branch as a statement list so nested transforms retain the leak
        # statements they insert; keeping only sub[0] drops those leaks and
        # leaves downstream phi names undefined.
        if isinstance(stmt, cst.If) and is_if_with_yield_branches(stmt):
            leak_target = extract_if_leak_target(stmt)
            then_body_stmts = list(stmt.body.body) if isinstance(stmt.body, cst.IndentedBlock) else []
            else_node = stmt.orelse
            else_body_stmts = (
                list(else_node.body.body)
                if isinstance(else_node, cst.Else) and isinstance(else_node.body, cst.IndentedBlock)
                else []
            )

            def _conv_branch(stmts: list[cst.BaseStatement]) -> list[cst.BaseStatement]:
                converted: list[cst.BaseStatement] = []
                for branch_stmt in stmts:
                    if isinstance(branch_stmt, cst.SimpleStatementLine) and get_yield_assign_info(
                        branch_stmt
                    ):
                        converted.append(convert_yield_assign_to_bare(branch_stmt))
                        continue
                    transformed, _ = transform_block([branch_stmt])
                    converted.extend(transformed)
                return converted

            new_then = _conv_branch(then_body_stmts)
            new_else = _conv_branch(else_body_stmts)
            new_if = stmt.with_changes(
                body=cst.IndentedBlock(body=new_then),
                orelse=cst.Else(body=cst.IndentedBlock(body=new_else))
                if isinstance(else_node, cst.Else)
                else else_node,
            )
            new_stmts.append(new_if)
            if leak_target:
                new_stmts.append(make_leak_stmt(leak_target))
            changed = True
            continue
        # Generic recursion — rebuild only if a subtree changed (lossless)
        if isinstance(stmt, cst.If):
            new_then_body, then_changed = (
                transform_block(cast("list[cst.BaseStatement]", list(stmt.body.body)))
                if isinstance(stmt.body, cst.IndentedBlock)
                else ([], False)
            )
            new_orelse = stmt.orelse
            else_changed = False
            if isinstance(new_orelse, cst.Else) and isinstance(new_orelse.body, cst.IndentedBlock):
                eb, else_changed = transform_block(
                    cast("list[cst.BaseStatement]", list(new_orelse.body.body))
                )
                new_orelse = new_orelse.with_changes(body=cst.IndentedBlock(body=eb))
            elif isinstance(new_orelse, cst.If):
                eb, else_changed = transform_block([new_orelse])
                new_orelse = eb[0]
            if then_changed or else_changed:
                new_body = (
                    cst.IndentedBlock(body=new_then_body)
                    if isinstance(stmt.body, cst.IndentedBlock)
                    else stmt.body
                )
                new_stmts.append(stmt.with_changes(body=new_body, orelse=new_orelse))
                changed = True
            else:
                new_stmts.append(stmt)
        elif isinstance(stmt, (cst.With, cst.While)) and isinstance(stmt.body, cst.IndentedBlock):
            new_body_stmts, body_changed = transform_block(
                cast("list[cst.BaseStatement]", list(stmt.body.body))
            )
            if body_changed:
                new_stmts.append(stmt.with_changes(body=cst.IndentedBlock(body=new_body_stmts)))
                changed = True
            else:
                new_stmts.append(stmt)
        elif isinstance(stmt, cst.FunctionDef):
            new_body_stmts, body_changed = transform_block(
                cast("list[cst.BaseStatement]", list(stmt.body.body))
            )
            if body_changed:
                flipped = _flip_decorators(list(stmt.decorators))
                new_stmts.append(
                    stmt.with_changes(decorators=flipped, body=cst.IndentedBlock(body=new_body_stmts))
                )
                changed = True
            else:
                new_stmts.append(stmt)
        elif isinstance(stmt, cst.ClassDef) and isinstance(stmt.body, cst.IndentedBlock):
            new_body_stmts, body_changed = transform_block(
                cast("list[cst.BaseStatement]", list(stmt.body.body))
            )
            if body_changed:
                flipped = _flip_decorators(list(stmt.decorators))
                new_stmts.append(
                    stmt.with_changes(decorators=flipped, body=cst.IndentedBlock(body=new_body_stmts))
                )
                changed = True
            else:
                new_stmts.append(stmt)
        elif isinstance(stmt, cst.Try):
            # ``try:/finally:`` blocks wrap backend-setup fixtures
            # (``backend.reset_for_testing()`` + ``try:``). Without this
            # branch the kernel inside the ``try`` body is never reached
            # and stays on the legacy ``init_values`` spelling.
            new_body, body_changed = (
                transform_block(cast("list[cst.BaseStatement]", list(stmt.body.body)))
                if isinstance(stmt.body, cst.IndentedBlock)
                else ([], False)
            )
            new_handlers = stmt.handlers
            handlers_changed = False
            rebuilt_handlers = []
            for h in stmt.handlers:
                if isinstance(h.body, cst.IndentedBlock):
                    hb, hc = transform_block(list(h.body.body))
                    if hc:
                        rebuilt_handlers.append(h.with_changes(body=cst.IndentedBlock(body=hb)))
                        handlers_changed = True
                    else:
                        rebuilt_handlers.append(h)
                else:
                    rebuilt_handlers.append(h)
            if handlers_changed:
                new_handlers = rebuilt_handlers
            new_orelse = stmt.orelse
            orelse_changed = False
            if isinstance(new_orelse, cst.Else) and isinstance(new_orelse.body, cst.IndentedBlock):
                eb, orelse_changed = transform_block(
                    cast("list[cst.BaseStatement]", list(new_orelse.body.body))
                )
                new_orelse = new_orelse.with_changes(body=cst.IndentedBlock(body=eb))
            new_finalbody = stmt.finalbody
            finalbody_changed = False
            if isinstance(new_finalbody, cst.Finally) and isinstance(new_finalbody.body, cst.IndentedBlock):
                fb, finalbody_changed = transform_block(list(new_finalbody.body.body))
                new_finalbody = new_finalbody.with_changes(body=cst.IndentedBlock(body=fb))
            if body_changed or handlers_changed or orelse_changed or finalbody_changed:
                new_body_block = (
                    cst.IndentedBlock(body=new_body)
                    if isinstance(stmt.body, cst.IndentedBlock)
                    else stmt.body
                )
                new_stmts.append(
                    stmt.with_changes(
                        body=new_body_block,
                        handlers=new_handlers,
                        orelse=new_orelse,
                        finalbody=new_finalbody,
                    )
                )
                changed = True
            else:
                new_stmts.append(stmt)
        else:
            new_stmts.append(stmt)
    return new_stmts, changed


def migrate_file(path: str) -> bool:
    src = pathlib.Path(path).read_text()
    mod = cst.parse_module(src)
    new_body, changed = transform_block(list(mod.body))
    if not changed:
        print(f"no change {path}")
        return False
    new_code = mod.with_changes(body=new_body).code
    if new_code != src:
        pathlib.Path(path).write_text(new_code)
        print(f"migrated {path}")
        return True
    print(f"no change {path}")
    return False


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        print("usage: migrate_intrin.py <files>")
        sys.exit(1)
    for f in files:
        migrate_file(f)
