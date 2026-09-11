# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The conformance-anchor harness: the acceptance gate holding the
tracing and parser frontends to identical IR.

One ``assert_structural_equal(pl.function(k), pl.function_parser(k))`` per
construct. Two regimes, by design:

- **Raw** (straight-line): the tracing frontend's ``bind`` emits
  the same ``let``-on-first / ``assign``-on-rebind as the parser frontend's
  ``parse_assignment``/``parse_annotated_assignment`` (any
  divergence is a ``bind`` bug, not a pass artifact).
- **Post-canonical** (staged control flow): the tracing frontend emits *less* than the
  parser frontend — no ``iter_arg``/``return_var``/explicit phi. The SSA pass
  synthesizes them (loop-carried values from self-referential ``AssignStmt``
  updates; branch phi from mutable locals bound in both
  branches, converging through SSA promotion).
  Both sides run ``canonicalize()`` before comparison.

``assert_structural_equal`` (the ``pypto_core`` extension) ignores source
location (Span) — the anchors need no span work (pinned by
``test_span_policy``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pypto.language as pl  # noqa: F401 - the AST run's expression-resolution scope
from pypto.language.parser import function_parser, program_parser
from pypto.pypto_core import ir, passes

__all__ = ["canonicalize", "canonicalize_program", "conformance", "program_conformance"]


def canonicalize_program(prog: ir.Program) -> ir.Program:
    """Run the canonical dataflow pair over a whole Program.

    ``pypto_core.passes``' ``convert_to_ssa()`` then ``simplify()`` — the
    load-bearing dataflow pair. The Program-level analog of
    ``canonicalize``: the staged-control-flow regime for the multi-function anchors
    — every function in the program is canonicalized together,
    so cross-function call sites see the same pass state as a real compile.
    """
    prog = passes.convert_to_ssa()(prog)
    return passes.simplify()(prog)


def canonicalize(
    fn: ir.Function,
    *,
    inline: bool = False,
    callees: Sequence[ir.Function] = (),
) -> ir.Function:
    """Run the canonical dataflow passes over the function's program.

    ``pypto_core.passes``' ``convert_to_ssa()`` then ``simplify()`` — the
    load-bearing dataflow pair: SSA promotion synthesizes phi and
    loop-carried values from mutable ``ir.Var``s updated by ``AssignStmt``;
    ``simplify`` folds the
    copies). Passes run over a ``Program``; the single function is extracted back.

    The inline regime: when ``inline=True``, the function
    is first packaged with its inline callees (``callees``) into a Program
    and ``inline_functions`` — the first pipeline pass — splices the
    callees' bodies at the call sites, before the canonical pair runs.
    The tracing frontend's caller goes
    through the pass, the AST side is already parse-time-spliced (its
    ``_parse_inline_call``), and the comparison happens post-canonical.
    """
    if inline:
        program = ir.Program([fn, *callees], f"inline_{fn.name}", ir.Span.unknown())
        program = passes.inline_functions()(program)
        extracted = program[fn.name]
        if extracted is None:
            raise RuntimeError(
                f"inline regime: caller {fn.name!r} vanished after inline_functions "
                "— an Inline callee must never absorb its caller"
            )
        fn = extracted
    program = ir.Program([fn], f"canonicalize_{fn.name}", ir.Span.unknown())
    program = passes.convert_to_ssa()(program)
    program = passes.simplify()(program)
    (result,) = program.functions.values()
    return result


def conformance(
    kernel: Callable[..., Any],
    build: Callable[[Callable[..., Any]], ir.Function],
    *,
    canonical: bool = False,
) -> None:
    """Assert the tracing frontend's ``build(kernel)`` matches the AST
    frontend's ``function_parser(kernel)`` — raw equality for straight-line
    kernels, post-canonical equality for staged-control-flow kernels.

    Args:
        kernel: The decorated kernel (both frontends parse the same source).
        build: The tracing frontend's entry.
        canonical: Assert post-canonical equality (staged-control-flow regime).
    """
    new = build(kernel)
    ast_ = function_parser(kernel)
    if canonical:
        new = canonicalize(new)
        ast_ = canonicalize(ast_)
    # auto-mapping: each frontend builds its own Var objects; the anchors
    # compare shape + names, never object identity.
    ir.assert_structural_equal(new, ast_, enable_auto_mapping=True)


def program_conformance(
    cls: type,
    build: Callable[[type], ir.Program],
    *,
    canonical: bool = False,
) -> None:
    """Assert the tracing frontend's ``build(cls)`` matches the parser frontend's
    ``program_parser(cls)`` — the Program-level analog of ``conformance``:
    raw equality for straight-line programs, post-canonical
    equality for staged-control-flow programs.

    Args:
        cls: The undecorated program class (both frontends parse the same
            source; the methods carry ``@pl.function`` marks that ``program_parser``
            detects by name).
        build: The tracing frontend's program entry.
        canonical: Assert post-canonical equality (staged-control-flow regime).

    The shape is pinned by the program probes:
    ``assert_structural_equal`` works directly on ``ir.Program`` with
    auto-mapping — no per-function-by-name walk needed.
    """
    new = build(cls)
    ast_ = program_parser(cls)
    if canonical:
        new = canonicalize_program(new)
        ast_ = canonicalize_program(ast_)
    ir.assert_structural_equal(new, ast_, enable_auto_mapping=True)
