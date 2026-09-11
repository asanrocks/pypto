# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The protocol between the language-agnostic collector and a language frontend.

The surface splits along natural lifetimes:

- **Sema** is one function-trace compilation. It owns one `IRBuilder` and
  the trace result, and creates one fresh `Frame` per trace
  (``begin_frame``/``result``). A mutable `Sema` is one-shot: reuse after
  success or failure is rejected, never reset.
- **Frame** is one function tracing context. Interpreter events that need
  language semantics route through Frame methods (binding,
  expression_statement, return_, the ``create_*_construct`` constructors, and
  the logical value algebra).
- **Construct** (defined in :mod:`collector.interpreter` together with its
  host kinds) is one active construct. Frames create
  language-specific Constructs; the trace interpreter owns their stack order
  and lifetime.

Lifecycle-sensitive ``None``/boolean combinations are replaced by closed
result types: bind results distinguish decline, ordinary binding, result
binding progress, and exact completion; close results distinguish finalized
closes from pending result constructs that stay on the interpreter stack
until result binding removes them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from .interpreter import Construct, ForConstruct, IfConstruct, Interpreter, WhileConstruct, WithConstruct

V = TypeVar("V")
# Covariant: the binding-result records are frozen value carriers, so a
# ``StagedBinding[Marker]`` is a ``StagedBinding[object]`` — a Frame over the
# object domain (the NullFrame extension pattern) can report its precise
# staged values where ``BindingResult[object]`` is expected.
V_co = TypeVar("V_co", covariant=True)

# ---------------------------------------------------------------------------
# Typed lifecycle results — no multi-meaning None/bool
# ---------------------------------------------------------------------------


class BranchDisposition(Enum):
    """What the trace interpreter does with a staged if's branch mappings at close."""

    COMMIT = "commit"
    DISCARD = "discard"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class Closed:
    """The active construct is done; the trace interpreter removes it from its stack."""

    branches: BranchDisposition = BranchDisposition.NOT_APPLICABLE


@dataclass(frozen=True)
class AwaitingResultBindings:
    """A result-producing close: the construct stays as the interpreter's exact top
    pending result construct until result binding finishes it."""

    branches: BranchDisposition


CloseResult = Closed | AwaitingResultBindings


@dataclass(frozen=True)
class HostBinding:
    """The Frame declined; the trace interpreter stores the original host value."""


@dataclass(frozen=True)
class StagedBinding(Generic[V_co]):
    """An ordinary staged binding; the trace interpreter stores the value representation."""

    value: V_co


@dataclass(frozen=True)
class PartialResultBinding(Generic[V]):
    """A result binding that advanced but did not finish the pending result construct."""

    construct: Construct[V]
    value: V


@dataclass(frozen=True)
class FinalResultBinding(Generic[V]):
    """The result binding that finished the pending result construct; the trace
    interpreter removes the construct after storing the value."""

    construct: Construct[V]
    value: V


# Generic in the Frame's V result domain: a Frame[V] produces these results
# over V. (A bare use is the Any-domain spelling.)
BindingResult = HostBinding | StagedBinding[V] | PartialResultBinding[V] | FinalResultBinding[V]


@dataclass(frozen=True)
class ForOpen(Generic[V]):
    """A staged ``for`` opening: the exact construct plus its loop-target binding proxy."""

    construct: ForConstruct[V]
    binding: object


@dataclass(frozen=True)
class WithOpen(Generic[V]):
    """A staged ``with`` opening: the exact construct plus its binding proxy."""

    construct: WithConstruct[V]
    binding: object


# ---------------------------------------------------------------------------
# Frame — the function-level semantic interface
# ---------------------------------------------------------------------------


class Frame(Protocol[V]):
    """One function tracing context.

    The trace interpreter calls these methods and nothing else. For eagerly
    classified constructs the ``create_*_construct`` constructors return
    ``None`` so the interpreter installs its host-kind Construct; a returned
    Construct is trusted and installed exactly. ``note_interpreter`` records the
    interpreter association (installed once at frame entry and used by named
    stdlib operations). ``finish`` runs once on normal completion and stores
    the result on the Sema.

    ``V`` is the Frame's staged value representation — the type of value that
    comes back after the interpreter hands it an input. Host values, binding
    proxies, and trace-protocol objects cross this boundary as ``object``;
    only produced staged values are ``V``. The logical-value algebra takes
    ``object`` operands (a host-literal right-hand side is legal and lifted at
    the operand boundary) and produces a ``V``.
    """

    def note_interpreter(self, interpreter: Interpreter[V], /) -> None: ...
    def is_staged(self, value: object) -> bool: ...
    def add_leading_comments(self, comments: Sequence[str]) -> None: ...

    def bind(
        self, interpreter: Interpreter[V], /, name: str, value: object, span: object | None = None
    ) -> BindingResult[V]: ...
    def expression_statement(
        self, interpreter: Interpreter[V], /, value: object, span: object | None = None
    ) -> None: ...
    def return_(
        self, interpreter: Interpreter[V], /, values: Sequence[object] | None, span: object | None = None
    ) -> None: ...

    def create_if_construct(
        self, interpreter: Interpreter[V], /, predicate: object, span: object | None = None
    ) -> IfConstruct[V] | None: ...
    def create_for_construct(
        self, interpreter: Interpreter[V], /, iterable: object, span: object | None = None
    ) -> ForOpen[V] | None: ...
    def create_while_construct(
        self,
        interpreter: Interpreter[V],
        /,
        predicate_thunk: Callable[[], object],
        span: object | None = None,
    ) -> WhileConstruct[V] | None: ...
    def create_with_construct(
        self,
        interpreter: Interpreter[V],
        /,
        context: object,
        name: str | None = None,
        span: object | None = None,
        single_call_body: bool = False,
    ) -> WithOpen[V] | None: ...

    def logical_and(self, a: object, b: object, span: object | None = None) -> V: ...
    def logical_or(self, a: object, b: object, span: object | None = None) -> V: ...
    def logical_not(self, a: object, span: object | None = None) -> V: ...

    def resolve_local_name(self, expr: object) -> str | None: ...

    def finish(self) -> object: ...


# ---------------------------------------------------------------------------
# Sema — one function-trace compilation
# ---------------------------------------------------------------------------


class Sema(Protocol[V]):
    """One function-trace compilation.

    ``begin_frame`` succeeds once; a mutable Sema is never reset. The
    trace driver retains a :data:`SemaFactory` (or immutable
    configuration) and acquires a fresh Sema per independent trace.
    """

    def begin_frame(self) -> Frame[V]: ...
    def note_span(self, source_file: str, lineno: int, col: int) -> None: ...
    def abort(self) -> None: ...
    def result(self) -> object: ...


SemaFactory = Callable[[], Sema[V]]


class NullFrame:
    """The host-execution Frame (the ``object`` value domain): every
    staged operation declines.

    Not generic — host execution never produces a staged value, so the
    ``object`` domain is the whole contract. A recording/test Frame subclasses
    this and overrides the ``create_*_construct`` constructors and/or ``bind``
    (the declared ``BindingResult[object]`` return is the extension surface).
    """

    def note_interpreter(self, interpreter: Interpreter[object], /) -> None:
        del interpreter

    def is_staged(self, value: object) -> bool:
        del value
        return False

    def add_leading_comments(self, comments: Sequence[str]) -> None:
        del comments

    def bind(
        self, interpreter: Interpreter[object], /, name: str, value: object, span: object | None = None
    ) -> BindingResult[object]:
        del interpreter, name, value, span
        return HostBinding()

    def expression_statement(
        self, interpreter: Interpreter[object], /, value: object, span: object | None = None
    ) -> None:
        del interpreter, value, span

    def return_(
        self, interpreter: Interpreter[object], /, values: Sequence[object] | None, span: object | None = None
    ) -> None:
        del interpreter, values, span

    def create_if_construct(
        self, interpreter: Interpreter[object], /, predicate: object, span: object | None = None
    ) -> IfConstruct[object] | None:
        del interpreter, predicate, span
        return None

    def create_for_construct(
        self, interpreter: Interpreter[object], /, iterable: object, span: object | None = None
    ) -> ForOpen[object] | None:
        del interpreter, iterable, span
        return None

    def create_while_construct(
        self,
        interpreter: Interpreter[object],
        /,
        predicate_thunk: Callable[[], object],
        span: object | None = None,
    ) -> WhileConstruct[object] | None:
        del interpreter, predicate_thunk, span
        return None

    def create_with_construct(
        self,
        interpreter: Interpreter[object],
        /,
        context: object,
        name: str | None = None,
        span: object | None = None,
        single_call_body: bool = False,
    ) -> WithOpen[object] | None:
        del interpreter, context, name, span, single_call_body
        return None

    def logical_and(self, a: object, b: object, span: object | None = None) -> object:
        del span
        return a and b

    def logical_or(self, a: object, b: object, span: object | None = None) -> object:
        del span
        return a or b

    def logical_not(self, a: object, span: object | None = None) -> object:
        del span
        return not a

    def resolve_local_name(self, expr: object) -> str | None:
        del expr
        return None

    def finish(self) -> None:
        return None


class NullSema:
    """The host-execution Sema (the ``object`` value domain): fresh no-op
    frames and no trace result.

    Stateless and therefore safely reusable across invocations, unlike a
    mutable language Sema, which is one-shot by contract.
    """

    def begin_frame(self) -> NullFrame:
        return NullFrame()

    def note_span(self, source_file: str, lineno: int, col: int) -> None:
        del source_file, lineno, col

    def abort(self) -> None:
        return None

    def result(self) -> None:
        return None


# Keep this boundary independent of the language frontend.
__all__ = [
    "AwaitingResultBindings",
    "BindingResult",
    "BranchDisposition",
    "Closed",
    "CloseResult",
    "FinalResultBinding",
    "ForOpen",
    "Frame",
    "HostBinding",
    "NullFrame",
    "NullSema",
    "PartialResultBinding",
    "Sema",
    "SemaFactory",
    "StagedBinding",
    "WithOpen",
]
