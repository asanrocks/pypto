# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The expansion tier's acceptance tests — ``current_trace()`` and ``pl.expand``.

Pins the borrowing-expansion contract: the capability is opaque and
invocation-scoped, the expanded body's IR lands in the caller's function
(no function, ``GlobalVar``, or ``ir.Call`` is emitted for the expansion),
host side effects run once per occurrence, failure discards the complete
parent trace, early returns under staged control flow are loud, and
recursion gets the expansion-stack diagnostic.
"""

import dataclasses

import pypto.language as pl
import pytest
from pypto.language import entrance
from pypto.language.entrance.annotation import derive
from pypto.pypto_core import ir


def _structural_equal(left, right) -> bool:
    return ir.structural_equal(left, right, enable_auto_mapping=True)


# ---------------------------------------------------------------------------
# current_trace
# ---------------------------------------------------------------------------


class TestCurrentTrace:
    def test_none_on_host(self):
        assert entrance.current_trace() is None

    def test_capability_during_collection(self):
        captured: dict = {}

        def probe(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
            captured["trace"] = entrance.current_trace()
            return x

        @pl.function
        def traced(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
            return probe(x)

        # ``traced`` eager-traces at decoration: the body ran with a live
        # trace, so the probe captured a capability, not None.
        assert captured["trace"] is not None

    def test_capability_is_stale_after_invocation(self):
        capabilities = []

        def probe(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
            capabilities.append(entrance.current_trace())
            return x

        @pl.function
        def traced(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
            return probe(x)

        del traced
        assert len(capabilities) == 1
        capability = capabilities[0]
        assert capability is not None
        body = entrance.TracedCallable.from_callable(probe)
        with pytest.raises(RuntimeError, match="stale"):
            capability.expand(body, args=(1,))

    def test_expand_rejects_param_placeholder(self):
        def kernel(o: pl.Out[pl.Tensor[[4], pl.FP32]]) -> pl.Tensor[[4], pl.FP32]:
            param = entrance.Param("o", o.unwrap().type, direction=ir.ParamDirection.Out)
            body = entrance.TracedCallable.from_callable(identity)
            trace = entrance.current_trace()
            assert trace is not None
            trace.expand(body, args=(param,))
            return o

        def identity(x):
            return x

        with pytest.raises(TypeError, match="Param placeholder"):
            derive_and_invoke(kernel)


def derive_and_invoke(kernel):
    plan = derive(kernel)
    body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
    return body.invoke(args=plan.args, kwargs=plan.kwargs)


# ---------------------------------------------------------------------------
# pl.expand
# ---------------------------------------------------------------------------


class TestPlExpand:
    def test_expand_body_lands_in_caller(self):
        """The §7 shape: the expanded body's ops are emitted into the caller's
        function; the expansion itself emits no function, GlobalVar, or Call."""

        @pl.expand
        def scaled_add(x, y, scale):
            return pl.add(x, pl.mul(y, scale))

        @pl.function
        def kernel(
            a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]
        ):
            ta = pl.load(a, [0], [4])
            tb = pl.load(b, [0], [4])
            r = scaled_add(ta, tb, 2.0)
            pl.store(r, [0, 0], o)
            return o

        @pl.function
        def reference(
            a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]
        ):
            ta = pl.load(a, [0], [4])
            tb = pl.load(b, [0], [4])
            r = pl.add(ta, pl.mul(tb, 2.0))
            pl.store(r, [0, 0], o)
            return o

        assert _structural_equal(kernel, reference)
        assert "scaled_add" not in kernel.as_python()

    def test_plain_helper_frame_does_not_bind_but_expansion_frame_does(self):
        """Frame depth decides the traced statement shape (dsv4 corpus finding).

        Ops written in the expansion body's own frame become named bindings in
        the caller — an expansion returning a pack of its locals is structurally
        equal to inlining those statements. Ops emitted by a *plain* helper
        called from inside the expansion body cross the boundary as expression
        values: no binding is created and the caller nests them. A traced
        function calling the same plain helper directly keeps the call-site
        binding, so extraction under an IR-equality contract must account for
        this: shared op-emitting helpers are expansions called by the traced
        function itself, never plain helpers inside an expansion.
        """
        D, N = 128, 4

        @dataclasses.dataclass(frozen=True)
        class Pack:
            a: object

        def _plain(s, layer):
            a = pl.slice(s, [D], [layer * D])
            return Pack(a)

        @pl.expand
        def _via_plain(s, layer):
            return _plain(s, layer)

        @pl.expand
        def _inline(s, layer):
            a = pl.slice(s, [D], [layer * D])
            return Pack(a)

        def make(expansion):
            @pl.jit
            def kernel(w: pl.Tensor[[N * D], pl.FP32], y: pl.Out[pl.Tensor[[D], pl.FP32]]):
                v = expansion(w, 2)
                y[:] = v.a

            return kernel

        @pl.jit
        def bound(w: pl.Tensor[[N * D], pl.FP32], y: pl.Out[pl.Tensor[[D], pl.FP32]]):
            a = pl.slice(w, [D], [2 * D])
            y[:] = a

        @pl.jit
        def nested(w: pl.Tensor[[N * D], pl.FP32], y: pl.Out[pl.Tensor[[D], pl.FP32]]):
            y[:] = pl.slice(w, [D], [2 * D])

        assert _structural_equal(make(_inline), bound)
        assert _structural_equal(make(_via_plain), nested)
        assert not _structural_equal(make(_via_plain), bound)

    def test_break_targets_the_expansion_body_loop(self):
        """Control inside the expansion body targets the body's own constructs
        — a break binds to the body's loop, never to a caller construct."""

        @pl.expand
        def sum_first(x, n):
            acc = pl.mul(x, 0.0)
            for i in pl.range(n):
                acc = pl.add(acc, pl.mul(x, 2.0))
                if pl.const(True, pl.BOOL):
                    break
            return acc

        def kernel(o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(o, [0], [4])
            pl.store(sum_first(t, 4), [0, 0], o)
            return o

        fn = derive_and_invoke(kernel)
        text = fn.as_python()
        assert "for i in pl.range(4)" in text
        assert "break" in text

    def test_default_arguments_survive_preparation(self):
        @pl.expand
        def scaled(x, scale=3.0):
            return pl.mul(x, scale)

        def kernel(o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(o, [0], [4])
            pl.store(scaled(t), [0, 0], o)
            return o

        fn = derive_and_invoke(kernel)
        assert "pl.tile.muls" in fn.as_python()

    def test_expansion_under_caller_staged_control_flow(self):
        """IR placement follows the caller's current insertion point: an
        expansion inside a staged loop/if lands its ops inside that region."""

        @pl.expand
        def scaled(x, scale):
            return pl.mul(x, scale)

        def looped(o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(o, [0], [4])
            for i in pl.range(4):
                t = scaled(t, 2.0)
            pl.store(t, [0, 0], o)
            return o

        fn = derive_and_invoke(looped)
        text = fn.as_python()
        assert text.index("for i in pl.range(4)") < text.index("pl.tile.muls")

        def branched(o: pl.Out[pl.Tensor[[4], pl.FP32]], flag: pl.Scalar[pl.BOOL]):
            t = pl.load(o, [0], [4])
            if flag:
                t = scaled(t, 3.0)
            pl.store(t, [0, 0], o)
            return o

        fn2 = derive_and_invoke(branched)
        text2 = fn2.as_python()
        assert text2.index("if flag:") < text2.index("pl.tile.muls")

    def test_host_call_is_rejected(self):
        @pl.expand
        def body(x):
            return x

        with pytest.raises(TypeError, match="requires an active trace"):
            body(1)

    def test_body_executes_once_per_occurrence(self):
        effects: list[int] = []

        @pl.expand
        def noisy(x):
            effects.append(1)
            return x

        def kernel(o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(o, [0], [4])
            a = noisy(t)
            b = noisy(t)
            pl.store(pl.add(a, b), [0, 0], o)
            return o

        derive_and_invoke(kernel)
        assert effects == [1, 1]
        derive_and_invoke(kernel)
        assert effects == [1, 1, 1, 1]

    def test_closure_and_globals_stay_live(self):
        closure_cell = {"scale": 2.0}

        @pl.expand
        def scaled(x):
            return pl.mul(x, closure_cell["scale"])

        def kernel(o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(o, [0], [4])
            pl.store(scaled(t), [0, 0], o)
            return o

        first = derive_and_invoke(kernel)
        closure_cell["scale"] = 3.0
        second = derive_and_invoke(kernel)
        assert not _structural_equal(first, second)

    def test_expansion_failure_discards_parent_trace(self):
        sentinel = ValueError("expansion boom")

        @pl.expand
        def boom(x):
            raise sentinel

        def kernel(o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(o, [0], [4])
            pl.store(boom(t), [0, 0], o)
            return o

        with pytest.raises(ValueError) as caught:
            derive_and_invoke(kernel)
        assert caught.value is sentinel

    def test_early_return_under_staged_control_flow_is_loud(self):
        @pl.expand
        def guarded(x, flag):
            if flag:
                return x
            return x

        def kernel(x: pl.Scalar[pl.BOOL], o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(o, [0], [4])
            pl.store(guarded(t, x), [0, 0], o)
            return o

        with pytest.raises(TypeError, match="one trailing return"):
            derive_and_invoke(kernel)

    def test_recursion_gets_expansion_stack_diagnostic(self):
        @pl.expand
        def recursive(x):
            return recursive(x)

        def kernel(o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(o, [0], [4])
            pl.store(recursive(t), [0, 0], o)
            return o

        with pytest.raises(TypeError, match="expansion of 'recursive' is already active"):
            derive_and_invoke(kernel)

    def test_host_branch_early_return_is_host_semantics(self):
        """A host ``if`` around an early return is host control: it takes the
        native path and stays legal (only staged control flow is rejected)."""

        @pl.expand
        def pick(x, y, which):
            if which:
                return x
            return y

        def kernel(o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(o, [0], [4])
            pl.store(pick(t, t, False), [0, 0], o)
            return o

        fn = derive_and_invoke(kernel)
        assert "pl.tile.add" in fn.as_python() or "pl.tile.load" in fn.as_python()


class TestExpansionWriteThrough:
    """ADR-0001: item-assignment on a substituted parameter writes the
    caller's binding — value-level write-through, never scope inheritance."""

    def test_write_through_lands_as_inlined_assignment(self):
        """The strongest witness: the expansion's store is structurally the
        caller's own inlined item-assignment — same statements, same target
        binding — exactly what the jit-inline + inlining pass produced."""

        @pl.expand
        def write_row(y, x):
            y[:] = pl.slice(x, [4], [0])

        @pl.function
        def kernel(x: pl.Tensor[[8], pl.FP32], y: pl.Out[pl.Tensor[[4], pl.FP32]]):
            write_row(y, x)
            return y

        @pl.function
        def reference(x: pl.Tensor[[8], pl.FP32], y: pl.Out[pl.Tensor[[4], pl.FP32]]):
            y[:] = pl.slice(x, [4], [0])
            return y

        assert _structural_equal(kernel, reference)

    def test_write_through_from_staged_branch(self):
        """A write-through inside the body's staged ``if`` assigns through the
        caller's branch-visible binding — textual-inlining placement."""

        @pl.expand
        def cond_write(y, x, flag):
            if flag:
                y[:] = pl.slice(x, [4], [0])

        @pl.function
        def kernel(
            x: pl.Tensor[[8], pl.FP32],
            y: pl.Out[pl.Tensor[[4], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
        ) -> pl.Tensor[[4], pl.FP32]:
            cond_write(y, x, flag)
            return y

        @pl.function
        def reference(
            x: pl.Tensor[[8], pl.FP32],
            y: pl.Out[pl.Tensor[[4], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
        ) -> pl.Tensor[[4], pl.FP32]:
            if flag:
                y[:] = pl.slice(x, [4], [0])
            return y

        assert _structural_equal(kernel, reference)

    def test_plain_rebind_shadows_not_leaks(self):
        """A plain rebinding of the parameter name creates a body-local — the
        caller's binding is untouched (scope inheritance would fail this).

        The trailing return crosses as an alias binding (``r = <body-local>``) —
        the known seam-2 fact the real pipeline's Simplify folds — so the
        witness is the traced shape, not pre-pass structural equality.
        """

        @pl.expand
        def shadow(o, t):
            o = pl.add(t, t)
            return o

        @pl.function
        def kernel(a: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(a, [0], [4])
            r = shadow(o, t)
            pl.store(r, [0, 0], o)
            return o

        text = kernel.as_python()
        # The rebind landed as a body-local (``o_1``), not a caller rebind:
        # the caller's ``o`` still names the parameter, so the store targets
        # it — a leaked rebind would have rebound ``o`` to a Tile and the
        # store would have failed loudly at trace time.
        assert "o_1: pl.Tile" in text
        assert "pl.tile.store(r, [0, 0], o)" in text

    def test_expression_backed_target_stays_loud(self):
        """A target that is no frame's local — not the expansion's, not the
        borrowed parent's — keeps the loud item-assignment error."""

        def _detached(x):
            return pl.slice(x, [4], [4])

        @pl.expand
        def bad(x):
            v = pl.slice(x, [4], [0])
            _detached(x)[:] = v

        def kernel(x: pl.Tensor[[8], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            bad(x)
            return o

        with pytest.raises(TypeError, match="must be a bound local name"):
            derive_and_invoke(kernel)

    def test_write_through_during_pending_result_is_loud(self):
        """ADR-0001's guard: the owner's bind path cannot see the expansion
        body's own pending result construct, so the write-through consults
        the requester's interpreter and rejects."""

        @pl.expand
        def slip(y, x, flag):
            t0 = pl.load(x, [0], [4])
            t1 = pl.load(x, [4], [4])
            if flag:
                pl.intrin.yield_(t0)
            else:
                pl.intrin.yield_(t1)
            y[:] = pl.slice(x, [4], [4])

        def kernel(
            x: pl.Tensor[[8], pl.FP32],
            y: pl.Out[pl.Tensor[[4], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
        ) -> pl.Tensor[[4], pl.FP32]:
            slip(y, x, flag)
            return y

        with pytest.raises(TypeError, match="awaits result bindings"):
            derive_and_invoke(kernel)


# ---------------------------------------------------------------------------
# The capability's expand, driven directly
# ---------------------------------------------------------------------------


class TestCapabilityExpand:
    def test_user_expansion_wrapper(self):
        """The §4.4 litmus: a user-written expansion wrapper needs only the
        opaque capability — no Sema, Frame, or collector imports."""

        def scaled_add(x, y, scale):
            return pl.add(x, pl.mul(y, scale))

        body = entrance.TracedCallable.from_callable(scaled_add)

        def expanded(*args, **kwargs):
            trace = entrance.current_trace()
            if trace is None:
                raise TypeError("expanded() requires an active trace")
            return trace.expand(body, args=args, kwargs=kwargs)

        def kernel(
            a: pl.Tensor[[4], pl.FP32], b: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]
        ):
            ta = pl.load(a, [0], [4])
            tb = pl.load(b, [0], [4])
            pl.store(expanded(ta, tb, 2.0), [0, 0], o)
            return o

        fn = derive_and_invoke(kernel)
        assert "pl.tile.muls" in fn.as_python()
        assert "scaled_add" not in fn.as_python()

    def test_expansion_with_structured_and_host_values(self):
        calls: list = []

        def noted(x, tag):
            calls.append(tag)
            return x

        body = entrance.TracedCallable.from_callable(noted)

        def kernel(o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(o, [0], [4])
            trace = entrance.current_trace()
            assert trace is not None
            return trace.expand(body, args=(t,), kwargs={"tag": "first"})

        fn = derive_and_invoke(kernel)
        assert calls == ["first"]
        assert "pl.tile.load" in fn.as_python()
