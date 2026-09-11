# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The entrance foundation's acceptance tests.

Primary acceptance: a user-implemented annotation entrance built only on
``pypto.language.entrance`` (plus the standard annotation deriver) produces
IR structurally equal to the built-in ``pl.function``. The remaining tests
pin the declaration data contract, the invocation recipe, preparation
reuse, and the failure lifetime.
"""

import dataclasses

import pypto.language as pl
import pytest
from pypto.language import entrance
from pypto.language.entrance.annotation import InvocationPlan, derive
from pypto.pypto_core import ir

LIVE_SCALE = 2.0


def _structural_equal(left, right) -> bool:
    return ir.structural_equal(left, right, enable_auto_mapping=True)


def my_function(kernel):
    """The litmus entrance: derivation + prepared body + root invoke.

    Implements the ``pl.function`` standalone contract using only the public
    foundation (no private collector, Sema, Frame, or decorator helpers).
    """
    plan = derive(kernel)
    body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
    return body.invoke(args=plan.args, kwargs=plan.kwargs)


# ---------------------------------------------------------------------------
# Param and Decl
# ---------------------------------------------------------------------------


class TestParamDecl:
    def test_param_is_frozen(self):
        param = entrance.Param("a", ir.ScalarType(pl.INT64))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(param, "name", "b")

    def test_param_identity_not_structure(self):
        """Two structurally equal Params are distinct placeholders: identity,
        never structural equality, resolves an invocation."""
        one = entrance.Param("a", ir.ScalarType(pl.INT64))
        two = entrance.Param("a", ir.ScalarType(pl.INT64))
        same_one = one
        assert one == same_one  # identity (eq is identity-pinned)
        assert one != two

    def test_param_validates(self):
        with pytest.raises(TypeError, match="ir.Type"):
            entrance.Param("a", "not-a-type")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="non-empty str"):
            entrance.Param("", ir.ScalarType(pl.INT64))

    def test_decl_rejects_duplicate_parameter_names(self):
        a = entrance.Param("a", ir.ScalarType(pl.INT64))
        b = entrance.Param("a", ir.ScalarType(pl.FP32))
        with pytest.raises(ValueError, match="duplicate parameter name"):
            entrance.Decl(name="f", params=(a, b))

    def test_decl_is_frozen_and_normalizes(self):
        a = entrance.Param("a", ir.ScalarType(pl.INT64))
        decl = entrance.Decl(name="f", params=[a], returns=[ir.ScalarType(pl.INT64)], attrs={"k": 1})
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(decl, "name", "g")
        assert decl.params == (a,)
        assert decl.returns == (ir.ScalarType(pl.INT64),)
        assert decl.attrs == {"k": 1}

    def test_decl_symbol_defaults_to_name(self):
        decl = entrance.Decl(name="gemm")
        assert decl.symbol_name == "gemm"
        named = entrance.Decl(name="gemm", symbol="gemm$64x128")
        assert named.symbol_name == "gemm$64x128"

    def test_decl_returns_contract(self):
        inferred = entrance.Decl(name="f")
        assert inferred.returns is entrance.Infer
        explicit = entrance.Decl(name="f", returns=(ir.ScalarType(pl.INT64),))
        assert explicit.returns == (ir.ScalarType(pl.INT64),)
        with pytest.raises(TypeError, match="Infer"):
            entrance.Decl(name="f", returns=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The litmus: user-implemented pl.function
# ---------------------------------------------------------------------------


class TestUserEntranceEquivalence:
    def test_user_function_matches_builtin(self):
        """The primary acceptance test: the user-implemented entrance equals
        the built-in ``pl.function`` (same kernel source, structural IR)."""

        @pl.function
        def builtin(x: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(x, [0], [4])
            r = pl.mul(t, 2.0)
            pl.store(r, [0, 0], o)
            return o

        def kernel(x: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(x, [0], [4])
            r = pl.mul(t, 2.0)
            pl.store(r, [0, 0], o)
            return o

        mine = my_function(kernel)
        assert isinstance(mine, ir.Function)
        assert mine.name == "kernel"
        assert _structural_equal(builtin, mine)

    def test_user_function_matches_builtin_with_declared_return(self):
        @pl.function
        def builtin(s: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
            return pl.add(s, s)

        def kernel(s: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
            return pl.add(s, s)

        mine = my_function(kernel)
        assert _structural_equal(builtin, mine)
        assert len(mine.return_types) == 1
        assert isinstance(mine.return_types[0], ir.ScalarType)

    def test_user_function_matches_builtin_multi_return(self):
        """A flat multi-return contract: the Sema sees no structural return
        type during the trace; the explicit contract is pinned after."""

        @pl.function
        def builtin(s: pl.Scalar[pl.FP32]) -> tuple[pl.Scalar[pl.FP32], pl.Scalar[pl.FP32]]:
            return pl.add(s, s), pl.mul(s, 2.0)

        def kernel(s: pl.Scalar[pl.FP32]) -> tuple[pl.Scalar[pl.FP32], pl.Scalar[pl.FP32]]:
            return pl.add(s, s), pl.mul(s, 2.0)

        mine = my_function(kernel)
        assert _structural_equal(builtin, mine)
        assert len(mine.return_types) == 2

    def test_user_function_matches_builtin_classification_facts(self):
        """Function classification facts (type, auto_scope, strict SSA) flow
        through the foundation identically to the built-in decorator."""

        @pl.function(type=ir.FunctionType.InCore, auto_scope=False, strict_ssa=True)
        def builtin(x: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(x, [0], [4])
            pl.store(pl.mul(t, 2.0), [0, 0], o)
            return o

        def kernel(x: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(x, [0], [4])
            pl.store(pl.mul(t, 2.0), [0, 0], o)
            return o

        plan = derive(
            kernel,
            function_type=ir.FunctionType.InCore,
            auto_scope=False,
            strict_ssa=True,
        )
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        mine = body.invoke(args=plan.args, kwargs=plan.kwargs)
        assert _structural_equal(builtin, mine)
        assert mine.func_type is ir.FunctionType.InCore
        assert mine.attrs is not None and mine.attrs.get("auto_scope") is False


# ---------------------------------------------------------------------------
# The invocation recipe
# ---------------------------------------------------------------------------


class TestInvocationRecipe:
    def _body(self):
        def kernel(
            x: pl.Tensor[[4], pl.FP32], y: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]
        ):
            t = pl.add(pl.load(x, [0], [4]), pl.load(y, [0], [4]))
            pl.store(t, [0, 0], o)
            return o

        return kernel

    def test_args_none_means_declaration_order(self):
        kernel = self._body()
        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        fn = body.invoke()
        assert fn.name == "kernel"
        assert [p.name_hint for p in fn.params] == ["x", "y", "o"]

    def test_recipe_may_reorder(self):
        kernel = self._body()
        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        x, y, o = plan.decl.params
        swapped = body.invoke(args=(y, x, o))
        straight = body.invoke()
        assert not _structural_equal(swapped, straight)
        text = swapped.as_python().replace("\n", " ")
        # The IR param order is the declaration's; the recipe feeds y into x.
        assert text.index("pl.tile.load(y") < text.index("pl.tile.load(x")

    def test_recipe_may_repeat(self):
        kernel = self._body()
        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        x, y, o = plan.decl.params
        fn = body.invoke(args=(x, x, o))
        text = fn.as_python()
        assert "pl.load(y" not in text.replace("\n", " ")

    def test_recipe_may_inject_host_values(self):
        kernel = self._body()
        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        x, y, o = plan.decl.params
        fn = body.invoke(args=(x, y, o), kwargs={})
        assert fn.name == "kernel"

    def test_kwargs_only_recipe_passes_empty_args(self):
        """``args=None`` means all declared parameters *positionally*: a
        kwargs-only recipe must pass ``args=()`` explicitly."""
        kernel = self._body()
        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        x, y, o = plan.decl.params
        by_keyword = body.invoke(args=(), kwargs={"x": x, "y": y, "o": o})
        straight = body.invoke()
        assert _structural_equal(by_keyword, straight)

    def test_kwargs_recipe_with_args_none_is_loud(self):
        """``args=None`` plus a non-empty ``kwargs`` duplicates every name —
        rejected by normal Python binding, never silently merged."""
        kernel = self._body()
        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        x, y, o = plan.decl.params
        with pytest.raises(TypeError, match="multiple values"):
            body.invoke(kwargs={"x": x, "y": y, "o": o})

    def test_unowned_param_is_loud(self):
        kernel = self._body()
        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        foreign = entrance.Param("x", ir.ScalarType(pl.INT64))
        with pytest.raises(TypeError, match="not.*owned by this declaration"):
            body.invoke(args=(foreign, foreign, foreign))

    def test_recipe_may_omit_and_let_python_binding_validate(self):
        kernel = self._body()
        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            body.invoke(args=plan.decl.params[:2])


# ---------------------------------------------------------------------------
# Preparation reuse and the failure lifetime
# ---------------------------------------------------------------------------


class TestPreparationAndLifetime:
    def test_invoke_requires_a_declaration(self):
        def kernel(x: pl.Scalar[pl.FP32]):
            return x

        body = entrance.TracedCallable.from_callable(kernel)
        assert body.decl is None
        with pytest.raises(TypeError, match="without a declaration"):
            body.invoke()

    def test_repeated_invocations_get_fresh_state(self):
        def kernel(x: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(x, [0], [4])
            pl.store(t, [0, 0], o)
            return o

        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        first = body.invoke()
        second = body.invoke()
        assert _structural_equal(first, second)
        assert first.params[0] is not second.params[0]

    def test_live_globals_observed_per_invocation(self):
        """Preparation caches code, not host observations: a module global the
        body reads is re-observed on every invocation."""

        def kernel(x: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            t = pl.load(x, [0], [4])
            r = pl.mul(t, LIVE_SCALE)
            pl.store(r, [0, 0], o)
            return o

        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        at_two = body.invoke()
        globals()["LIVE_SCALE"] = 3.0
        try:
            at_three = body.invoke()
        finally:
            globals()["LIVE_SCALE"] = 2.0
        assert not _structural_equal(at_two, at_three)

    def test_escaping_exception_propagates_unchanged(self):
        sentinel = ValueError("kernel boom")

        def kernel(x: pl.Scalar[pl.FP32]):
            raise sentinel

        plan = derive(kernel)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        with pytest.raises(ValueError) as caught:
            body.invoke()
        assert caught.value is sentinel

    def test_from_callable_rejects_non_function(self):
        with pytest.raises(TypeError, match="expects a Python function"):
            entrance.TracedCallable.from_callable(print)


# ---------------------------------------------------------------------------
# from_string
# ---------------------------------------------------------------------------


class TestFromString:
    SOURCE = """
def generated(x, o):
    t = pl.load(x, [0], [4])
    r = pl.mul(t, 2.0)
    pl.store(r, [0, 0], o)
    return o
"""

    def _decl(self):
        tensor = ir.TensorType([ir.ConstInt(4, pl.INDEX, ir.Span.unknown())], pl.FP32)
        x = entrance.Param("x", tensor)
        o = entrance.Param("o", tensor, direction=ir.ParamDirection.Out)
        return entrance.Decl(name="generated", params=(x, o))

    def test_from_string_traces(self):
        environment: dict[str, object] = {"pl": pl}
        body = entrance.TracedCallable.from_string(
            self.SOURCE, name="generated", environment=environment, decl=self._decl()
        )
        fn = body.invoke()
        assert fn.name == "generated"
        assert "pl.tile.muls" in fn.as_python()

    def test_from_string_environment_is_live(self):
        environment: dict = {"pl": pl, "SCALE": 2.0}
        source = """
def generated(x, o):
    t = pl.load(x, [0], [4])
    pl.store(pl.mul(t, SCALE), [0, 0], o)
    return o
"""
        body = entrance.TracedCallable.from_string(
            source, name="generated", environment=environment, decl=self._decl()
        )
        at_two = body.invoke()
        environment["SCALE"] = 3.0
        at_three = body.invoke()
        assert not _structural_equal(at_two, at_three)

    def test_from_string_missing_name_is_loud(self):
        with pytest.raises(SyntaxError, match="cannot find function"):
            entrance.TracedCallable.from_string("def other(): ...", name="generated", environment={"pl": pl})

    def test_from_string_environment_is_required(self):
        """Text has no globals, closures, defaults, or identity: the
        environment is an explicit input, not a defaulted empty one."""
        with pytest.raises(TypeError, match="environment"):
            entrance.TracedCallable.from_string(self.SOURCE, name="generated")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# The annotation deriver
# ---------------------------------------------------------------------------


class TestDerivation:
    def test_derive_produces_plan(self):
        def kernel(x: pl.Tensor[[4], pl.FP32], o: pl.Out[pl.Tensor[[4], pl.FP32]]) -> pl.Tensor[[4], pl.FP32]:
            return o

        plan = derive(kernel)
        assert isinstance(plan, InvocationPlan)
        assert plan.decl.name == "kernel"
        assert [p.name for p in plan.decl.params] == ["x", "o"]
        assert plan.decl.params[0].direction is ir.ParamDirection.In
        assert plan.decl.params[1].direction is ir.ParamDirection.Out
        assert plan.args == tuple(plan.decl.params)
        assert plan.kwargs == {}

    def test_derive_return_contract(self):
        def typed(s: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
            return s

        def untyped(s: pl.Scalar[pl.FP32]):
            return s

        assert derive(typed).decl.returns == (ir.ScalarType(pl.FP32),)
        assert derive(untyped).decl.returns is entrance.Infer

    def test_derive_classification_facts_flow(self):
        def kernel(o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            return o

        plan = derive(kernel, function_type=ir.FunctionType.InCore, auto_scope=False)
        assert plan.decl.function_type is ir.FunctionType.InCore
        assert plan.decl.auto_scope is False
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        fn = body.invoke()
        assert fn.func_type is ir.FunctionType.InCore
        assert fn.attrs is not None and fn.attrs.get("auto_scope") is False

    def test_derive_rejects_unannotated_param(self):
        def kernel(x, o: pl.Out[pl.Tensor[[4], pl.FP32]]):
            return o

        with pytest.raises(TypeError, match="needs a pl.Tensor"):
            derive(kernel)

    def test_derive_resolves_pep563_names_from_closure_and_globals(self):
        """PEP 563 string annotations may reference enclosing-scope names: the
        kernel's own closure cells resolve them even when the defining frame
        is gone (derivation after the fact), and caller-visible names complete
        the resolution."""

        def make_kernel():
            cols = 4

            def kernel(x, o):
                t = pl.load(x, [0], [cols])
                pl.store(t, [0, 0], o)
                return o

            kernel.__annotations__ = {
                "x": "pl.Tensor[[cols], pl.FP32]",
                "o": "pl.Out[pl.Tensor[[cols], pl.FP32]]",
            }
            return kernel

        kernel = make_kernel()
        plan = derive(kernel)
        shapes = [list(p.type.shape) for p in plan.decl.params]  # type: ignore[union-attr]
        assert all(getattr(extent, "value", None) == 4 for shape in shapes for extent in shape)
        body = entrance.TracedCallable.from_callable(kernel, decl=plan.decl)
        fn = body.invoke()
        assert "cols" not in fn.as_python()
        assert fn.param_directions[1] is ir.ParamDirection.Out
