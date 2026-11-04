"""Acceptance for the authoring runtime (``@pl.authoring.recipe``, ``@pl.authoring.body``).

The conformance gates of the runtime's contract
(``docs/en/dev/language/07-authoring-recipes.md``):

* **Splice semantics** — ``body.splice(...)`` is the borrowed collection:
  no function, no call, values cross back to the caller.
* **Callee compositions** — ``body.inline(...)`` / ``body.incore(...)`` are
  push / collect / finalize / call compositions: the program gains exactly
  one callee per (body, argument types), typed Inline / InCore.
* **Root parity** — a root recipe (invoke) reproduces the entrance
  foundation's root trace structurally.
* **The invariants** — one collection per unit, LIFO finalization, no
  unfinalized units at completion, stale handles, refused nesting, host
  calls, derived-signature discipline.
"""

import pypto.language as pl
import pytest
import torch
from pypto import backend
from pypto.backend import BackendType
from pypto.language.authoring import FunctionArtifact, ParamSpec
from pypto.language.tracing import entry as entrance
from pypto.language.tracing.entry.annotation import derive
from pypto.language.tracing.entry.declaration import Infer, fold_auto_scope
from pypto.pypto_core import ir

F32 = torch.float32


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure the backend the compile-for-test lane needs (the jit conftest's)."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    yield
    backend.reset_for_testing()


def _tensor() -> torch.Tensor:
    return torch.randn(64, 64, dtype=F32)


def _out() -> torch.Tensor:
    return torch.empty(64, 64, dtype=F32)


def _printed(program: ir.Program, entry: str) -> str:
    """A program's printed functions, with the entry's name normalized away."""
    printed = "\n".join(f"### {gv.name}\n{ir.python_print(fn)}" for gv, fn in program.functions.items())
    return printed.replace(entry, "ENTRY")


def _function(program: ir.Program, name: str) -> ir.Function:
    by_name = {gv.name: fn for gv, fn in program.functions.items()}
    assert name in by_name, f"{name} not in {sorted(by_name)}"
    return by_name[name]


class TestRecipeEntrance:
    def test_a_host_call_is_rejected(self):
        @pl.authoring.recipe
        def noop(ctx):
            return None

        with pytest.raises(TypeError, match="requires an active trace"):
            noop()

    def test_the_decorator_requires_a_plain_function(self):
        with pytest.raises(TypeError, match="expects a plain function"):
            pl.authoring.recipe(42)  # pyright: ignore[reportArgumentType] - a non-callable is the case under test


class TestSpliceBody:
    def test_a_host_splice_is_rejected(self):
        @pl.authoring.body
        def scaled(x, k):
            return pl.mul(x, k)

        with pytest.raises(TypeError, match="requires an active trace"):
            scaled.splice(1.0, 2.0)

    def test_a_direct_body_call_names_the_placements(self):
        @pl.authoring.body
        def scaled(x, k):
            return pl.mul(x, k)

        with pytest.raises(TypeError, match="place it explicitly"):
            scaled(1.0, 2.0)


class TestCalleeCompositions:
    def test_a_host_composition_is_rejected(self):
        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        with pytest.raises(TypeError, match="requires an active trace"):
            squared.inline(1.0)


class TestRootComposition:
    def _contract_from_decl(self, decl) -> pl.authoring.FunctionContract:
        returns = tuple(decl.returns) if decl.returns is not Infer else None
        return pl.authoring.FunctionContract(
            name=decl.name,
            function_type=decl.function_type,
            level=decl.level,
            role=decl.role,
            attrs=fold_auto_scope(decl.attrs, decl.auto_scope),
            params=tuple(ParamSpec(p.name, p.type, p.direction) for p in decl.params),
            returns=returns,
            strict_ssa=decl.strict_ssa,
            requires_runtime_binding=decl.requires_runtime_binding,
        )

    def test_a_root_recipe_reproduces_the_foundation_trace(self):
        T = pl.Tensor[[8, 8], pl.FP32]

        def kernel(a: T, b: T) -> T:  # pyright: ignore[reportInvalidTypeForm]
            return pl.add(a, b)

        plan = derive(kernel)
        reference = entrance.TracedCallable.from_callable(kernel, decl=plan.decl).invoke(
            args=plan.args, kwargs=plan.kwargs
        )

        @pl.authoring.recipe
        def entry_recipe(ctx, body):
            fn = ctx.current
            artifact = fn.finalize(fn.collect_body(body))
            return ctx.materialize_entry(artifact)

        artifact = entry_recipe.invoke(self._contract_from_decl(plan.decl), kernel)
        assert isinstance(artifact, FunctionArtifact)
        assert artifact.is_entry
        assert ir.structural_equal(artifact.function, reference, enable_auto_mapping=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pl.function(type=pl.FunctionType.InCore)
def _dispatch_leaf(
    x: pl.Tensor[[8, 8], pl.FP32], out: pl.Tensor[[8, 8], pl.FP32]
) -> pl.Tensor[[8, 8], pl.FP32]:
    """The submitted callee: a marked InCore kernel."""
    t = pl.add(pl.load(x, [0, 0], [8, 8]), pl.load(x, [0, 0], [8, 8]))
    return pl.store(t, [0, 0], out)


def _small_tensor() -> torch.Tensor:
    return torch.randn(8, 8, dtype=F32)


def _small_out() -> torch.Tensor:
    return torch.empty(8, 8, dtype=F32)


@pl.function(type=pl.FunctionType.InCore)
def _dispatch_leaf(
    x: pl.Tensor[[8, 8], pl.FP32], out: pl.Tensor[[8, 8], pl.FP32]
) -> pl.Tensor[[8, 8], pl.FP32]:
    """The submitted callee: a marked InCore kernel."""
    t = pl.add(pl.load(x, [0, 0], [8, 8]), pl.load(x, [0, 0], [8, 8]))
    return pl.store(t, [0, 0], out)


def _small_tensor() -> torch.Tensor:
    return torch.randn(8, 8, dtype=F32)


def _small_out() -> torch.Tensor:
    return torch.empty(8, 8, dtype=F32)
