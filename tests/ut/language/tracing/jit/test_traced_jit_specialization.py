# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Collection timing and specialization reuse for the tracing jit.

The corpus suite proves the emitted IR is right. This one proves the lifecycle
around it: decoration must not run a body; the first structural use collects it
and later compatible uses reuse that trace; runtime scalars and tensor contents
must not split the artifact; a host knob, a shape/dtype change, or a compile
option must; a runtime-only config must reach dispatch without recompiling; a
world mismatch must fail loudly instead of silently tracing the wrong thing;
and a returned value must be delivered by a parameter — by *provenance*, so a
callee result that returns its Out argument counts.

Everything here is CPU-side. Compilation is spied at ``_compile_program`` and
dispatch at the returned handle, so no code generation or device is needed; the
device path is exercised by the system-tier corpus.
"""

import pypto.language as pl
import pytest
import torch
from pypto._target import ExecutionMode
from pypto.backend import BackendType
from pypto.ir._entry_outputs import materialize_entry_outputs
from pypto.language.tracing.decorators import function as function_module
from pypto.language.tracing.decorators import jit_trace as jit_module
from pypto.pypto_core import ir, passes
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime import RunConfig

from tests.ut.language.tracing.jit._traced_corpus_fixtures import annotated_add, controlled_accumulate

F32 = torch.float32

# A dynamic extent a callee's declared `->` may carry while its specialized
# body produced a concrete one (the documented callee-lane specialization).
ROUND2_DYN = pl.dynamic("ROUND2_DYN")


def _lazy_kernel():
    """A fresh tracing of the frozen F1 fixture (``controlled_accumulate``).

    The lifecycle assertions need a pristine specializer (empty cache, no
    eager record) per test, so the fixture's raw function is re-decorated
    instead of sharing the module-level object. Re-decoration — not a
    copied body — is what keeps these assertions on the corpus program: a
    copy can drift, while the fixture cannot without failing the corpus
    suite that pins its IR.
    """
    return pl.jit_trace(controlled_accumulate.kernel)


def _annotated_kernel():
    """A fresh tracing of the frozen F2 fixture (``annotated_add``).

    Same isolation rationale as :func:`_lazy_kernel`: the body under test
    is the corpus fixture's, re-decorated per test for pristine state.
    """
    return pl.jit_trace(annotated_add.kernel)


def _tensor(shape: tuple[int, int] = (64, 64), dtype: torch.dtype = F32) -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype)


@pytest.fixture
def trace_count(monkeypatch) -> dict[str, int]:
    """Count traced functions: every collection goes through this hook.

    A parser fallback would not call it, so a nonzero count is also the
    provenance assertion: the kernel under test was collected by the tracer.
    """
    counter = {"n": 0}
    original = function_module._trace_with_annotations

    def counting(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    # Both lanes funnel through this function: the lazy lane calls it through
    # the jit module's binding, the eager lane through ``function_trace``.
    monkeypatch.setattr(function_module, "_trace_with_annotations", counting)
    monkeypatch.setattr(jit_module, "_trace_with_annotations", counting)
    return counter


class _FakeCompiled:
    """Stand-in for the compiled artifact: records the dispatch it receives."""

    def __init__(self, log: list[tuple]) -> None:
        self._log = log

    def __call__(self, *args, config=None):
        self._log.append(("dispatch", args, config))
        return ("dispatched", args, config)


@pytest.fixture
def compile_log(monkeypatch) -> list[tuple]:
    """Replace ``_compile_program`` so dispatch caching is observable CPU-side."""
    log: list[tuple] = []

    def fake_compile(program, config):
        log.append(("compile", config))
        return _FakeCompiled(log)

    monkeypatch.setattr(jit_module, "_compile_program", fake_compile)
    return log


def _compiles(log: list[tuple]) -> int:
    return sum(1 for entry in log if entry[0] == "compile")


class TestDeferredCollection:
    """Decoration is inert; the first structural use collects the trace."""

    def test_decorating_executes_no_body(self, trace_count):
        kernel = _annotated_kernel()
        assert trace_count["n"] == 0, "decoration must not trace"
        # Read the raw slot: the ``_eager_record`` property materializes on
        # access, which would itself be the first use this test rules out.
        assert kernel._JitSpecializer__eager_record is None, "decoration must not materialize the trace"

        kernel.specialize(_tensor(), 1.0, torch.empty(64, 64, dtype=F32))
        assert trace_count["n"] == 1, "the first structural use collects exactly once"

    def test_first_use_collects_then_compatible_uses_reuse(self, trace_count):
        kernel = _annotated_kernel()
        assert trace_count["n"] == 0

        first = kernel.specialize(_tensor(), 1.0, torch.empty(64, 64, dtype=F32))
        assert trace_count["n"] == 1

        second = kernel.compile_for_test(_tensor(), 1.0, torch.empty(64, 64, dtype=F32))
        kernel.specialize(_tensor(), -2.0, torch.empty(64, 64, dtype=F32))
        assert trace_count["n"] == 1, "annotation-determined traces must be reused, not re-collected"
        assert first.name == second.name

    def test_lazy_kernel_retraces_per_call(self, trace_count):
        kernel = _lazy_kernel()
        kernel.specialize(_tensor(), 3, 2.0, torch.empty(64, 64, dtype=F32))
        kernel.specialize(_tensor(), 3, 2.0, torch.empty(64, 64, dtype=F32))
        assert trace_count["n"] == 2, "a lazily traced kernel re-traces per call by design"


class TestSpecializationReuse:
    """The dispatch cache keys on the artifact contract, not on incidental state."""

    def test_runtime_scalar_value_and_tensor_contents_do_not_split(self, compile_log):
        kernel = _lazy_kernel()
        out = torch.empty(64, 64, dtype=F32)

        for n, contents in ((3, 1.0), (7, 2.0), (0, 3.0)):
            kernel(torch.full((64, 64), contents, dtype=F32), n, 2.0, out)

        assert _compiles(compile_log) == 1, "runtime scalars and contents must not recompile"

    def test_shape_and_dtype_split_the_artifact(self, compile_log):
        kernel = _lazy_kernel()
        out64 = torch.empty(64, 64, dtype=F32)

        kernel(_tensor(), 3, 2.0, out64)
        kernel(_tensor((128, 64)), 3, 2.0, out64)
        kernel(_tensor(dtype=torch.float16), 3, 2.0, out64)

        assert _compiles(compile_log) == 3, "a shape or dtype change must compile a new artifact"

    def test_host_knob_value_splits_the_artifact(self, compile_log):
        kernel = _lazy_kernel()
        out = torch.empty(64, 64, dtype=F32)

        kernel(_tensor(), 3, 2.0, out)
        kernel(_tensor(), 3, 2.0, out)
        assert _compiles(compile_log) == 1

        kernel(_tensor(), 3, 3.0, out)
        assert _compiles(compile_log) == 2, "the baked host knob must specialize the trace"

    def test_compile_option_splits_the_artifact(self, compile_log):
        kernel = _lazy_kernel()
        out = torch.empty(64, 64, dtype=F32)

        kernel(_tensor(), 3, 2.0, out)
        with passes.PassContext([], memory_planner=MemoryPlanner.PTOAS):
            kernel(_tensor(), 3, 2.0, out)

        assert _compiles(compile_log) == 2, "a compile option must not share a cached artifact"

    def test_runtime_only_config_does_not_recompile_and_reaches_dispatch(self, compile_log):
        kernel = _lazy_kernel()
        out = torch.empty(64, 64, dtype=F32)

        kernel(_tensor(), 3, 2.0, out, config=RunConfig(device_id=1))
        kernel(_tensor(), 3, 2.0, out, config=RunConfig(device_id=2))

        assert _compiles(compile_log) == 1, "a runtime-only field must not split compilation"
        dispatches = [entry for entry in compile_log if entry[0] == "dispatch"]
        assert [entry[2].device_id for entry in dispatches] == [1, 2], "config must reach dispatch"


class TestCompileConfigForwarding:
    """``config=`` reaches ``ir.compile`` through the canonical mapping.

    ``RunConfig.compile_kwargs()`` is the one mapping onto ``ir.compile()``'s
    parameters; the tracing jit must not carry a second one. A duplicated
    mapping dropped ``dump_ptoas_passes``, so the flag silently did nothing
    on this lane while working on the parser lane.
    """

    @pytest.fixture
    def compile_kwargs_log(self, monkeypatch) -> list[dict]:
        """Capture the kwargs each ``ir.compile`` call receives."""
        log: list[dict] = []

        def fake_compile(program, **kwargs):
            log.append(kwargs)
            return lambda *args, **kw: None

        import pypto.ir as ir_module  # noqa: PLC0415

        monkeypatch.setattr(ir_module, "compile", fake_compile)
        return log

    def test_host_call_forwards_the_canonical_kwargs(self, compile_kwargs_log):
        kernel = _lazy_kernel()
        out = torch.empty(64, 64, dtype=F32)

        kernel(
            _tensor(),
            3,
            2.0,
            out,
            config=RunConfig(platform="a2a3sim", dump_ptoas_passes=True),
        )

        assert len(compile_kwargs_log) == 1
        kwargs = compile_kwargs_log[0]
        assert kwargs["dump_ptoas_passes"] is True, "the canonical mapping's fields must all arrive"
        assert kwargs["backend_type"] is BackendType.Ascend910
        assert kwargs["execution_mode"] is ExecutionMode.SIM
        assert "memory_planner" not in kwargs, "an unset optional field defers to ir.compile"
        assert "device_id" not in kwargs, "a runtime-only field is not a compile input"

    def test_explicit_compile_forwards_the_same_mapping(self, compile_kwargs_log):
        kernel = _lazy_kernel()
        out = torch.empty(64, 64, dtype=F32)

        impl = kernel.compile(
            _tensor(),
            3,
            2.0,
            out,
            config=RunConfig(
                platform="a2a3sim",
                dump_ptoas_passes=True,
                memory_planner=MemoryPlanner.PTOAS,
            ),
        )

        assert len(compile_kwargs_log) == 1
        kwargs = compile_kwargs_log[0]
        assert kwargs["dump_ptoas_passes"] is True
        assert kwargs["memory_planner"] == MemoryPlanner.PTOAS
        # The returned handle speaks the runtime ABI: the staged-declared
        # positions (x, n, out) — the host knob was consumed at compile.
        impl(_tensor(), 3, out)

    def test_no_config_defers_to_ir_compile_defaults(self, compile_kwargs_log):
        kernel = _lazy_kernel()
        kernel(_tensor(), 3, 2.0, torch.empty(64, 64, dtype=F32))
        assert compile_kwargs_log == [{}], "no config means no kwargs — ir.compile owns the defaults"


class TestDiagnosticRequestsBypassTheCache:
    """A request to run the compiler is never served from the artifact cache.

    Diagnostic options change what a compilation *does*, not the artifact it
    produces, so they stay out of the cache key — which is exactly why the
    lookup must be skipped: a warm entry would silently swallow the request.
    The parsing-frontend jit holds the same contract through the same shared
    resolver; these pins are the tracing lane's half.
    """

    @pytest.fixture(autouse=True)
    def _default_sensitive_env(self, monkeypatch):
        # The PassContext-vs-default bypass check compares against the
        # process-default verification level; an outer PYPTO_VERIFY_LEVEL
        # would invert it.
        monkeypatch.delenv("PYPTO_VERIFY_LEVEL", raising=False)

    @pytest.mark.parametrize(
        "options",
        [
            {"dump_passes": True},
            {"dump_ptoas_passes": True},
            {"compile_profiling": True},
            {"save_kernels_dir": "requested-output"},
            {"diagnostic_phase": passes.DiagnosticPhase.PRE_PIPELINE},
            {"disabled_diagnostics": passes.DiagnosticCheckSet()},
        ],
    )
    @pytest.mark.parametrize("warm", [False, True])
    def test_diagnostic_requests_compile_fresh_and_leave_the_cache_alone(self, compile_log, options, warm):
        kernel = _lazy_kernel()
        out = torch.empty(64, 64, dtype=F32)
        if warm:
            kernel(_tensor(), 3, 2.0, out)
        before = dict(kernel._cache)  # noqa: SLF001 — the cache is the object under test

        kernel(_tensor(), 3, 2.0, out, config=RunConfig(**options))
        kernel(_tensor(), 3, 2.0, out, config=RunConfig(**options))

        assert _compiles(compile_log) == (3 if warm else 2), "each request must run the compiler"
        assert kernel._cache == before, "a diagnostic compile must not insert or evict"  # noqa: SLF001

    def test_custom_pass_checks_bypass_a_warm_cache(self, compile_log):
        kernel = _lazy_kernel()
        out = torch.empty(64, 64, dtype=F32)

        kernel(_tensor(), 3, 2.0, out)
        with passes.PassContext([], verification_level=passes.VerificationLevel.ROUNDTRIP):
            kernel(_tensor(), 3, 2.0, out)
            kernel(_tensor(), 3, 2.0, out)
        assert _compiles(compile_log) == 3, "an instrumented context must actually run"

        kernel(_tensor(), 3, 2.0, out)
        assert _compiles(compile_log) == 3, "an ordinary call still reuses the warm artifact"

    def test_conflicting_config_raises_even_on_a_warm_cache(self, compile_log):
        kernel = _lazy_kernel()
        out = torch.empty(64, 64, dtype=F32)

        kernel(_tensor(), 3, 2.0, out)
        with passes.PassContext([], memory_planner=MemoryPlanner.PTOAS):
            with pytest.raises(RuntimeError, match="memory_planner"):
                kernel(_tensor(), 3, 2.0, out, config=RunConfig(memory_planner=MemoryPlanner.DSA_RP))

        assert _compiles(compile_log) == 1, "the conflict must fire before any compilation"


class TestWorldMismatchIsLoud:
    """A misplaced argument must fail instead of silently tracing the wrong world."""

    def test_host_scalar_into_a_tensor_slot(self):
        kernel = _lazy_kernel()
        with pytest.raises(TypeError, match="runtime parameter"):
            kernel(3.5, 3, 2.0, torch.empty(64, 64, dtype=F32))

    def test_tensor_into_a_host_knob(self):
        kernel = _lazy_kernel()
        with pytest.raises(TypeError, match="host"):
            kernel(_tensor(), 3, _tensor(), torch.empty(64, 64, dtype=F32))

    def test_tensor_into_a_scalar_slot(self):
        kernel = _lazy_kernel()
        with pytest.raises(TypeError):
            kernel(_tensor(), _tensor(), 2.0, torch.empty(64, 64, dtype=F32))


@pl.jit_trace.incore
def add_one(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [8, 8]), 1.0), [0, 0], out)
    return out


@pl.jit_trace.inline
def project(x: pl.Tensor[[64, 64], pl.FP32], y: pl.Tensor[[64, 64], pl.FP32]):
    """A write-through helper: the buffer it assembled into, spelled as a return."""
    y[:, :] = pl.add(x, 1.0)
    return y


@pl.jit_trace
def tuple_of_results(
    x: pl.Tensor[[64, 64], pl.FP32],
    a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    """Multiple results, each an alias of the declared Out argument it wrote."""
    qa = project(x, a)
    qb = project(x, b)
    return pl.Tuple(qa, qb)


@pl.jit_trace.incore
def relay_results(
    x: pl.Tensor[[8, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    logits: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
):
    """A callee that hands back one staged tuple under a flat 3-way contract."""
    with pl.at(level=pl.Level.CORE_GROUP):
        scaled = pl.mul(pl.load(x, [0, 0], [8, 8]), 2.0)
        pl.store(scaled, [0, 0], out)
        pl.store(scaled, [0, 0], logits)
    return pl.Tuple(out, logits)


@pl.jit_trace
def destructures_a_callees_tuple(
    x: pl.Tensor[[8, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    logits: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
):
    """Destructure a MakeTuple-returning callee and hand the parts back.

    Both positions are write-through aliases of this entry's own ``pl.Out``
    parameters, so the delivered map must name both. Element 0 is what a
    1-entry callee map loses: the walk resolves ``TupleGetItem`` through the
    callee's position-faithful map, and a callee returning one ``pl.Tuple``
    under N flat positions has to expand it at position 0 too.
    """
    first, second = relay_results(x, out, logits)
    return pl.Tuple(first, second)


@pl.jit_trace
def store_expression_result(
    x: pl.Tensor[[64, 64], pl.FP32],
    out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    """``return pl.store(...)`` — the returned expression writes into ``out``."""
    with pl.at(level=pl.Level.CORE_GROUP):
        return pl.store(pl.mul(pl.load(x, [0, 0], [64, 64]), 2.0), [0, 0], out)


@pl.jit_trace
def tuple_with_a_created_element(
    x: pl.Tensor[[64, 64], pl.FP32],
    a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    """The declared output is delivered; the created one is not."""
    qa = project(x, a)
    created = pl.create_tensor([64, 64], dtype=pl.FP32)
    return pl.Tuple(qa, created)


class TestEntryReturnAnnotation:
    """An entry written without ``return`` still declares its output buffer.

    The parser lane synthesizes ``-> <first Out param>`` for a jit entry whose
    body ends without returning; the tracing lane derived nothing from a body
    with no return statement, so a host entry printed with no annotation and the
    two lanes disagreed about the signature the printer round-trips.
    """

    def test_a_body_without_return_declares_its_first_out_param(self):
        @pl.jit_trace
        def entry(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            add_one(x, out)

        program = entry.specialize()
        fn = next(f for f in program.functions.values() if f.name == "entry")
        assert [str(t) for t in fn.return_types] == ["pl.Tensor[[8, 8], pl.FP32]"]

    def test_a_declared_return_is_never_overwritten(self):
        # The fixture's point is a body that returns nothing: the declaration
        # alone must survive, so no ``return`` is allowed here either.
        @pl.jit_trace
        def declared(
            x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]
        ) -> pl.Tensor[[8, 8], pl.FP32]:  # pyright: ignore[reportReturnType]
            add_one(x, out)

        program = declared.specialize()
        fn = next(f for f in program.functions.values() if f.name == "declared")
        assert [str(t) for t in fn.return_types] == ["pl.Tensor[[8, 8], pl.FP32]"]

    def test_disagreeing_return_paths_are_refused(self):
        """An empty signature must not hide two incompatible branch results."""

        @pl.jit_trace
        def split(
            n: pl.Scalar[pl.INDEX],
            a: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            b: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
        ):
            if n > 0:
                return a
            return b

        with pytest.raises(TypeError, match=r"return paths disagree"):
            split.specialize()

    def test_an_unknown_typed_value_cannot_hide_behind_a_consistent_pair(self):
        """A dropped unknown must not make a known+unknown pair agree.

        ``return_value_types`` records a value whose own type is
        ``UnknownType`` in its path — filtering it would let this branch pair
        compare equal and pin a one-result signature onto a body whose other
        branch returns two values. The unknown is the mismatch marker.
        """

        @pl.jit_trace
        def mixed_paths(
            n: pl.Scalar[pl.INDEX],
            a: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ):
            unknown = pl.Scalar(expr=ir.Var("u", ir.UnknownType(), ir.Span.unknown()))
            if n > 0:
                return a, unknown
            return a

        with pytest.raises(TypeError, match=r"return paths disagree"):
            mixed_paths.specialize()

    def test_a_declared_return_is_checked_against_the_values(self):
        @pl.jit_trace
        def lies(
            a: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ) -> pl.Tensor[[16, 8], pl.FP32]:
            return a

        with pytest.raises(TypeError, match=r"return position 0 is declared .*16, 8.*returns .*8, 8"):
            lies.specialize()

    def test_a_flat_multi_result_declaration_is_checked_position_wise(self):
        @pl.jit_trace
        def tuple_lies(
            a: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            b: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ) -> tuple[pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[16, 8], pl.FP32]]:
            return pl.Tuple(a, b)

        with pytest.raises(TypeError, match=r"return position 1 is declared .*16, 8.*returns .*8, 8"):
            tuple_lies.specialize()

    def test_a_declared_arity_must_match_the_returned_values(self):
        @pl.jit_trace
        def arity_lies(
            a: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            b: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ) -> pl.Tensor[[8, 8], pl.FP32]:
            return pl.Tuple(a, b)

        with pytest.raises(TypeError, match=r"declared `->` has 1 result.*returns 2 value"):
            arity_lies.specialize()

    def test_a_body_local_valid_shape_symbol_is_not_pinned_onto_the_signature(self):
        """A derived ``->`` must be spellable, view metadata included.

        ``signature_expressible`` inspects a recovered type's view metadata
        (``valid_shape``), not just its outer shape: a tile whose valid extent
        is a body-local Var (a scalar read inside the body) has no name in
        scope at the signature, and pinning it would print an annotation that
        cannot be re-parsed ("Unknown shape variable"). The derivation
        declines instead of lying.
        """

        @pl.jit_trace
        def entry(
            cfg: pl.Tensor[[4], pl.INT32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ):
            vlen = pl.read(cfg, [0])
            tile = pl.load(
                cfg,
                [0, 0],
                [64, 64],
                valid_shape=[64, vlen],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(tile, [0, 0], out)
            return tile

        program = entry.specialize()
        fn = next(f for f in program.functions.values() if f.name == "entry")
        assert fn.return_types == []

    def test_a_param_named_valid_shape_symbol_still_pins_the_signature(self):
        """A view extent naming a scalar param is spellable and still pins.

        The printed ``pl.TileView(valid_shape=[64, vlen])`` re-parses — the
        annotation resolves ``vlen`` against the param list — so a
        param-declared symbol must not be declined by the expressibility
        guard.
        """

        @pl.jit_trace
        def entry(
            cfg: pl.Tensor[[64, 64], pl.FP32],
            vlen: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ):
            tile = pl.load(
                cfg,
                [0, 0],
                [64, 64],
                valid_shape=[64, vlen],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(tile, [0, 0], out)
            return tile

        program = entry.specialize()
        fn = next(f for f in program.functions.values() if f.name == "entry")
        assert [str(t) for t in fn.return_types] == [
            "pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[64, vlen])]"
        ]

    def test_a_body_local_extent_inside_a_returned_tuple_is_not_pinned(self):
        """The expressibility guard recurses through tuple elements.

        A multi-result return may carry a structural ``pl.Tuple`` slot; a
        body-local extent Var inside that slot has no name in scope at the
        signature either, so the whole derivation declines (all-or-nothing).
        Driven directly on the seam: a trace reaching it is already refused
        by the entry's provenance guard, which would mask the pin decision.
        """
        from pypto.ir.utils import derived_return_types  # noqa: PLC0415 - regression seam

        span = ir.Span.unknown()
        body_local = ir.Var("vlen", ir.ScalarType(pl.INDEX), span)
        static_out = ir.TensorType([64, 64], pl.FP32, None)
        nested = ir.TupleType([ir.TensorType([body_local, 64], pl.FP32, None)])

        def make_function(returns):
            params = [ir.Var("x", static_out, span), ir.Var("n", ir.ScalarType(pl.INDEX), span)]
            body = ir.ReturnStmt([ir.Var(f"v{i}", t, span) for i, t in enumerate(returns)], span)
            return ir.Function("entry", params, [], body, span)

        # The body-local extent hides inside the tuple: without recursion the
        # guard would accept it and pin an unprintable signature.
        assert derived_return_types(make_function([static_out, nested])) == []

        # A view extent naming a Var that is not one of the params is declined
        # too (``pl.TensorView(valid_shape=[<body-local>])`` does not re-parse).
        stranger = ir.Var("stranger", ir.ScalarType(pl.INDEX), span)
        view = ir.TensorView([ir.ConstInt(1, pl.INDEX, span)], ir.TensorLayout.ND, [stranger])
        declined = ir.TensorType([8, 8], pl.FP32, None, view)
        assert derived_return_types(make_function([static_out, declined])) == []

        # The same view shape naming the function's own scalar param pins.
        nameable = ir.Var("n", ir.ScalarType(pl.INDEX), span)
        params = [ir.Var("x", static_out, span), nameable]
        pinned_view = ir.TensorView([ir.ConstInt(1, pl.INDEX, span)], ir.TensorLayout.ND, [nameable])
        pinned = ir.TensorType([8, 8], pl.FP32, None, pinned_view)
        body = ir.ReturnStmt([ir.Var("v0", static_out, span), ir.Var("v1", pinned, span)], span)
        fn = ir.Function("entry", params, [], body, span)
        assert len(derived_return_types(fn)) == 2


class TestCalleeReturnAnnotation:
    """A nested callee's declared ``->`` is a contract, not a suggestion.

    The per-call variant pins its return types on the traced values (the
    callee-lane contract: a declared dynamic extent meets concrete
    specialized parameters, and pinning the declared form would fork the
    caller's variable). The declaration's static parts — kind, dtype, static
    extents, arity — must still describe those values: without the check a
    callee whose ``->`` lied was silently retyped in its caller, while the
    same lie at an entry was already loud.
    """

    def test_a_wrong_dtype_is_refused_at_the_nested_call(self):
        @pl.jit_trace
        def lies(x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.INT32]:
            return pl.mul(x, 2.0)

        @pl.jit_trace
        def caller(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            out[:] = lies(x)
            return out

        with pytest.raises(TypeError, match=r"return position 0 is declared .*INT32.*returns .*FP32"):
            caller.specialize()

    def test_a_wrong_static_extent_is_refused_at_the_nested_call(self):
        @pl.jit_trace
        def lies(x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[16, 8], pl.FP32]:
            return pl.mul(x, 2.0)

        @pl.jit_trace
        def caller(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            out[:] = lies(x)
            return out

        with pytest.raises(TypeError, match=r"return position 0 is declared .*16, 8.*returns .*8, 8"):
            caller.specialize()

    def test_a_declared_dynamic_extent_still_specializes(self):
        """The positive control: a declared dynamic extent is the one
        difference a specialized callee may carry — the variant keeps the
        traced signature, never the declared symbolic form."""

        @pl.jit_trace
        def symbolic(x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[ROUND2_DYN, 8], pl.FP32]:
            return pl.mul(x, 2.0)

        @pl.jit_trace
        def caller(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            out[:] = symbolic(x)
            return out

        program = caller.specialize()
        callee = next(fn for fn in program.functions.values() if fn.name == "symbolic")
        # The variant's signature is the value's own concrete type, not the
        # declared symbolic form — and the call was accepted, not refused.
        assert [str(t) for t in callee.return_types] == ["pl.Tensor[[8, 8], pl.FP32]"]


class TestReturnCreatedOutputs:
    """Fresh static entry outputs can acquire hidden host buffers at compilation."""

    def test_returning_a_created_tensor_is_materializable(self):
        @pl.jit_trace
        def make_out(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
            result = pl.create_tensor([64, 64], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP):
                pl.store(pl.mul(pl.load(x, [0, 0], [64, 64]), 2.0), [0, 0], result)
            return result

        program = make_out.compile_for_test(_tensor())
        assert any(fn.name == "make_out" for fn in program.functions.values())

    def test_write_through_return_still_compiles(self):
        @pl.jit_trace
        def write_through(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                pl.store(pl.mul(pl.load(x, [0, 0], [64, 64]), 2.0), [0, 0], out)
            return out

        program = write_through.compile_for_test(_tensor(), torch.empty(64, 64, dtype=F32))
        assert {fn.name for fn in program.functions.values()} == {
            "write_through",
            "write_through_incore_0",
        }


class TestWriteThroughResultsAreDelivered:
    """Delivery is a property of the value, not of the name it was bound to.

    A call result whose callee returns the buffer it wrote *is* that buffer, so
    the return is delivered even though the returned variable is not a
    parameter. The check is ``return_lineage``'s walk (the one orchestration
    codegen uses), which is why a tuple of such results is accepted while a
    created element is still refused.
    """

    def test_a_tuple_of_results_is_accepted(self):
        program = tuple_of_results.specialize()
        entry = next(fn for fn in program.functions.values() if fn.name == "tuple_of_results")
        assert [str(t) for t in entry.return_types] == [
            "pl.Tensor[[64, 64], pl.FP32]",
            "pl.Tensor[[64, 64], pl.FP32]",
        ], "one staged tuple still declares the parser-lane flat signature"

    def test_the_delivered_map_names_the_backing_parameters(self):
        program = tuple_of_results.specialize()
        entry = next(fn for fn in program.functions.values() if fn.name == "tuple_of_results")
        assert ir.delivered_param_indices(entry, program) == [1, 2]

    def test_a_store_result_is_delivered(self):
        program = store_expression_result.specialize()
        entry = next(fn for fn in program.functions.values() if fn.name == "store_expression_result")
        assert ir.delivered_param_indices(entry, program) == [1]

    def test_a_callees_staged_tuple_is_expanded_for_its_caller(self):
        """A callee returning ``pl.Tuple`` under N flat positions maps N entries.

        Position 0 is the one a 1-entry map loses: the caller resolves
        ``TupleGetItem`` through the callee's position-faithful map, so a
        literal ``MakeTuple`` has to expand there as well as in the
        value-faithful expansion. Before this, an entry that destructured such
        a callee reported its first result as created by the return value.
        """
        program = destructures_a_callees_tuple.specialize()
        callee = next(fn for fn in program.functions.values() if fn.name == "relay_results")
        assert len(callee.return_types) == 2
        assert ir.returned_param_indices(callee, program) == [1, 2]
        entry = next(fn for fn in program.functions.values() if fn.name == "destructures_a_callees_tuple")
        assert ir.delivered_param_indices(entry, program) == [1, 2]

    @pytest.mark.parametrize("spelling", ["direct", "bound", "destructured"])
    def test_a_forwarded_tuple_has_the_same_lineage_in_every_spelling(self, spelling):
        @pl.jit_trace.host
        def forward(
            x: pl.Tensor[[8, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            logits: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ):
            # Reversing the outputs proves the map follows the callee's values,
            # rather than assuming returned buffers occupy the last parameters.
            if spelling == "direct":
                return relay_results(x, logits, out)
            if spelling == "bound":
                result = relay_results(x, logits, out)
                return result
            first, second = relay_results(x, logits, out)
            return pl.Tuple(first, second)

        @pl.jit_trace.host
        def outer(
            x: pl.Tensor[[8, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            logits: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ):
            return forward(x, out, logits)

        program = outer.specialize()
        for name in ("forward", "outer"):
            entry = next(fn for fn in program.functions.values() if fn.name == name)
            assert len(entry.return_types) == 2
            assert ir.returned_param_indices(entry, program) == [2, 1]
            assert ir.delivered_param_indices(entry, program) == [2, 1]

    def test_a_direct_tuple_call_preserves_a_single_declared_tuple_slot(self):
        @pl.jit_trace.host
        def forward(
            x: pl.Tensor[[8, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            logits: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ) -> pl.Tuple[pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32]]:
            return relay_results(x, logits, out)

        program = forward.specialize()
        entry = next(fn for fn in program.functions.values() if fn.name == "forward")
        # Delivery checks each buffer, but consumers of the declared ABI must
        # still see one tuple slot rather than two flat return positions.
        assert len(entry.return_types) == 1
        assert isinstance(entry.return_types[0], ir.TupleType)
        assert ir.returned_param_indices(entry, program) == [None]
        assert ir.delivered_param_indices(entry, program) == [2, 1]

    def test_a_direct_tuple_call_survives_the_default_pipeline(self):
        # The body owns a CORE_GROUP scope, so let the pipeline outline it
        # from an orchestration entry rather than nesting it inside InCore.
        callee = pl.jit_trace(relay_results.kernel)

        @pl.jit_trace
        def forward(
            x: pl.Tensor[[8, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            logits: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ):
            return callee(x, out, logits)

        program = forward.compile_for_test()
        entry = next(fn for fn in program.functions.values() if fn.name == "forward")
        assert ir.returned_param_indices(entry, program) == [1, 2]

    def test_a_created_element_is_materializable(self):
        program = tuple_with_a_created_element.specialize()
        entry = program.get_function("tuple_with_a_created_element")
        assert entry is not None
        assert ir.delivered_param_indices(entry, program) == [1, None]

    def test_a_forwarded_tuple_retains_an_undelivered_element(self):
        # Construct IR directly: the tracing entry guard would reject the
        # callee before we could inspect the forwarding caller's map.
        @pl.program
        class Program:
            @pl.function
            def inner(
                self, out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]
            ) -> tuple[pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32]]:
                created = pl.create_tensor([8, 8], dtype=pl.FP32)
                return out, created

            @pl.function
            def outer(
                self, out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]
            ) -> tuple[pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32]]:
                return self.inner(out)

        entry = Program.get_function("outer")
        assert entry is not None
        assert ir.returned_param_indices(entry, Program) == [0, None]
        assert ir.delivered_param_indices(entry, Program) == [0, None]

    def test_created_position_gets_a_distinct_physical_slot(self):
        program = tuple_with_a_created_element.specialize()
        entry = program.get_function("tuple_with_a_created_element")
        assert entry is not None
        adapted = materialize_entry_outputs(program, entry)
        assert adapted.result_indices == (1, len(entry.params))


class TestMultiResultSpelling:
    """The parser's two return contracts both trace (B4), and nothing is dropped.

    ``return a, b`` (a tuple literal) splices into the flat multi-result
    form; ``return [a, b]`` (a list literal) and ``return pl.Tuple(a, b)``
    are the one structural value. A container mixing staged and host
    members is refused rather than partially dropped.
    """

    def test_a_python_tuple_return_splices_flat(self):
        @pl.jit_trace
        def host_tuple(
            a: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            b: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ):
            return a, b

        program = host_tuple.specialize()
        entry = next(fn for fn in program.functions.values() if fn.name == "host_tuple")
        assert [str(t) for t in entry.return_types] == [
            "pl.Tensor[[8, 8], pl.FP32]",
            "pl.Tensor[[8, 8], pl.FP32]",
        ]
        stmt = entry.body
        assert type(stmt).__name__ == "ReturnStmt"
        assert len(stmt.value) == 2, "the tuple spelling splices into flat results"

    def test_a_list_return_is_the_structural_tuple(self):
        @pl.jit_trace
        def list_return(
            a: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            b: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ):
            return [a, b]

        program = list_return.specialize()
        entry = next(fn for fn in program.functions.values() if fn.name == "list_return")
        # ``return_value_types`` flattens the MakeTuple, so both spellings
        # derive the same flat signature; the difference lives in the return
        # statement itself — one structural value, not two flat results.
        assert [str(t) for t in entry.return_types] == [
            "pl.Tensor[[8, 8], pl.FP32]",
            "pl.Tensor[[8, 8], pl.FP32]",
        ]
        stmt = entry.body
        assert type(stmt).__name__ == "ReturnStmt"
        assert len(stmt.value) == 1, "the list spelling is one structural value"
        assert type(stmt.value[0]).__name__ == "MakeTuple"

    def test_a_host_element_is_refused_instead_of_dropped(self):
        @pl.jit_trace
        def mixed(
            a: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ):
            return a, 7

        with pytest.raises(TypeError, match=r"mixes staged values with host value"):
            mixed.specialize()

    def test_a_staged_tuple_is_accepted(self):
        program = tuple_of_results.specialize()
        entry = next(fn for fn in program.functions.values() if fn.name == "tuple_of_results")
        assert [str(t) for t in entry.return_types] == [
            "pl.Tensor[[64, 64], pl.FP32]",
            "pl.Tensor[[64, 64], pl.FP32]",
        ]


class TestValuelessEntryPipeline:
    """An entry that writes its Out parameter and returns nothing lowers."""

    def test_valueless_entry_runs_the_pipeline(self):
        """The terminal scope's dead intermediates must not become returns.

        The InCore outliner's no-context fallback used to export every
        non-scalar definition of the scope, so the outlined function declared
        tile-typed returns and the tile conversion rejected them with an
        INTERNAL_CHECK — for a shape the tracing contract supports (a body
        with no value-bearing return keeps its declared type; the entry
        describes its output buffer through ``pl.Out``).
        """
        torch = pytest.importorskip("torch")

        @pl.jit_trace
        def valueless_add(
            a: pl.Tensor[[32, 32], pl.FP32],
            b: pl.Tensor[[32, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                left = pl.load(a, [0, 0], [32, 32])
                right = pl.load(b, [0, 0], [32, 32])
                pl.store(pl.add(left, right), [0, 0], out)

        a = torch.zeros(32, 32, dtype=torch.float32)
        b = torch.zeros(32, 32, dtype=torch.float32)
        out = torch.empty(32, 32, dtype=torch.float32)
        program = valueless_add.lower(a, b, out)
        assert isinstance(program, ir.Program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
