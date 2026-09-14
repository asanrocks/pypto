# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Corpus conformance pins for the tracing frontend.

Each test pins one root cause found when the pypto-lib ``deepseek_v4_pro``
model zoo (the real tracing consumer corpus) was imported under the tracing
frontend (2026-09 probe; 37/38 modules failed at decoration). The corpus is
outside this repo's gate, so these minimal repros keep its contract visible
here. When a test goes green, remove its entry from ``_migration_xfail.txt``
(strict xfail reports the fix as an XPASS).

Coverage — FIXED FRONTEND GAPS (regression pins; the owning change is in
git history):

* ``rmsnorm.py``        → eager ``@pl.jit`` (and standalone ``@pl.function``)
  calling a *module-global* standalone callee. Fixed by the shared-tail
  callee adaptation (``_adapt_captured_functions`` +
  ``_wrap_global_callee_functions`` in ``function.py``).
* ``lm_head.py``        → the lazy callee lane retyped ``DistributedTensorType``
  call-site args to plain ``Tensor`` annotations. Fixed by the type-sound
  ``_in_trace_annotation`` bridge (DistributedTensorType checked before its
  TensorType base; memref/tensor_view preserved).
* ``qwen3_14b/decode_fwd.py`` → the same bridge rejected ``pl.Array`` args
  outright. Fixed by the ``ArrayType`` bridge branch.
* ``decode_sparse_attn.py`` → ``deps=[tids[i]]``: the parser's Form-1
  desugaring is ported to the traced ``deps=`` coercion (``_coerce_deps`` in
  ``constructs/submit.py``, shared by ``pl.submit`` / ``pl.spmd`` / ``pl.at``).
* ``moe.py``           → ``self.<sibling>`` in jit bodies
  (``pl.spmd_submit(self.shared_routed, ...)`` inside ``@pl.jit.inline
  combine``): the jit lanes now materialize ``self`` as the module's
  traced-kernel registry (the ``_JitSelfProxy`` injection in
  ``_trace_with_annotations`` — the jit counterpart of
  ``_trace_program_method``'s self-less ``_SelfProxy`` injection).
* ``moe.py``           → annotationless ``spmd_submit`` callee
  (``shared_routed`` declares no ``->`` but returns its ``pl.Out`` param):
  the eager jit lane now pins the return contract derived from the trace
  onto the Function signature — the tracing counterpart of the
  specializer's synthesized ``->`` — so the Submit's
  ``Tuple[<ret>, TASK_ID]`` types from a recovered, checked fact. The
  derivation is all-or-nothing and never pins ``UnknownType`` (a
  path-disagreeing callee keeps ``[]`` and the submit stays loud; see
  ``derived_return_types`` in ``return_contract.py``).
* ``~20 modules``      → ``pl.Out``/``pl.InOut`` params on ``@pl.jit.inline``
  helpers (hc_pre's ``_hc_pre_separate``, moe's/hc_post's ``x_next`` —
  the corpus's standard write-through-alias style): the jit lanes now
  drop the directions to In, mirroring the specializer's generated-source
  normalization (warning included). A direction left on an Inline param
  made the call pass the arg as Out, so the caller's ``return x_next``
  after ``hc_post(..., x_next)`` tripped the InOutUseDiscipline verifier
  at pipeline_input (first corpus compile probe, 2026-09).
* ``expert_routed.py``  → the with-form ``pl.spmd`` body shape
  (``with pl.spmd(...): <multi-statement body reading get_block_idx()>`` —
  the exp_gate_mm/exp_up_mm/exp_w2_mm scopes): the tracer now wraps a
  multi-statement body in the nested InCore region exactly like the
  parser's ``_emit_spmd_body`` (the rewriter threads the static
  single-call fact through ``rt.on_begin_with``; TaskId capture is
  orthogonal). The pre-fix tracer never wrapped the plain with-form, so
  ``OutlineClusterScopes`` built the Spmd wrapper with an inline body and
  orchestration codegen's ``FindWrapperInnerCall`` failed ("no inner call
  found in Spmd function 'exp_gate_mm_spmd'" — the last frontend blocker
  on the corpus compile). Full matrix pinned in
  ``tests/ut/language/conformance/test_spmd_with_form.py``.
* ``gate.py``           → tile subscript-write sugar (``sq[0:1, :] = row``):
  ``Tile.__setitem__`` now lowers to ``tile.assemble``, mirroring
  ``Tensor.__setitem__``.
* ``decode_sparse_attn*.py`` (and six import-chain modules) → 22
  type-changing same-name rebinds (``g_row_max = pl.reshape(g_row_max,
  ...)`` et al.) silently carried the first binding's type; fixed by the
  relaxed reassignment rule — fresh Var on type change in both frontends
  (``tests/ut/language/test_rebind_type_rule.py``). The same corpus
  surface exposed ``pl.log`` bound to the tile-only wrapper — fixed by the
  unified binding (``tests/ut/language/test_unified_op_dispatch.py``).

Coverage — CADENCE PINS (``TestCadence*`` classes; NOT fixable in the
current repo state — they require adopting the post-fork upstream commits
listed below; the fix is a rebase/adopt, never a test or corpus edit):

* ``valid_shape`` kwarg on ``pl.load`` (``2e54cd660`` renamed
  ``valid_shapes`` → ``valid_shape``) and on ``gather_row`` (``#2142``).
* ``pl.prefetch.*`` async GM→L2 prefetch frontend (``#2089``).
* ``pl.cross_core_slot`` as an ``optimizations=`` entry.
* ``pld.system.defer_wait`` deferred task completion waits (``#2369``).

NEXT WORK — deepseek v4-pro beyond the frontend (verified 2026-09): the
former import blockers are all resolved (the relaxed reassignment rule,
the jit ``self`` materialization, the annotationless-submit derived
return pin, the inline Out-direction drop, and the with-form spmd InCore
wrap above) — **38/38 modules import as-is, and the assembled L3 program
compiles end-to-end** (entry ``l3_moe`` + its callees, collected
transitively, driven through ``ir.compile`` with the corpus's
``DistributedConfig``: pipeline verification, the Default pass pipeline,
codegen, and ptoas all green — a ``DistributedCompiledProgram`` results).
One deliberate divergence from upstream's specializer remains: a submit
callee whose body has *no* return statement still requires a declared
``->`` (the specializer's last-resort "first ``pl.Out`` param" fallback
is not adopted — it derives from a convention, not a checked trace; the
corpus has no such site). **The remaining blocker is the ptoas version
skew**: the installed ptoas (0.60) is ahead of this branch's pin
(v0.50) and tightened two checks — a non-saturating narrowing ``pto.tcvt``
requires an explicit tmp (0.58+), and a mask-pattern ``pto.tgather``
requires the axis attribute (0.57+). The same failing ``.pto`` files
compile under ptoas 0.54/0.57 — the end-to-end green above was verified
with ``PTOAS_ROOT`` pointing at a 0.54 install (the newest that accepts
this branch's codegen); the failure is identical under the AST-era
``pl.jit_specializer`` (byte-identical ``.pto`` modulo temp counters),
confirming it is toolchain rebase debt, not tracing. The jit device-execution surface
(``.compile(*args, config=RunConfig)`` returning a CompiledProgram, then
dispatch) is served for the eager lane from the decoration record, and the
name-keyed runtime-scalar compile form (``fn.compile(config=...,
scalar=pl.RUNTIME)`` — the golden runner's ``compile_runtime`` path,
corpus-shape in moe.py / decode_layer.py) is declaration-owned: ``RUNTIME``
is a no-op on an already-runtime scalar and a host literal is accepted and
ignored — both pinned in ``TestEagerJitHostSurface`` below.

The cadence pins below are qwen3-family surfaces — v4-pro uses none of
them (zero ``pl.prefetch``/``cross_core_slot``/``defer_wait`` hits; its
30 ``valid_shape=`` sites are all ``pl.slice``, which this repo
supports). Beyond the frontend: ptoas codegen and the multi-rank HCCL
window runtime (``pld`` notify/wait/put, ``DistributedConfig``) are
untested for this corpus.
"""

import pypto.language as pl
import pytest
from pypto.language import distributed as pld
from pypto.pypto_core import ir


# Module-level inline callee: the corpus shape — a module-global standalone
# function reached by bare name (rmsnorm's ``rms_norm`` / ``rms_norm_test``).
# Closure-captured callees adapt through the same shared tail
# (``_adapt_captured_functions``).
@pl.jit.inline
def _inline_copy(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
    y[:] = x


# Module-level sibling callee: the corpus shape — moe.py's ``shared_routed``
# is a module-global ``@pl.jit.incore`` kernel referenced from the ``combine``
# jit body as ``self.shared_routed`` (upstream's jit materializes ``self`` as
# the module's traced-kernel registry). The kernels below must live at module
# level: a kernel nested in a test *method* compiles ``self`` as a closure
# freevar bound to the test instance, so the materialized ``self`` never
# engages there.
@pl.jit.incore
def _self_sibling(
    x: pl.Tensor[[64], pl.FP32],
    y: pl.Out[pl.Tensor[[64], pl.FP32]],
) -> pl.Tensor[[64], pl.FP32]:
    y[:] = x
    return y


@pl.jit.inline
def _self_sibling_caller(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
    _self_sibling(x, y)  # the bare-name call: the module-global callee lane
    self._self_sibling(x, y)  # noqa: F821 - materialized as a sibling by @pl.jit


@pl.jit.inline
def _self_sibling_submitter(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
    _out, _tid = pl.spmd_submit(
        self._self_sibling,  # noqa: F821 - materialized as a sibling by @pl.jit
        x,
        y,
        core_num=1,
    )


# The annotationless variant: no ``->`` on the callee (the corpus shape —
# moe.py's ``shared_routed`` returns its ``pl.Out`` param and declares
# nothing). Upstream's specializer synthesized the inferred ``->`` into its
# rewritten source; the tracing counterpart pins the derived return contract
# onto the Function signature, so the Submit's Tuple[<ret>, TASK_ID] types
# from a recovered fact. Like the sibling kernels above, these must live at
# module level (a test-method kernel binds ``self`` as a closure freevar).
@pl.jit.incore
def _submit_sibling(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
    y[:] = x
    return y


@pl.jit.inline
def _submit_sibling_submitter(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
    _out, _tid = pl.spmd_submit(
        self._submit_sibling,  # noqa: F821 - materialized as a sibling by @pl.jit
        x,
        y,
        core_num=1,
    )


# Path-disagreeing annotationless callee: the derivation must decline —
# pinning UnknownType would type a Submit ``Tuple[Unknown, TASK_ID]`` and
# every caller query on the destructured result would lie. It keeps ``[]``
# and a submit referencing it stays loud (pinned in the class below).
@pl.jit.incore
def _disagree_sibling(
    n: pl.Scalar[pl.INT32],
    x: pl.Tensor[[64], pl.FP32],
    y: pl.Out[pl.Tensor[[64], pl.FP32]],
):
    y[:] = x
    if n > 0:
        return y
    return pl.reshape(y, [1, 64])


# The corpus's standard inline style: a helper that writes through an
# ``pl.Out`` param (hc_pre's ``_hc_pre_separate``, moe's ``x_next`` —
# ~20 modules). Upstream's specializer dropped the wrapper from the
# generated source (with a deprecation warning) because an Inline body is
# spliced at the call site: the parameter already aliases the caller's
# variable. The tracing port kept the direction, so the call passed the
# arg as Out and the caller's later ``return x_next`` tripped the
# InOutUseDiscipline verifier at pipeline_input (first corpus compile
# probe, 2026-09). The jit lanes now mirror the specializer's drop.
@pl.jit.inline
def _inline_out_writer(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
    y[:] = x


@pl.jit
def _inline_out_caller(
    x: pl.Tensor[[64], pl.FP32],
    y: pl.Out[pl.Tensor[[64], pl.FP32]],
) -> pl.Tensor[[64], pl.FP32]:
    _inline_out_writer(x, y)  # the corpus shape: write through the alias, no capture
    return y  # read the caller's own name after the Out-slot pass


class TestEagerJitGlobalCallee:
    def test_eager_entry_calls_module_global_inline_callee(self):
        """A fully-annotated ``@pl.jit`` entry traces at decoration and must
        be able to call a module-global ``@pl.jit.inline`` callee."""

        @pl.jit
        def entry(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            _inline_copy(x, y)
            return y

        assert isinstance(entry.fn, ir.Function)
        # The call must actually emit (not be swallowed): the callee name
        # appears in the traced body's round-trip text.
        assert "_inline_copy" in entry.fn.as_python()


class TestJitSelfSibling:
    def test_jit_body_calls_self_sibling(self):
        """``self.<sibling>(...)`` in a jit body must resolve to the sibling's
        detached-emission Marker — the same call a bare-name reference emits
        (upstream's jit contract; moe.py ``combine``'s constant-false
        specialization branch is the corpus shape)."""

        assert isinstance(_self_sibling_caller.fn, ir.Function)
        assert _self_sibling_caller.fn.as_python().count("_self_sibling") >= 2

    def test_spmd_submit_accepts_self_sibling(self):
        """``pl.spmd_submit(self.<sibling>, ...)`` — the exact corpus shape
        (moe.py:387): the materialized ``self`` must hand the construct the
        sibling's Marker, not a raw value."""

        assert isinstance(_self_sibling_submitter.fn, ir.Function)
        assert "_self_sibling" in _self_sibling_submitter.fn.as_python()


def test_unknown_self_sibling_is_loud():
    """An unresolvable ``self.<name>`` in a jit body is a loud AttributeError
    (the parser frontend's undefined-program-function discipline). A plain
    test function, not a method: a kernel nested in a method captures the
    method's ``self`` as a closure freevar, so the materialized ``self``
    never engages there. The Inline lane defers its trace to first use
    (01-language.md §10), so the loud unknown fires at materialization —
    here, the first ``.fn`` access."""

    @pl.jit.inline
    def entry(x: pl.Tensor[[64], pl.FP32]):
        self.nope(x)  # noqa: F821 - the loud-unknown pin

    with pytest.raises(AttributeError, match="self.nope does not resolve"):
        _ = entry.fn


class TestAnnotationlessSubmitCallee:
    def test_eager_jit_pins_derived_return_types(self):
        """An annotationless eager jit kernel pins the return contract
        derived from its trace onto the Function signature — the tracing
        counterpart of the specializer's synthesized ``->`` (its
        ``_infer_return_type``), so callers and the printer see the real
        type instead of nothing."""

        assert _submit_sibling.fn.return_types
        assert "-> pl.Tensor[[64], pl.FP32]" in _submit_sibling.fn.as_python()

    def test_spmd_submit_accepts_derived_annotationless_sibling(self):
        """The remaining corpus shape (moe.py ``combine`` → ``shared_routed``):
        ``pl.spmd_submit`` on an annotationless callee whose return was
        derived at decoration. The Submit's tuple carries the recovered
        type — never ``UnknownType`` — so the destructured result is
        soundly typed."""

        assert isinstance(_submit_sibling_submitter.fn, ir.Function)
        text = _submit_sibling_submitter.fn.as_python()
        assert "pl.spmd_submit(_submit_sibling" in text
        assert "pl.Tuple[pl.Tensor[[64], pl.FP32], pl.Scalar[pl.TASK_ID]]" in text

    def test_path_disagreement_stays_loud(self):
        """A callee whose return paths disagree types nothing: the derivation
        declines (no ``UnknownType`` pin — a Submit typed
        ``Tuple[Unknown, TASK_ID]`` would lie about every caller query on
        the destructured result), and a submit referencing it raises the
        declared-return rule."""

        assert not _disagree_sibling.fn.return_types

        @pl.jit.inline
        def submitter(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Out[pl.Tensor[[64], pl.FP32]],
        ):
            n = pl.const(1, pl.INT32)
            _out, _tid = pl.spmd_submit(_disagree_sibling, n, x, y, core_num=1)

        # The Inline lane defers its trace to first use (01-language.md
        # §10) — the loud declared-return rule fires at materialization.
        with pytest.raises(TypeError, match="must declare a -> return type"):
            _ = submitter.fn


class TestInlineOutDirectionNormalization:
    def test_eager_inline_lane_drops_out_directions(self):
        """An ``@pl.jit.inline`` kernel's Out/InOut param directions drop to
        In (the specializer's generated-source normalization, warning
        included): the Inline body is spliced at the call site, so the param
        already aliases the caller's variable — a direction left on it
        makes the call pass the arg as Out and every later read of the
        caller's name an InOutUseDiscipline violation at pipeline_input
        verification (the corpus's ``return x_next`` after
        ``hc_post(..., x_next)``)."""

        @pl.jit.inline
        def writer(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            y[:] = x

        # The Inline lane defers its trace to first use (01-language.md
        # §10) — the warning fires at materialization, not decoration.
        with pytest.warns(DeprecationWarning, match="deprecated for inline helpers"):
            _ = writer.fn

        assert all(d == ir.ParamDirection.In for d in writer.fn.param_directions)
        # The printed signature spells the normalized (bare) form.
        assert "pl.Out" not in writer.fn.as_python()

    def test_lazy_inline_lane_drops_out_directions(self):
        """The lazy jit lane (a bare param forces it) applies the same
        normalization on every per-call trace. The warning and the drop are
        one code path (``_drop_inline_directions`` fires both together);
        post-pass the Inline function is spliced away by design (no
        downstream pass observes Inline functions), so the warning is the
        pin."""

        import torch  # noqa: PLC0415 - this test only needs torch for the runtime bridge

        @pl.jit.inline
        def writer(x: pl.Tensor, y: pl.Out[pl.Tensor[[64], pl.FP32]]):  # x shape-bare: forces lazy
            y[:] = x

        x = torch.zeros(64, dtype=torch.float32)
        y = torch.zeros(64, dtype=torch.float32)
        with pytest.warns(DeprecationWarning, match="deprecated for inline helpers"):
            prog = writer.compile_for_test(x, y)
        assert isinstance(prog, ir.Program)

    def test_non_inline_jit_keeps_out_direction(self):
        """``@pl.jit`` (Opaque) keeps its Out directions — the output contract
        of a real function (moe_test's ``x_next``); only the Inline alias
        normalizes."""

        @pl.jit
        def keeper(
            x: pl.Tensor[[64], pl.FP32],
            y: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            y[:] = x
            return y

        assert keeper.fn.param_directions[-1] == ir.ParamDirection.Out

    def test_inline_out_caller_traces(self):
        """The corpus shape end-to-end at the frontend: a caller passes its
        Out param to the inline writer (no capture) and returns its own
        name — the pattern the InOutUseDiscipline verifier rejected before
        the drop. A clean trace with all-In directions on the callee is the
        oracle."""

        assert isinstance(_inline_out_caller.fn, ir.Function)
        assert all(d == ir.ParamDirection.In for d in _inline_out_writer.fn.param_directions)
        assert "_inline_out_writer" in _inline_out_caller.fn.as_python()


class TestSpmdWithFormBodyShape:
    def test_multi_statement_body_wraps_incore(self):
        """The corpus shape (expert_routed's ``with pl.spmd(...)`` around the
        matmul loops, reading ``pl.tile.get_block_idx()``): a multi-statement
        body wraps in the nested InCore region with split name hints, so
        ``OutlineIncoreScopes``/``OutlineClusterScopes`` build the Spmd
        wrapper around an inner call — the shape ``GenerateSpmdCallCode``
        requires. The printer canonicalizes this to the for-form spelling."""

        @pl.jit.inline
        def wrapped(y: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            with pl.spmd(2, name_hint="w"):
                _i = pl.tile.get_block_idx()
                y[:] = pl.add(y, y)

        text = wrapped.fn.as_python()
        assert 'for _i in pl.spmd(2, name_hint="w_spmd")' in text
        # The split Spmd hint (not the user's bare ``w``) and no direct form.
        assert 'with pl.spmd(2, name_hint="w")' not in text

    def test_single_call_body_is_direct_dispatch(self):
        """A single call-valued body statement is the direct-dispatch shape:
        ``SpmdScopeStmt(<the call stmt>)`` with the user's unsplit hint, no
        InCore wrapper (what ``pl.intrin.spmd_direct`` spells in the
        assembly dialect) — the parser's early decision, mirrored."""

        @pl.jit.inline
        def direct(y: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            with pl.spmd(2, name_hint="w"):
                y[:] = pl.add(y, y)

        text = direct.fn.as_python()
        assert 'with pl.spmd(2, name_hint="w")' in text
        assert "pl.incore" not in text
        assert "for _" not in text

    def test_capture_is_orthogonal_to_body_shape(self):
        """``as``-capture adds the ``task_id_var`` attr without changing the
        body shape: a captured single-call body is still direct (the pre-fix
        tracer wrongly wrapped it)."""

        @pl.jit.inline
        def captured(y: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            with pl.spmd(2, name_hint="w") as _tid:
                y[:] = pl.add(y, y)

        text = captured.fn.as_python()
        assert 'with pl.spmd(2, name_hint="w")' in text
        assert "pl.incore" not in text


# The host-surface fixtures, module level (an entry nested in a class body
# resolves its callee through the module's traced-kernel registry, and a
# class-scope name is invisible to method annotations anyway).
_HS_T = pl.dynamic("T_HOST_SURFACE")


@pl.jit.incore
def _hs_writer(x: pl.Tensor[[_HS_T, 64], pl.FP32], y: pl.Tensor[[_HS_T, 64], pl.FP16]):
    y[0:1, 0:64] = pl.cast(x[0:1, 0:64], pl.FP16)


@pl.jit
def _hs_entry(
    x: pl.Tensor[[_HS_T, 64], pl.FP32],
    y: pl.Out[pl.Tensor[[_HS_T, 64], pl.FP16]],
) -> pl.Tensor[[_HS_T, 64], pl.FP16]:
    x.bind_dynamic(0, _HS_T)  # redundant with the annotation — the corpus pattern
    y.bind_dynamic(0, _HS_T)
    _hs_writer(x, y)
    return y


@pl.jit
def _hs_static(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    y[:] = x
    return y


@pl.jit
def _hs_scalars(
    x: pl.Tensor[[_HS_T, 64], pl.FP32],
    y: pl.Out[pl.Tensor[[_HS_T, 64], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[_HS_T, 64], pl.FP32]:
    y[:] = x
    return y


class TestEagerJitHostSurface:
    """The eager jit's host surface — the golden runner's driving contract.

    pypto-lib's ``golden.run(fn=<eager @pl.jit kernel>, ...)`` was the first
    real consumer (the 2026-09 rmsnorm probe): it duck-types
    ``callable(getattr(fn, "compile", None))`` to pick its JIT entry, then
    drives ``fn.compile(*dummy_args, config=RunConfig)`` and dispatches the
    returned CompiledProgram. The eager decoration therefore returns the
    jit object (never a bare ``ir.Function`` — that misrouted the runner
    into its program entry, where ``ir.compile`` rejected the bare
    Function), and ``compile`` serves the decoration record's program:
    dummy args are dispatch vocabulary, not specialization. Re-tracing
    with the bridged static dummy shapes would demote the declared dynamic
    extents to static and trip the kernel's own redundant ``bind_dynamic``
    calls — the corpus's ~1061-call pattern.

    The fixtures live at module level, like the sibling fixtures above: an
    entry nested in a class body resolves its callee through the module's
    traced-kernel registry."""

    def test_the_runner_duck_typing_routes_to_the_jit_entry(self):
        """The decoration result is the jit object with the full surface; the
        decoration trace (what structural consumers read) is ``.fn``."""
        assert callable(getattr(_hs_entry, "compile", None))
        assert not isinstance(_hs_entry, ir.Function)
        assert isinstance(_hs_entry.fn, ir.Function)
        assert "_hs_writer" in _hs_entry.fn.as_python()

    def test_compile_serves_the_decoration_record(self):
        """``compile(*dummy_args, config=...)`` returns the CompiledProgram
        built from the decoration trace — dummy shapes never re-specialize,
        so the redundant ``bind_dynamic`` calls stay no-ops and the declared
        extent survives as a runtime symbol."""
        torch = pytest.importorskip("torch")
        from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415
        from pypto.runtime.runner import RunConfig  # noqa: PLC0415

        x = torch.zeros(8, 64, dtype=torch.float32)
        y = torch.empty(8, 64, dtype=torch.float16)
        compiled = _hs_entry.compile(x, y, config=RunConfig(platform="a2a3sim"))
        assert isinstance(compiled, CompiledProgram)
        # the declaration, not the dummy, typed the artifact
        assert isinstance(_hs_entry.fn.params[0].type.shape[0], ir.Var)

    def test_no_args_compile_is_annotation_determined(self):
        """An eager kernel needs no args to compile: the annotations
        determined the trace at decoration."""
        from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415
        from pypto.runtime.runner import RunConfig  # noqa: PLC0415

        compiled = _hs_entry.compile(config=RunConfig(platform="a2a3sim"))
        assert isinstance(compiled, CompiledProgram)

    def test_static_shape_contradiction_retraces(self):
        """A statically annotated eager kernel called at a different shape
        keeps the per-call re-trace: the artifact follows the call, exactly
        as it did before the decoration record existed."""
        torch = pytest.importorskip("torch")

        prog = _hs_static.compile_for_test(torch.empty(64), torch.empty(64))
        (entry,) = prog.functions.values()
        assert entry.params[0].type.shape[0] == 64

        prog128 = _hs_static.compile_for_test(torch.empty(128), torch.empty(128))
        (entry128,) = prog128.functions.values()
        assert entry128.params[0].type.shape[0] == 128

    def test_name_keyed_scalar_vocabulary_is_declaration_owned(self):
        """The ``compile_runtime`` form (moe.py's mixed call — one RUNTIME, one
        literal, no tensor args): the declaration pinned every scalar runtime,
        so ``pl.RUNTIME`` is a no-op, the literal is accepted and ignored
        (dispatch supplies the value), and the artifact keeps both scalars as
        real parameters — nothing specializes."""
        from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415
        from pypto.runtime.runner import RunConfig  # noqa: PLC0415

        compiled = _hs_scalars.compile(
            config=RunConfig(platform="a2a3sim"),
            layer_id=3,
            num_tokens=pl.RUNTIME,
        )
        assert isinstance(compiled, CompiledProgram)
        names = [p.name_hint for p in _hs_scalars.fn.params]
        assert names[-2:] == ["layer_id", "num_tokens"], "no scalar specialized away"

    def test_name_keyed_vocabulary_is_loud_where_the_declaration_cannot_speak(self):
        """Unknown names, host-knob parameters (the marker cannot retype a
        plain annotation), and non-scalar values all raise — compile is never
        looser than the call boundary it serves."""
        from pypto.runtime.runner import RunConfig  # noqa: PLC0415

        with pytest.raises(TypeError, match="has no parameter 'nope'"):
            _hs_scalars.compile(config=RunConfig(platform="a2a3sim"), nope=1)

        with pytest.raises(TypeError, match="host knob by declaration"):

            @pl.jit
            def host_knob_entry(
                x: pl.Tensor[[64], pl.FP32],
                scale: float,
                y: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y[:] = pl.mul(x, pl.const(scale, pl.FP32))
                return y

            host_knob_entry.compile(scale=2.0)

        torch = pytest.importorskip("torch")
        with pytest.raises(TypeError, match="scalar values only"):
            _hs_scalars.compile(config=RunConfig(platform="a2a3sim"), layer_id=torch.zeros(3))


class TestLazyCalleeDistributedWindow:
    def test_lazy_callee_lane_preserves_distributed_tensor_type(self):
        """A callee reached through the lazy lane (bare param →
        ``_JitSpecializer``) must receive the caller's
        ``pld.DistributedTensor`` facade unchanged — no retype at the call
        boundary. The in-body ``pld.tensor.put`` validates the window type
        at trace time, so a clean trace is the type-preservation oracle."""

        @pl.jit.inline
        def put_one(
            w: pld.DistributedTensor[[16, 64], pl.BF16],
            y: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
            x: pl.Tensor,  # bare: forces the lazy callee lane
        ):
            pld.tensor.put(
                dst=w,
                peer=0,
                src=y,
                dst_offsets=[0, 0],
                src_offsets=[0, 0],
                shape=[1, 64],
            )

        @pl.jit
        def entry(
            x: pl.Tensor[[16, 64], pl.BF16],
            w: pld.DistributedTensor[[16, 64], pl.BF16],
            y: pl.Out[pl.Tensor[[16, 64], pl.BF16]],
        ):
            put_one(w, y, x)
            return y

        assert isinstance(entry.fn, ir.Function)
        assert "put_one" in entry.fn.as_python()


class TestLazyCalleeArrayArg:
    def test_lazy_callee_lane_accepts_task_id_array_argument(self):
        """A ``pl.Array`` facade argument must cross the jit callee boundary
        with its type intact. Corpus shape: qwen3_14b ``decode_fwd`` passing
        ``scratch_ready: pl.Array[1, pl.TASK_ID]`` into ``_decode_fwd_body``."""

        @pl.jit.inline
        def body(
            ready: pl.Array[1, pl.TASK_ID],
            y: pl.Out[pl.Tensor[[64], pl.FP32]],
            x: pl.Tensor,  # bare: forces the lazy callee lane
        ):
            _tid = ready[0]

        @pl.jit
        def entry(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            ready = pl.array.create(1, pl.TASK_ID)
            body(ready, y, x)
            return y

        assert isinstance(entry.fn, ir.Function)
        assert "body" in entry.fn.as_python()


class TestAnnotationlessInexpressibleMultiReturn:
    """An annotationless inline callee whose derived *signature* pin declines
    (a body-local extent — e.g. ``t_dim * const`` — is unspellable in a printed
    ``->``) still types its callers' Call. Closed 2026-09-14: the jit lanes
    degraded the call to an untyped Scalar, so the caller's unpack raised
    ``cannot unpack non-iterable Scalar object`` — 13 corpus files' shape is
    ``decode_sparse_attn_swa``'s ``sparse_attn_swa`` → ``sparse_attn_swa_tp1``
    8-tuple handoff (t_blk-computed extents). The parser always typed the call
    (``_effective_return_types``); the tracer now mirrors it through
    ``Marker.body_derived_return_types`` while the signature pin keeps its
    all-or-nothing guard. The program-lane anchor lives in
    ``test_call.py::test_annotationless_inexpressible_multi_return_call_anchor``."""

    def test_inline_lane_unpack(self):
        @pl.jit.inline
        def producer(x: pl.Tensor[[16], pl.FP32], y: pl.Out[pl.Tensor[[16], pl.FP32]]):
            with pl.spmd(2, name_hint="work") as tid:
                b = pl.tile.get_block_idx()
                pl.store(pl.load(x, [b * 8], [8]), [b * 8], y)
            n = pl.tensor.dim(x, 0)
            n2 = n * 2  # computed extent — unspellable in a signature
            aux = pl.create_tensor([n2, 1], dtype=pl.FP32)
            return aux, tid

        @pl.jit.inline
        def caller(x: pl.Tensor[[16], pl.FP32], y: pl.Out[pl.Tensor[[16], pl.FP32]]):
            aux2, tid2 = producer(x, y)
            with pl.at(level=pl.Level.CORE_GROUP, deps=[tid2]):
                pl.store(pl.load(x, [0], [8]), [8], y)
            return y

        # The eager jit lane's decoration result is the specializer envelope;
        # its traced artifact is ``.fn``.
        assert isinstance(caller.fn, ir.Function)


class TestInlineSpliceTypeClosure:
    """The inline splice's substitution must be closed over Var *types*.

    Closed 2026-09-14, from the pypto-lib census (7 dspark/mtp files failed
    ``ConvertToSSA`` with ``Variable '…' used outside its defining scope``).
    Two leaks, one per mechanism:

    * **Same-splice** — ``CloneInlineBody`` seeds its fresh replacement Vars
      with the callee's type verbatim (the seed exists before the traversal,
      the complete substitution table only after), so a replacement Var's type
      kept referencing the callee's local ``tensor.dim`` def. Fixed in
      ``DeepClone``: after the traversal, every replacement Var whose type
      embeds a substituted Var is rebuilt with the remapped type.
    * **Cross-splice** — a callee traced with a dynamic-extent arg embeds the
      *caller-trace's* local in its param/loop-carried types (the variant
      lane), and a caller local typed from an inline callee's return type
      embeds the *callee's* local. Neither is a substitution key of the splice
      that meets them, so the renames accumulate in ``SpliceRenames`` and one
      repair sweep runs after the fixpoint over the kept functions.

    The corpus shape (``prefill_hca``): an Orchestration entry calling an
    inline callee whose ``tensor.dim``-derived local reaches both a nested
    inline callee's carried rebinds and the caller's unpacked local. The pin
    compiles — trace-level assertions cannot see this (the IR only breaks in
    the pass pipeline)."""

    def test_two_level_chain_completes_the_pipeline(self):
        import torch  # noqa: PLC0415 - runtime bridge for compile_for_test

        n_dyn = pl.dynamic("N_DYN")
        b_rows = pl.dynamic("B_ROWS_DYN")

        @pl.jit.inline
        def commit(
            flat_b: pl.InOut[pl.Tensor[[b_rows, 32], pl.BF16]], x: pl.Tensor[[n_dyn, 32], pl.FP32]
        ):
            # Param type minted from the caller's arg (variant lane); the
            # carried rebind's type embeds the caller-trace's local.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="commit"):
                for row in pl.range(4):
                    v = pl.read(x, [row, 0])
                    if v > pl.const(0, pl.FP32):
                        slot = pl.cast(pl.read(x, [row + 1, 0]), pl.INDEX)
                        if slot >= 0:
                            tile = pl.cast(pl.tensor.slice(x, [1, 32], [row, 0]), pl.BF16)
                            flat_b = pl.tensor.assemble(flat_b, tile, [slot, 0])
            return flat_b

        @pl.jit.inline
        def producer(x: pl.Tensor[[n_dyn, 32], pl.FP32], y: pl.Out[pl.Tensor[[n_dyn, 32], pl.BF16]]):
            with pl.spmd(2, name_hint="work") as tid:
                b = pl.tile.get_block_idx()
                pl.store(pl.cast(pl.load(x, [b * 32, 0], [32, 32]), pl.BF16), [b * 32, 0], y)
            d = pl.tensor.dim(y, 0)  # runtime extent, unfolded in Inline bodies
            flat = pl.reshape(y, [d, 32])
            flat = commit(flat, x)
            return flat, tid

        @pl.jit
        def entry(x: pl.Tensor[[n_dyn, 32], pl.FP32], y: pl.Out[pl.Tensor[[n_dyn, 32], pl.BF16]]):
            flat, tid = producer(x, y)
            with pl.at(level=pl.Level.CORE_GROUP, deps=[tid]):
                pl.store(pl.cast(pl.load(x, [0, 0], [32, 32]), pl.BF16), [32, 0], y)
            return y

        prog = entry.compile_for_test(
            torch.zeros(64, 32, dtype=torch.float32), torch.zeros(64, 32, dtype=torch.bfloat16)
        )
        assert isinstance(prog, ir.Program)


class TestSpmdDepsWithoutCapture:
    """``deps=`` on the uncaptured ``pl.spmd`` forms — closed 2026-09-14.

    The tracer rejected ``deps=`` on the plain with-form and the for-form
    while the parser (the round-trip owner) accepts every form and the
    Spmd outliner synthesises the uncaptured TaskId Var — a surface
    asymmetry that broke 13 corpus files at decoration. Corpus shapes:
    ``gate.py``'s for-form quant block (deps + allow_early_resolve) and
    ``expert_routed.py``'s plain with-form expert scopes. The conformance
    anchors live in ``test_spmd_submit_only_kwargs.py``; these pins hold
    the corpus shapes (clean trace is the oracle)."""

    def test_for_form_deps_with_early_resolve(self):
        @pl.function
        def k(y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="seed") as tid0:
                t = pl.load(y, [0], [8])
                pl.store(t, [0], y)
            for b in pl.spmd(8, name_hint="gate_quant", deps=[tid0], allow_early_resolve=True):
                pl.store(pl.load(y, [b * 8], [8]), [b * 8], y)

        assert isinstance(k, ir.Function)

    def test_plain_with_form_deps(self):
        @pl.function
        def k(y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="seed") as tid0:
                t = pl.load(y, [0], [8])
                pl.store(t, [0], y)
            with pl.spmd(8, name_hint="expert_mm", deps=[tid0]):
                i = pl.tile.get_block_idx()
                pl.store(pl.load(y, [i * 8], [8]), [i * 8], y)

        assert isinstance(k, ir.Function)


class TestSpmdDepsFromTaskIdArray:
    def test_spmd_deps_accept_task_id_array_element(self):
        """``deps=[tids[i]]`` (an ``Array[_, TASK_ID]`` element read) must
        lower the way the AST parser's Form 1 does — a fresh synthesized
        ``Array[K, TASK_ID]`` — so the builder accepts the emitted
        ``manual_dep_edges``; a clean trace is the oracle."""

        @pl.function
        def k(y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            tids = pl.array.create(2, pl.TASK_ID)
            for i in pl.parallel(2):
                with pl.spmd(2, name_hint="producer") as tid:
                    tids[i] = tid  # type: ignore[reportArgumentType]  # parallel() loop var stub
            with pl.spmd(2, name_hint="consumer", deps=[tids[0]]) as tid:
                pass

        assert isinstance(k, ir.Function)

    def test_task_dummy_deps_accept_task_id_array_element(self):
        """``pl.system.task_dummy(deps=[arr[i]])`` — the same Form-1 element
        read through the dependency-only barrier. ``manual_dep_edges`` is
        Var-only, so the per-element read must fold through the shared
        ``deps=`` coercion (``_coerce_deps``) exactly like the with-form's
        ``deps=``; a clean trace is the oracle. Corpus shape:
        ``prefill_sparse_attn``'s wave-completion fence and the dspark MoE
        scatter fan-ins."""

        @pl.function
        def k(y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            tids = pl.array.create(2, pl.TASK_ID)
            for i in pl.parallel(2):
                with pl.spmd(2, name_hint="producer") as tid:
                    tids[i] = tid  # type: ignore[reportArgumentType]  # parallel() loop var stub
            fence = pl.system.task_dummy(deps=[tids[0], None])
            with pl.spmd(2, name_hint="consumer", deps=[fence]) as tid:
                pass

        assert isinstance(k, ir.Function)


class TestTileSubscriptWrite:
    def test_tile_setitem_lowering(self):
        """``sq[0:1, :] = row`` must lower to ``tile.assemble`` (the AST's
        subscript-write sugar) — the tile-target counterpart of
        ``Tensor.__setitem__``. The source must be a Tile (the parser's
        same-kind rule: ``tile.assemble`` takes a TileType source)."""

        @pl.function
        def k(y: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            sq = pl.create_tile([64, 64], dtype=pl.FP32)
            row = pl.tile.full([1, 64], dtype=pl.FP32, value=1.0)
            sq[0:1, :] = row
            pl.tile.store(sq, [0, 0], y)

        assert isinstance(k, ir.Function)


class TestCadenceValidShapeKwarg:
    """CADENCE PINS — NOT fixable in the current repo state.

    These spellings are post-fork upstream API (see the class methods for
    the commits). The corpus is correct against upstream; fixing them here
    means adopting/rebasing the upstream commits — do not "fix" by
    changing the test to the pre-fork spelling."""

    def test_cadence_load_accepts_upstream_singular_valid_shape(self):
        """``pl.load(..., valid_shape=...)`` is the post-fork upstream
        spelling (``2e54cd660`` renamed ``valid_shapes`` → ``valid_shape``).
        Cadence pin: the corpus is right against upstream; the fix here is
        adopting (or aliasing) the rename."""

        @pl.function
        def k(
            x: pl.Tensor[[16, 32], pl.FP32],
            y: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
        ):
            t = pl.load(x, [0, 0], [16, 16], valid_shape=[16, 8])
            pl.tile.store(t, [0, 0], y)

        assert isinstance(k, ir.Function)

    def test_cadence_gather_row_accepts_upstream_valid_shape(self):
        """``gather_row(..., valid_shape=...)`` — the same post-fork kwarg
        family on a second op (``#2142``). Corpus shape:
        ``decode_sparse_attn_csa``'s dynamic-row gathers."""

        @pl.function
        def k(
            acc: pl.Tensor[[8, 64], pl.FP32],
            src: pl.Tensor[[16, 64], pl.FP32],
        ):
            pl.gather_row(
                acc,
                src,
                [4, 0],
                [0, 0],
                [8, 64],
                valid_shape=[4, 64],
            )

        assert isinstance(k, ir.Function)


class TestCadenceUpstreamSurface:
    """CADENCE PINS — NOT fixable in the current repo state.

    Post-fork upstream surface the corpus uses unguarded (commits cited per
    method). The fix is adopting the upstream commits on this branch; until
    then the failures reflect a rebase debt, not a tracing bug. Do not
    "fix" by deleting the corpus spelling."""

    def test_cadence_prefetch_context_traces(self):
        """``pl.prefetch.*`` (#2089) — the async GM→L2 prefetch frontend.
        Corpus shape: ``decode_swa``'s warm-context prefetch scopes."""

        @pl.function
        def k(x: pl.Tensor[[64, 64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="warm") as _tid:
                # The ignores are part of the pin: pl.prefetch does not exist on
                # this branch yet — remove them with the xfail entry (#2089).
                ctx = pl.prefetch.make_context()  # type: ignore[reportAttributeAccessIssue]
                pl.prefetch.async_prefetch(x, ctx)  # type: ignore[reportAttributeAccessIssue]

        assert isinstance(k, ir.Function)

    def test_cadence_cross_core_slot_optimization_traces(self):
        """``pl.cross_core_slot(slot_num=N)`` as an ``optimizations=`` entry
        (upstream split of the NONE spelling's slot count). The traced
        ``_parse_spmd_optimizations`` only accepts ``pl.split`` entries.
        Corpus shape: ``lm_head``/``decode_o_proj`` slot pipelines."""

        @pl.function
        def k(y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            with pl.spmd(
                2,
                name_hint="slot_pipe",
                optimizations=[pl.cross_core_slot(slot_num=2)],  # type: ignore[reportAttributeAccessIssue]  # part of the pin
            ) as _tid:
                pass

        assert isinstance(k, ir.Function)

    def test_cadence_defer_wait_traces(self):
        """``pld.system.defer_wait`` (#2369) — deferred task completion
        waits on a window signal. Corpus shape: ``prefill_o_proj``'s
        weight-reuse fencing."""

        @pl.function
        def k(
            signal: pld.DistributedTensor[[2, 1], pl.INT32],
            epoch: pl.Scalar[pl.INT32],
        ):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="reuse_wait") as _tid:
                for peer in pl.range(2):
                    pld.system.defer_wait(
                        signal=signal,
                        offsets=[peer, 0],
                        expected=epoch,
                        cmp=pld.WaitCmp.Ge,
                    )

        assert isinstance(k, ir.Function)


class TestBareReturnContract:
    """The under-specified return contract: a bare facade class (``-> pl.Tensor``;
    a bare element of ``tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]`` — the corpus's
    submit-style multi-return) states the value's *kind* only. Shape and dtype
    are derived from the traced body and the kind is class-checked (typing from
    values: the annotation states no shape, so it cannot lie, it under-specifies).
    Before the fix the bare *element* form raised ``cannot derive an IR type from
    annotation`` at decoration while the top-level bare form silently degraded
    to an unchecked empty contract — an internal inconsistency the corpus never
    met because the AST-era specializer never converted annotations at
    decoration time."""

    def test_bare_tensor_in_tuple_multi_return(self):
        """The corpus shape: dspark/mtp ``decode_swa``-style helpers declare
        ``-> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]`` — a bare Tensor element
        beside a concrete TASK_ID scalar. The composed contract pins the traced
        types; the scalar slot stays checked."""

        @pl.jit.inline
        def submit_style(
            x: pl.Tensor[[64], pl.FP32],
        ) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
            tids = pl.array.create(2, pl.TASK_ID)
            for i in pl.parallel(2):
                with pl.spmd(2, name_hint="producer") as tid:
                    tids[i] = tid  # type: ignore[reportArgumentType]  # parallel() loop var stub
            return pl.add(x, x), tids[0]

        fn = submit_style.fn
        assert isinstance(fn, ir.Function)
        assert len(fn.return_types) == 2
        assert isinstance(fn.return_types[0], ir.TensorType)
        assert fn.return_types[1] == ir.ScalarType(pl.TASK_ID)

    def test_top_level_bare_tensor_return(self):
        """``-> pl.Tensor`` (top-level) — the same rule at one slot: derived
        shape/dtype, class-checked kind. No longer a silent empty contract."""

        @pl.function
        def single(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor:
            return pl.add(x, x)

        assert isinstance(single, ir.Function)
        assert len(single.return_types) == 1
        assert isinstance(single.return_types[0], ir.TensorType)

    def test_bare_kind_mismatch_is_loud(self):
        """A bare ``-> pl.Scalar`` over a tensor-returning body is the loud
        'the `->` was a lie' — the class check, not dead text."""

        with pytest.raises(TypeError, match="declared kind does not match"):

            @pl.function
            def wrong(x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar:
                return pl.add(x, x)

    def test_bare_arity_mismatch_is_loud(self):
        """A one-slot bare contract over a two-value body is loud."""

        with pytest.raises(TypeError, match="slot"):

            @pl.function
            def pair(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor:
                return pl.add(x, x), pl.sub(x, x)


class TestInlineForwardReference:
    """The Inline lane defers its trace to first use (01-language.md §10):
    an Inline callee has no dispatch arguments, so tracing at decoration
    buys nothing — and eager execution before the module finishes loading
    turned a definition-order choice into a raw NameError. The AST-era jit
    decorated lazily, so corpus trees (dspark ``decode_indexer``'s
    ``indexer_topk_leaf`` → ``indexer_topk_full_leaf``, the
    ``prefill_indexer_query`` chain) freely defined callees after their
    callers. Def order is free again; an unresolvable name is still loud —
    just at first use."""

    def test_callee_defined_after_caller(self):
        @pl.jit.inline
        def caller(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor:
            return callee(x)

        @pl.jit.inline
        def callee(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor:
            return pl.add(x, x)

        # First use after both defs: the materialized trace resolves the
        # global callee — no NameError.
        assert "callee(" in caller.fn.as_python()

    def test_forward_ref_inside_a_caller_trace(self):
        """The corpus's actual chain shape: an inline helper referencing a
        later-defined inline helper, itself called by an eagerly traced
        entry — the whole chain materializes at the entry's decoration,
        when the module is fully loaded."""

        @pl.jit.inline
        def mid(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor:
            return leaf(x)

        @pl.jit.inline
        def leaf(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor:
            return pl.add(x, x)

        @pl.jit
        def entry(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            y[:] = mid(x)

        assert "mid(" in entry.fn.as_python()
        assert "leaf(" in mid.fn.as_python()

    def test_incore_lane_defers_the_same_way(self):
        """The InCore lane is the same callee tier — the corpus chain that
        stayed broken after the Inline fix (dspark ``decode_indexer``: an
        ``@pl.jit.incore`` caller referencing a later-defined sibling)."""

        @pl.jit.incore
        def incore_caller(q: pl.Tensor[[64], pl.FP32]) -> pl.Tensor:
            return incore_leaf(q)

        @pl.jit.incore
        def incore_leaf(q: pl.Tensor[[64], pl.FP32]) -> pl.Tensor:
            return pl.add(q, q)

        assert "incore_leaf(" in incore_caller.fn.as_python()

    def test_unresolvable_name_still_loud_at_first_use(self):
        @pl.jit.inline
        def broken(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor:
            return nowhere(x)  # noqa: F821 - the loud-unknown pin

        with pytest.raises(NameError, match="nowhere"):
            _ = broken.fn
