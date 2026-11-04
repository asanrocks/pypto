# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Logical entry results survive ABI lowering and artifact reload in both frontends."""

import json

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto._artifact_contract import ArtifactExecutionMode
from pypto.ir._compile_contract import validate_kernel_program
from pypto.ir._entry_outputs import materialize_entry_outputs
from pypto.ir.compiled_program import CompiledProgram
from pypto.ir.param_info import _extract_func_param_infos


def _metadata(compiled: CompiledProgram):
    """The artifact metadata of a freshly compiled program, asserted present.

    ``ir.compile`` returns the distributed flavour when the program asks for it
    and ``CompiledArtifact.metadata`` is optional until the sidecar is read, so
    these single-chip cases narrow both before inspecting delivery metadata.
    """
    artifact = compiled._artifact
    assert artifact.metadata is not None
    return artifact.metadata


def _created(x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
    out = pl.create_tensor([8, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [8, 8]), 1.0), [0, 0], out)
    return out


def _unannotated(x: pl.Tensor[[8, 8], pl.FP32]):
    out = pl.create_tensor([8, 8], dtype=pl.FP32)
    return out


def _wrong_type(x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[16, 8], pl.FP32]:
    out = pl.create_tensor([8, 8], dtype=pl.FP32)
    return out


def _different_output(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    created = pl.create_tensor([16, 8], dtype=pl.FP32)
    return created


def _mixed(
    x: pl.Tensor[[8, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
) -> tuple[pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32]]:
    created = pl.create_tensor([8, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, 1.0), [0, 0], created)
        pl.store(tile, [0, 0], out)
    return created, out, created


def _mixed_trace(
    x: pl.Tensor[[8, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
) -> tuple[pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32]]:
    created = pl.create_tensor([8, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, 1.0), [0, 0], created)
        pl.store(tile, [0, 0], out)
    return pl.Tuple(created, out, created)


def _mixed_structural_tuple_trace(
    x: pl.Tensor[[8, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
) -> pl.Tuple[pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32]]:
    """One structural ``pl.Tuple`` slot: an Out-backed element beside a created one."""
    created = pl.create_tensor([8, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, 1.0), [0, 0], created)
        pl.store(tile, [0, 0], out)
    return pl.Tuple(created, out)


def _manual_dep_created(x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
    out = pl.create_tensor([8, 8], dtype=pl.FP32, manual_dep=True)
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [8, 8]), 1.0), [0, 0], out)
    return out


def _reordered(
    x: pl.Tensor[[8, 8], pl.FP32],
    a: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    b: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
) -> tuple[pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32]]:
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, 1.0), [0, 0], a)
        pl.store(pl.add(tile, 2.0), [0, 0], b)
    return b, a


def _reordered_host(
    x: pl.Tensor[[8, 8], pl.FP32],
    a: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    b: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
) -> tuple[pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32]]:
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, 1.0), [0, 0], a)
        pl.store(pl.add(tile, 2.0), [0, 0], b)
    return b, a


def _returned_inout(
    x: pl.Tensor[[8, 8], pl.FP32],
    acc: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
) -> pl.Tensor[[8, 8], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(acc, [0, 0], [8, 8]), pl.load(x, [0, 0], [8, 8])), [0, 0], acc)
    return acc


def _duplicated(
    x: pl.Tensor[[8, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
) -> tuple[pl.Tensor[[8, 8], pl.FP32], pl.Tensor[[8, 8], pl.FP32]]:
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [8, 8]), 1.0), [0, 0], out)
    return out, out


def _manual_dep_scratch(x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
    scratch = pl.create_tensor([8, 8], dtype=pl.FP32, manual_dep=True)
    out = pl.create_tensor([8, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, 1.0), [0, 0], scratch)
        pl.store(pl.load(scratch, [0, 0], [8, 8]), [0, 0], out)
    return out


@pytest.mark.parametrize("decorator", [pl.jit, pl.jit_trace])
def test_created_output_compiles_and_reloads(decorator, tmp_path, monkeypatch):
    program = decorator(_created).specialize()
    entry = next(f for f in program.functions.values() if f.name == "_created")
    assert len(entry.return_types) == 1
    compiled = ir.compile(program, output_dir=str(tmp_path), platform="a2a3sim", skip_ptoas=True)
    # ``ir.compile`` returns the distributed flavour when the program asks for
    # it; these cases are single-chip, so the narrow type is the one under test.
    assert isinstance(compiled, CompiledProgram)
    metadata = _metadata(compiled)
    assert compiled.param_names == ["x"]
    assert len(metadata.parameters) == 1
    assert len(metadata.synthesized_outputs) == 1
    assert metadata.result_indices == (1,)
    assert compiled.result_indices == [1]
    assert metadata.execution_capabilities.modes == (ArtifactExecutionMode.PROGRAM,)

    # Isolate host delivery from native execution; full codegen above is real.
    def execute(_directory, args, **_kwargs):
        args[1].copy_(args[0] + 1)

    monkeypatch.setattr("pypto.runtime.runner._execute_compiled", execute)
    x = torch.randn(8, 8)
    for callable_ in (compiled, CompiledProgram.from_dir(tmp_path)):
        first = callable_(x)
        second = callable_(x)
        assert first is not second
        torch.testing.assert_close(first, x + 1)
        torch.testing.assert_close(second, x + 1)


@pytest.mark.parametrize("decorator", [pl.jit, pl.jit_trace])
def test_mixed_results_preserve_order_and_aliasing(decorator, tmp_path, monkeypatch):
    program = decorator(_mixed_trace if decorator is pl.jit_trace else _mixed).specialize()
    compiled = ir.compile(program, output_dir=str(tmp_path), platform="a2a3sim", skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    assert _metadata(compiled).result_indices == (2, 1, 2)
    assert compiled.result_indices == [2, 1, 2]

    def execute(_directory, args, **_kwargs):
        args[1].copy_(args[0])
        args[2].copy_(args[0] + 1)

    monkeypatch.setattr("pypto.runtime.runner._execute_compiled", execute)
    x = torch.randn(8, 8)
    for callable_ in (compiled, CompiledProgram.from_dir(tmp_path)):
        out = torch.empty_like(x)
        first, second, third = callable_(x, out)
        assert first is third
        assert second is out
        torch.testing.assert_close(first, x + 1)
        torch.testing.assert_close(second, x)


def test_a_mixed_structural_tuple_delivers(tmp_path, monkeypatch):
    """The structural spelling of the mixed case: one declared ``pl.Tuple``
    slot holding an Out-backed element beside a created one.

    The created-output path asked the position-faithful map — one tuple slot
    read ``[None]`` — so the created element could not materialize and
    compile refused the entry. Flattening the slot the way the early exit
    does lets the ABI adapt it like the flat spelling."""
    program = pl.jit_trace(_mixed_structural_tuple_trace).specialize()
    compiled = ir.compile(program, output_dir=str(tmp_path), platform="a2a3sim", skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    assert _metadata(compiled).result_indices == (2, 1)
    assert compiled.result_indices == [2, 1]

    def execute(_directory, args, **_kwargs):
        args[1].copy_(args[0])
        args[2].copy_(args[0] + 1)

    monkeypatch.setattr("pypto.runtime.runner._execute_compiled", execute)
    x = torch.randn(8, 8)
    for callable_ in (compiled, CompiledProgram.from_dir(tmp_path)):
        out = torch.empty_like(x)
        first, second = callable_(x, out)
        assert second is out
        torch.testing.assert_close(first, x + 1)
        torch.testing.assert_close(second, x)


@pytest.mark.parametrize("decorator", [pl.jit, pl.jit_trace])
def test_reordered_parameter_backed_results_deliver_in_return_order(decorator, tmp_path, monkeypatch):
    """A fully parameter-backed result set still needs its provenance map.

    The runtime's no-metadata fallback is the pure Out parameters in
    declaration order, so a kernel returning ``(b, a)`` against params
    ``a, b`` would silently deliver ``(a, b)``. The delivered map must be
    persisted whenever it disagrees with that fallback.
    """
    program = decorator(_reordered).specialize()
    compiled = ir.compile(program, output_dir=str(tmp_path), platform="a2a3sim", skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    assert _metadata(compiled).result_indices == (2, 1)
    assert compiled.result_indices == [2, 1]

    def execute(_directory, args, **_kwargs):
        args[1].copy_(args[0] + 1)
        args[2].copy_(args[0] + 2)

    monkeypatch.setattr("pypto.runtime.runner._execute_compiled", execute)
    x = torch.randn(8, 8)
    for callable_ in (compiled, CompiledProgram.from_dir(tmp_path)):
        first, second = callable_(x)
        torch.testing.assert_close(first, x + 2)
        torch.testing.assert_close(second, x + 1)


def test_a_host_entry_delivers_reordered_results(tmp_path, monkeypatch):
    """The host entry's result contract survives compilation and reload.

    A host entry reaches its chip orchestrators through traced calls, so its
    own return order is provenance the ABI must carry — not something the
    dispatch path may recover from parameter declarations.
    """
    program = pl.jit_trace.host(_reordered_host).specialize()
    compiled = ir.compile(program, output_dir=str(tmp_path), platform="a2a3sim", skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    assert compiled.param_names == ["x", "a", "b"]
    assert compiled.result_indices == [2, 1]
    assert _metadata(compiled).result_indices == (2, 1)

    def execute(_directory, args, **_kwargs):
        args[1].copy_(args[0] + 1)
        args[2].copy_(args[0] + 2)

    monkeypatch.setattr("pypto.runtime.runner._execute_compiled", execute)
    x = torch.randn(8, 8)
    for callable_ in (compiled, CompiledProgram.from_dir(tmp_path)):
        first, second = callable_(x)
        torch.testing.assert_close(first, x + 2)
        torch.testing.assert_close(second, x + 1)


@pytest.mark.parametrize("decorator", [pl.jit, pl.jit_trace])
def test_an_inout_returned_result_is_read_back(decorator, tmp_path, monkeypatch):
    """An InOut parameter is not an auto-allocatable output, but a result
    that returns it is still delivered by provenance, not dropped."""
    program = decorator(_returned_inout).specialize()
    compiled = ir.compile(program, output_dir=str(tmp_path), platform="a2a3sim", skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    assert compiled.output_indices == []
    assert compiled.result_indices == [1]
    assert _metadata(compiled).result_indices == (1,)

    def execute(_directory, args, **_kwargs):
        args[1].copy_(args[1] + args[0])

    monkeypatch.setattr("pypto.runtime.runner._execute_compiled", execute)
    x = torch.randn(8, 8)
    for callable_ in (compiled, CompiledProgram.from_dir(tmp_path)):
        acc = torch.randn(8, 8)
        expected = acc + x
        result = callable_(x, acc)
        assert result is acc
        torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("decorator", [pl.jit, pl.jit_trace])
def test_a_duplicate_result_preserves_the_alias(decorator, tmp_path, monkeypatch):
    """``return out, out`` keeps one buffer aliased across both slots."""
    program = decorator(_duplicated).specialize()
    compiled = ir.compile(program, output_dir=str(tmp_path), platform="a2a3sim", skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    assert compiled.result_indices == [1, 1]
    assert _metadata(compiled).result_indices == (1, 1)

    def execute(_directory, args, **_kwargs):
        args[1].copy_(args[0] + 1)

    monkeypatch.setattr("pypto.runtime.runner._execute_compiled", execute)
    x = torch.randn(8, 8)
    for callable_ in (compiled, CompiledProgram.from_dir(tmp_path)):
        first, second = callable_(x)
        assert first is second
        torch.testing.assert_close(first, x + 1)
        out = torch.empty_like(x)
        assert callable_(x, out) is None  # in-place style hands nothing back


def test_kernel_contract_stays_borrowed():
    program = pl.jit(_created).specialize()
    entry = next(iter(program.functions.values()))
    params, _, _ = _extract_func_param_infos(entry)
    with pytest.raises(ValueError, match="alias external tensor"):
        validate_kernel_program(program, params)


def test_dynamic_created_output_is_rejected():
    program = pl.parse("""@pl.program
class Dynamic:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[n, 8], pl.FP32]:
        out = pl.create_tensor([n, 8], dtype=pl.FP32)
        return out
""")
    assert isinstance(program, ir.Program)
    with pytest.raises(TypeError, match="static tensor shapes"):
        materialize_entry_outputs(program, next(iter(program.functions.values())))


def test_unknown_storage_is_rejected():
    """A result neither a whole parameter nor a fresh top-level create has
    no storage the ABI could hand the caller — fail rather than guess."""
    program = pl.parse("""@pl.program
class ScopedCreate:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.create_tensor([8, 8], dtype=pl.FP32)
            pl.store(pl.add(pl.load(x, [0, 0], [8, 8]), 1.0), [0, 0], out)
        return out
""")
    assert isinstance(program, ir.Program)
    with pytest.raises(TypeError, match="result provenance is incomplete"):
        materialize_entry_outputs(program, next(iter(program.functions.values())))


def test_created_view_result_is_rejected():
    """A view whose descriptor differs from any parameter's whole buffer has
    no delivery mapping — fail rather than hand the caller the whole buffer."""
    program = pl.parse("""@pl.program
class ViewReturn:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 4], pl.FP32]:
        return x[0:8, 0:4]
""")
    assert isinstance(program, ir.Program)
    with pytest.raises(TypeError, match="result provenance is incomplete"):
        materialize_entry_outputs(program, next(iter(program.functions.values())))


def test_manual_dep_created_output_is_rejected_at_the_abi_boundary():
    """Promotion deletes the create statement — sole carrier of the
    lifetime-granularity OverlapMap opt-out — so it must fail, not drop the
    semantic silently."""
    program = pl.parse("""@pl.program
class ManualDep:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self) -> pl.Tensor[[8, 8], pl.FP32]:
        out = pl.create_tensor([8, 8], dtype=pl.FP32, manual_dep=True)
        return out
""")
    assert isinstance(program, ir.Program)
    with pytest.raises(TypeError, match="manual_dep"):
        materialize_entry_outputs(program, next(iter(program.functions.values())))


def test_manual_dep_created_output_is_rejected_by_the_trace_frontend():
    # jit_trace checks deliverability while specializing.
    with pytest.raises(TypeError, match="manual_dep"):
        pl.jit_trace(_manual_dep_created).specialize()


def test_manual_dep_created_output_is_rejected_at_compile(tmp_path):
    # The parse frontend defers to the ABI adapter inside ir.compile.
    program = pl.jit(_manual_dep_created).specialize()
    with pytest.raises(TypeError, match="manual_dep"):
        ir.compile(program, output_dir=str(tmp_path), skip_ptoas=True)


@pytest.mark.parametrize("decorator", [pl.jit, pl.jit_trace])
def test_manual_dep_scratch_that_is_not_returned_stays_allowed(decorator, tmp_path):
    """Only promoted allocations are rejected; a manual_dep scratch buffer
    keeps its create statement and its opt-out."""
    program = decorator(_manual_dep_scratch).specialize()
    compiled = ir.compile(program, output_dir=str(tmp_path), skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    assert compiled.result_indices == [1]


@pytest.mark.parametrize("decorator", [pl.jit, pl.jit_trace])
def test_annotationless_result_retains_a_tensor_type(decorator, tmp_path):
    program = decorator(_unannotated).specialize()
    entry = next(iter(program.functions.values()))
    assert len(entry.return_types) == 1
    compiled = ir.compile(program, output_dir=str(tmp_path), skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    assert _metadata(compiled).result_indices == (1,)


@pytest.mark.parametrize("decorator", [pl.jit, pl.jit_trace])
def test_declared_return_mismatch_is_rejected(decorator):
    with pytest.raises(TypeError, match="declared"):
        decorator(_wrong_type).specialize()


@pytest.mark.parametrize("decorator", [pl.jit, pl.jit_trace])
def test_created_result_does_not_inherit_an_unrelated_out_type(decorator, tmp_path):
    program = decorator(_different_output).specialize()
    entry = next(iter(program.functions.values()))
    assert [d.value for d in entry.return_types[0].shape] == [16, 8]
    compiled = ir.compile(program, output_dir=str(tmp_path), skip_ptoas=True)
    assert isinstance(compiled, CompiledProgram)
    assert _metadata(compiled).synthesized_outputs[0].shape == (16, 8)


@pytest.mark.parametrize(
    "field,value", [("result_indices", [0, 1]), ("result_indices", [99]), ("synthesized_outputs", [])]
)
def test_corrupt_result_metadata_is_rejected(tmp_path, field, value):
    compiled = ir.compile(pl.jit(_created).specialize(), output_dir=str(tmp_path), skip_ptoas=True)
    path = compiled.output_dir / "compiled_meta.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        CompiledProgram.from_dir(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
