# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Flash Attention with loop-level iter_args.

Demonstrates advanced DSL control flow:
  - Derived loop carries: the body's rebinding of mi/li/attn/oi is the carry
  - Nested staged if/else with plain assignment at the branch joins
  - Online softmax with running max/sum accumulators
  - Multi-name carry through the KV-block loop

Algorithm:
  For each KV block j in [0, 16):
    sij = Q @ Kj^T (scaled)
    pij = softmax(sij) with online max/sum correction
    Oi = update(Oi, pij @ Vj)
  Return Oi / li (final normalization), written into the output tensor

Run:  python examples/models/03_flash_attention.py
Next: examples/models/04_paged_attention.py
"""

import pypto.language as pl


@pl.jit
def flash_attention(q_13: pl.Tensor, k_16: pl.Tensor, v_19: pl.Tensor, o_20: pl.Out[pl.Tensor]):
    # slot_num=1 halves the automatic cross-core pipe's Vec reservation (the
    # default depth-2 ring holds 64 KiB against a 188416 B budget); the packed
    # tiles then fit alongside it.
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(slot_num=1)]):
        # Loop-carried state; the loop body's rebinds derive the carries.
        attn_update = pl.create_tensor([64, 128], dtype=pl.FP32)
        oi_update = pl.create_tensor([64, 128], dtype=pl.FP32)
        li_update = pl.create_tensor([64, 1], dtype=pl.FP32)
        mi_update = pl.create_tensor([64, 1], dtype=pl.FP32)

        # statement.for with derived iter_args
        for i in pl.range(16):
            # Inner statement.block
            kj = pl.slice(k_16, [64, 128], [i * 64, 0])
            vj = pl.slice(v_19, [64, 128], [i * 64, 0])
            sij = pl.matmul(q_13, kj, out_dtype=pl.FP16, a_trans=False, b_trans=True, c_matrix_nz=False)
            sij_1 = pl.mul(sij, 0.0883883)
            row_max = pl.row_max(sij_1)
            sub = pl.sub(sij_1, row_max)
            p_ij = pl.exp(sub)
            l_ij = pl.row_sum(p_ij)
            tildaPij_83 = pl.cast(p_ij, target_type=pl.FP16, mode="round")

            # Nested staged if; both branches assign the same names
            if i == 0:
                # Inner statement.block
                oiUpdate_87 = pl.matmul(tildaPij_83, vj, out_dtype=pl.FP16)
                oiUpdate_90 = pl.assemble(oi_update, oiUpdate_87, offset=[0, 0])

                # Nested if inside first branch: a staged join, so plain assignment
                if i == 15:
                    attn_95 = pl.div(oiUpdate_90, l_ij)
                else:
                    attn_95 = attn_update

                # More statements in first branch
                liUpdate_98 = pl.assemble(li_update, l_ij, offset=[0, 0])
                miUpdate_101 = pl.assemble(mi_update, row_max, offset=[0, 0])

                # Rebinding *is* the carry update
                mi_update = miUpdate_101
                li_update = liUpdate_98
                attn_update = attn_95
                oi_update = oiUpdate_90
            else:
                # Else branch — the online-softmax correction against the carried
                # running max (mi_update), not a fresh tile: m_new = max(m_old, m_j),
                # O and l rescale by exp(m_old - m_new), the new block contributes
                # exp(s - m_new).
                miUpdate_103 = pl.maximum(mi_update, row_max)
                t1_104 = pl.sub(mi_update, miUpdate_103)
                t2_105 = pl.exp(t1_104)
                t3_106 = pl.sub(row_max, miUpdate_103)
                t4_107 = pl.exp(t3_106)
                t5_108 = pl.mul(t4_107, l_ij)
                t6_109 = pl.mul(t2_105, li_update)
                liUpdate_110 = pl.add(t6_109, t5_108)
                liUpdate_113 = pl.assemble(li_update, liUpdate_110, offset=[0, 0])
                q3_114 = pl.mul(oi_update, t2_105)
                q1_115 = pl.matmul(
                    tildaPij_83, vj, out_dtype=pl.FP16, a_trans=False, b_trans=False, c_matrix_nz=False
                )
                q2_116 = pl.mul(q1_115, t4_107)
                oiUpdate_117 = pl.add(q3_114, q2_116)
                oiUpdate_120 = pl.assemble(oi_update, oiUpdate_117, offset=[0, 0])

                # Nested staged if in else branch
                if i == 15:
                    attn_125 = pl.div(oiUpdate_120, liUpdate_113)
                else:
                    attn_125 = attn_update

                mi_update = miUpdate_103
                li_update = liUpdate_113
                attn_update = attn_125
                oi_update = oiUpdate_120
        o_20[:] = attn_update
    return o_20


if __name__ == "__main__":
    import torch
    from pypto.runtime import RunConfig

    torch.manual_seed(0)
    q = torch.randn(64, 128, dtype=torch.float16)
    k = torch.randn(1024, 128, dtype=torch.float16)
    v = torch.randn(1024, 128, dtype=torch.float16)
    o = torch.zeros(64, 128, dtype=torch.float32)

    flash_attention(q, k, v, o, config=RunConfig(platform="a2a3sim"))

    # Golden reference: full softmax(Q K^T / sqrt(d)) V in fp32.
    scores = torch.matmul(q.float(), k.float().transpose(0, 1)) / (128.0**0.5)
    expected = torch.matmul(torch.softmax(scores, dim=-1), v.float())
    assert torch.allclose(o, expected, rtol=2e-2, atol=2e-2), (
        f"validation failed: max diff = {(o - expected).abs().max().item()}"
    )
    print("OK")
