# Language Syntax

What ordinary Python syntax means inside a kernel: subscripts, operators, and
names taken from the enclosing scope.

> **Prerequisites:** [Types](00-types.md).

## Concept

A decorator **executes** your function as Python — collecting IR is a side
effect of execution ([Functions and Programs](01-functions.md)). The familiar
syntax in a kernel body therefore *runs*; what makes it kernel code is that
its operands are staged values, and the staged wrappers turn each operation
into an IR op:

- A subscript is not a Python index — it stages a slice, a read, or an
  assemble.
- An operator is not Python's `+` on numbers — it dispatches to the IR op
  chosen from the operand types.
- A name from the enclosing scope is an ordinary Python name, read when the
  body executes — host values fold in by value at trace time.

The practical consequence: these constructs fail at trace time — decoration
for eager kernels, the first call for lazy ones — as ordinary Python errors
with real line numbers, and a value no operator can stage is rejected loudly,
with the operator named.

## Quickstart: the same slice, two ways

<!-- doctest: setup -->
```python
import pypto.language as pl
```

<!-- doctest: run -->
```python
@pl.function(type=pl.FunctionType.InCore)
def head(x: pl.Tensor[[128, 64], pl.FP32],
         out: pl.Out[pl.Tensor[[16, 64], pl.FP32]]):
    top = x[0:16, :]                        # stages pl.tensor.slice(x, [16, 64], [0, 0])
    out[0:16, 0:64] = pl.mul(top, 2.0)      # stages pl.tensor.assemble(out, ..., [0, 0])
    return out


text = head.as_python()
assert "pl.tensor.slice(x, [16, 64], [0, 0])" in text
assert "pl.tensor.assemble(out, " in text
```

Both lines become IR operations. The first shows what the sugar does; the
second is the explicit spelling the sugar produces everywhere else.

## Mechanics

### Subscript sugar

A subscript on a staged `Tensor` or `Tile` executes as one of three IR
operations:

| Written | Executes as |
| ------- | ----------- |
| `A[0:16, :]` | `pl.slice(A, [16, N], [0, 0])` |
| `A[i, j]` | `pl.tensor.read(A, [i, j])` / `pl.tile.read(A, [i, j])` |
| `A[0:16, 0:32]` | `pl.slice(A, [16, 32], [0, 0])` |
| `dst[:] = src` | `dst = pl.assemble(dst, src, [0, 0, ...])` — one zero per axis |
| `dst[i:i+16, j:j+32] = src` | `dst = pl.assemble(dst, src, [i, j])` |

The write form follows the numpy width rule: a bare `dst[:]` covers every axis (the whole
tensor), and axes past the ones you name are implicit full slices — so the offsets are
always full-rank.

The write form rebinds `dst`, which is incompatible with strict SSA. Under
`@pl.function(strict_ssa=True)` — or any post-SSA context — the rebind is
rejected (`SSAViolationError: variable 'out' is already bound; strict_ssa
forbids rebind`); call `pl.assemble(...)` explicitly instead.

### Python operators

Standard operators map to IR operations on `Tensor`, `Tile`, and `Scalar`
values — dispatch happens by execution: the staged value's protocol methods
stage the op, so which op you get follows the operand's type.

| Python | Operation |
| ------ | --------- |
| `a + b` / `a - b` / `a * b` / `a / b` | `add` / `sub` / `mul` / `div` |
| `a == b` / `a != b` | `eq` / `ne` |
| `a < b` / `a > b` | `lt` / `gt` |

A scalar on either side is detected and dispatched to the scalar-operand form
(`pl.mul(t, 2.0)` → `muls`).

### Closure capture

The body is ordinary Python, so a name from the enclosing Python scope is read
when the body executes — at decoration for eager kernels, at each trace for
lazy ones (a lazy `@pl.jit` re-reads inherited names on every compile; see
[Functions and Programs](01-functions.md)). What happens to the value depends
on what the consuming operator can stage:

- A **host value in operand position** — an int shape element, a Python
  constant — participates by value at that moment: `N = 8` folds into the
  traced IR as `8`.
- A **staged value** is used as-is — the IR expression flows into the op.
- A value **no operator can stage** is a loud trace-time error naming the
  operator: `pl.mul: expected Tensor or Tile operands, got (Tensor, Widget)`.
  Pass it as a parameter instead of capturing it.

There is no parse-time resolution left to get wrong — and no re-parse
constraint either: binding a device-geometry query such as
`pl.system.available_cluster_count()` to a name before use is fine under
tracing; the printed IR binds it to an ordinary local.

## Edge Cases

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`dst[...] = src` rejected** | `strict_ssa=True` forbids the rebind | Call `pl.assemble(dst, src, [...])` |
| **A subscript produced a read where a slice was wanted** | `A[i, j]` reads one element; `A[i:i+1, :]` slices | Use slice syntax when you want a region |
| **A captured value rejected at trace time** | No operator can stage it | Pass it as a parameter instead of capturing it |

## Worked example

`examples/utils/error_handling.py` — what a bare rebinding costs: the body
rebinds `result` instead of writing through it, the earlier value is discarded
(ordinary Python semantics — a rebind drops the old binding), and codegen
surfaces it as a structural error because the `Out` parameter never receives
its write.

## See Also

- [Types](00-types.md) — what the operands of this syntax are, including `pl.Array`.
- [Compile-Time Directives](05-directives.md) — the parser-owned diagnostics and `pl.const`.
- [Operations](../ops/00-dispatch.md) — where the operators behind the sugar live.
- [Python IR Syntax Specification](../../dev/language/00-python_syntax.md) — the parser's full surface.
