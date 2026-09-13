# 语言语法

普通 Python 语法写在 kernel 里是什么意思：下标、运算符，以及来自外层作用域的名字。

> **前置**：[类型](00-types.md)。

## Concept

装饰器把你的函数**当 Python 执行** —— 收集 IR 是执行的副作用（见[函数与
程序](01-functions.md)）。kernel 体里那些眼熟的语法因此是*被执行*的；让它们成为
kernel 代码的是：操作数是 staged 值，而 staged 包装器把每个操作变成一个 IR op：

- 下标不是一次 Python 索引 —— 它 stage 成 slice、read 或 assemble。
- 运算符不是 Python 对数字的 `+` —— 它根据操作数类型派发到某个 IR 算子。
- 来自外层作用域的名字是普通的 Python 名字，在 body 执行时被读取 —— host 值
  在 trace 时按值折入。

实际后果：这些构造在 trace 时失败 —— eager kernel 在装饰时，lazy kernel 在首次
调用时 —— 以带真实行号的普通 Python 错误；而任何算子都无法 stage 的值会被大声
拒绝，错误信息里点名算子。

## Quickstart：同一个切片的两种写法

<!-- doctest: setup -->
```python
import pypto.language as pl
```

<!-- doctest: run -->
```python
@pl.jit.incore
def head(x: pl.Tensor[[128, 64], pl.FP32],
         out: pl.Out[pl.Tensor[[16, 64], pl.FP32]]):
    top = x[0:16, :]                        # stages pl.tensor.slice(x, [16, 64], [0, 0])
    out[0:16, 0:64] = pl.mul(top, 2.0)      # stages pl.tensor.assemble(out, ..., [0, 0])
    return out


text = head.as_python()
assert "pl.tensor.slice(x, [16, 64], [0, 0])" in text
assert "pl.tensor.assemble(out, " in text
```

两行都成为 IR 操作。第一行展示语法糖做了什么，第二行就是语法糖在别处产生的
显式拼写。

## Mechanics

### 下标语法糖

staged `Tensor` 或 `Tile` 上的下标执行为三种 IR 操作之一：

| 写法 | 执行为 |
| ---- | ------ |
| `A[0:16, :]` | `pl.slice(A, [16, N], [0, 0])` |
| `A[i, j]` | `pl.tensor.read(A, [i, j])` / `pl.tile.read(A, [i, j])` |
| `A[0:16, 0:32]` | `pl.slice(A, [16, 32], [0, 0])` |
| `dst[:] = src` | `dst = pl.assemble(dst, src, [0, 0, ...])` —— 每轴一个 0 |
| `dst[i:i+16, j:j+32] = src` | `dst = pl.assemble(dst, src, [i, j])` |

写形式遵循 numpy 的宽度规则：裸的 `dst[:]` 盖住每一个轴（整张量），没点到的轴是
隐式的全量切片 —— 所以 offsets 永远是满秩的。

写形式会重绑 `dst`，这与严格 SSA 不兼容。在 `@pl.function(strict_ssa=True)` 下
—— 或任何 SSA 之后的上下文 —— 重绑会被拒绝（`SSAViolationError: variable
'out' is already bound; strict_ssa forbids rebind`）；请显式调用
`pl.assemble(...)`。

### Python 运算符

标准运算符在 `Tensor`、`Tile`、`Scalar` 值上映射到 IR 操作 —— 派发靠执行发生：
staged 值的协议方法 stage 这个 op，所以得到哪个 op 由操作数的类型决定。

| Python | 操作 |
| ------ | ---- |
| `a + b` / `a - b` / `a * b` / `a / b` | `add` / `sub` / `mul` / `div` |
| `a == b` / `a != b` | `eq` / `ne` |
| `a < b` / `a > b` | `lt` / `gt` |

任一侧是标量都会被识别并派发到标量操作数形式（`pl.mul(t, 2.0)` → `muls`）。

### 闭包捕获

body 是普通 Python，所以来自外层 Python 作用域的名字在 body 执行时被读取 ——
eager kernel 在装饰时，lazy kernel 在每次 trace 时（lazy `@pl.jit` 每次编译都
重新读取继承的名字；见[函数与程序](01-functions.md)）。值会发生什么，取决于消费
它的算子能 stage 什么：

- **操作数位置上的 host 值** —— int shape 元素、Python 常量 —— 按当时的值参与：
  `N = 8` 折入被 trace 的 IR 就是 `8`。
- **staged 值**原样使用 —— IR 表达式直接流入这个 op。
- **任何算子都无法 stage** 的值是一个点名算子的 trace 期错误：`pl.mul: expected
  Tensor or Tile operands, got (Tensor, Widget)`。把它作为参数传入，而不是捕获。

不再存在会弄错的"解析期解析" —— 也不再受重解析约束：把 `pl.system.
available_cluster_count()` 这样的设备规模查询先绑定到名字再使用，在 tracing 下
没有问题；printed IR 会把它绑定为一个普通局部变量。

## 边界情况

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`dst[...] = src` 被拒绝** | `strict_ssa=True` 禁止该重绑 | 改用 `pl.assemble(dst, src, [...])` |
| **本想切片，下标却给了一次读** | `A[i, j]` 读单个元素，`A[i:i+1, :]` 才是切片 | 想要一块区域时用切片语法 |
| **某个捕获值在 trace 期被拒绝** | 没有算子能 stage 它 | 把它作为参数传入，而不是捕获 |

## 配套示例

`examples/utils/error_handling.py` —— 裸重绑定的代价：函数体重绑了 `result`
而不是写穿它，先前的值被丢弃（普通 Python 语义 —— 重绑丢弃旧绑定），codegen 把
它报成结构性错误，因为 `Out` 参数始终没有收到写入。

## See Also

- [类型](00-types.md) —— 这些语法的操作数是什么，包括 `pl.Array`。
- [编译期指令](05-directives.md) —— parser 专属诊断与 `pl.const`。
- [算子](../ops/00-dispatch.md) —— 语法糖背后的算子在哪里。
- [Python IR 语法规范](../../dev/language/00-python_syntax.md) —— 解析器的完整表面。
