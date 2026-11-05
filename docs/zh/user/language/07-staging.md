# 值与暂存

哪些值在 kernel 被构建时就已确定，哪些值在运行时才确定 —— 本页其余所有 tracing 规则都由这一区分推出。

> **前置**：[类型](00-types.md)。本页解释注解在运行时一侧的含义；
> [函数与程序 § tracing 前端](01-functions.md) 解释哪个装饰器选择哪个前端。

## 概念

一个 `@pl.jit_trace` kernel 是被**执行**的。收集 IR 只是把函数体运行一次的副作用，因此函数体里的每个值都有来源，而来源决定了这个值是否进入编译后的程序：

| 世界 | 值从哪来 | 程序里得到什么 |
| ---- | -------- | -------------- |
| **host（宿主）** | Python：字面量、字面量上的算术、配置对象、普通函数的返回值 | 什么也没有 —— *值*被内联到使用处，值本身消失 |
| **staged（暂存）** | 签名（`pl.Scalar[dtype]`、`pl.Tensor`、tile）、`pl.const(...)`、staged 循环变量，或某个 `pl.*` 运算的结果 | 一个操作数或参数；运行期由设备提供或计算 |

由此得到的规则是：**host 值对编译器可见，staged 值对硬件可见。** 分块方式、tile 形状以及程序里到底有哪些语句必须是 host；任何设备在两次调用之间可能需要变化的东西必须是 staged。

host 数据并不是二等公民。因为函数体会运行，普通 Python —— `if`、`for`、dataclass、列表推导、辅助函数 —— 都可以用来*组织* staged 计算，并且在程序里不留痕迹。本页讲这两个世界的含义以及它们在何处被守护；[元编程一个 kernel](../tutorials/06-metaprogramming.md) 把这件事一步一步搭起来。

## 快速上手：一个因子，两个世界

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto import ir
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
ROWS = COLS = 64


def program_text(kernel, *args) -> str:
    """The lowered program a call would ship, without compiling for a device."""
    return ir.python_print(kernel.compile_for_test(*args))
```

<!-- doctest: run -->
```python
@pl.jit_trace
def baked(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: float,                                    # host: a compile-time knob
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [ROWS, COLS])
        pl.store(pl.mul(tile, k), [0, 0], out)
    return out


@pl.jit_trace
def staged(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    s: pl.Scalar[pl.FP32],                       # staged: a runtime parameter
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [ROWS, COLS])
        pl.store(pl.mul(tile, s), [0, 0], out)
    return out


a = torch.randn(ROWS, COLS, dtype=torch.float32)
empty = torch.zeros(ROWS, COLS, dtype=torch.float32)

# Both compute the same thing for one value...
out = torch.zeros_like(a)
staged(a, 3.0, out, config=CFG)
assert torch.allclose(out, a * 3.0)

# ...but only the host knob changes the program, and only the staged factor
# survives into the signature.
assert program_text(baked, a, 2.0, empty) != program_text(baked, a, 7.0, empty)
assert program_text(staged, a, 3.0, empty) == program_text(staged, a, 9.0, empty)
assert "pl.Scalar[pl.FP32]" in program_text(staged, a, 3.0, empty)
assert "pl.Scalar[pl.FP32]" not in program_text(baked, a, 2.0, empty)
```

不同的 `k` 是**另一个程序**：字面量被固化，所以派发缓存以该值为键。不同的 `s` 是**同一个程序**换一个实参。

## 机制

### 判断一个值属于哪个世界

| 你写的 | 世界 | 原因 |
| ------ | ---- | ---- |
| 签名里的 `k: float`、`n: int`、`flag: bool` | host | 普通 Python 注解就是编译期旋钮 |
| 签名里的 `s: pl.Scalar[pl.FP32]` | staged | 该注解声明了一个运行期参数 |
| `tile = pl.load(a, ...)`、`pl.add(tile, 1.0)` | staged | 运算的结果是操作数 |
| `pl.range(...)` 内的 `chunk * 16` | staged | `chunk` 是 staged 循环变量，所以乘积也是操作数 |
| 函数体内的 `CHUNKS = ROWS // TILE` | host | host 值上的算术现在就求值 |
| `factor = pl.const(1.0, pl.FP32)` | staged | 显式的 staged 常量（具有运行期身份的局部值） |
| `cfg = TilePlan(...)`；`cfg.chunks` | host | 对象属于 Python；在某个字段携带 staged 值之前，它的字段都是 host |

把 host 值传到签名声明为 staged 的槽位会**提升**：调用 `staged(a, 3.0, out)` 时 `3.0` 变成 `pl.const` 操作数，因为世界由声明决定。反向则被拒绝，并抛出指明该参数的 `TypeError`：staged 值无法被固化进程序。

对象只是容器。字段都是 host 数据的 dataclass 是配置；携带 staged tile 的 dataclass 是 trace 可以传递的记录；
两者都不是 kernel 参数，而 bundle 在调用点展开进参数表。复用代码的四种方式见
[函数与程序](01-functions.md)，三种对象形态见 `examples/tracing/03_host_objects.py`。

### 控制流跟随其主体

同一区分决定了 `if` 或循环**是**什么。决定它的不是语法，而是被判断或被迭代对象所属的域：

| 主体 | `if` | `for` | `while` |
| ---- | ---- | ----- | ------- |
| **host** | 只运行一个分支；程序里没有分支 | `range` / list / tuple：函数体按元素逐个运行，所以程序里每个迭代一份拷贝 | 现在就运行（由 host 条件结束） |
| **staged** | 程序里的一个分支；两个分支各被 trace 一次，汇合处把各自留下的值串起来 | `pl.range` / `pl.parallel` / `pl.pipeline`：程序里一个循环，函数体只 trace 一次 | staged 条件上的普通 `while`：一个 `WhileStmt`，携带值在函数体内重新绑定 |

因为 staged 循环的函数体只运行**一次**，其中的 host 代码不会在设备上重复：trace 保留一份函数体，由循环来重复它。`pl.unroll` 是印证该规则的反例 —— 在 trace 时它是 staged 循环，经过 pass 之后是完全展开的函数体，这也是它的边界必须是字面量的原因。

parser 前端的循环拼写在这里不适用：`pl.while_` 与 `pl.cond` 是 parse 期语法，在 trace 内会抛出 `TypeError`。请改写为 `while i < n:`；其谓词与重新绑定都是 staged 的，因此循环会被记录。

### 携带状态

跨迭代变化的值必须有存放之处。两个世界的答案不同，而 trace 会强制这一差异，而不是让某一侧悄悄被固化：

```python
factor = pl.const(1.0, pl.FP32)             # staged state: a real carried value
for _i in pl.range(n):
    factor = factor * 2.0                   # rebind: the trace derives the carry
```

**staged** 局部值按上述方式携带 —— 循环前绑定，函数体内重新绑定；打印出的程序会显示推导出的 `init_values` / `pl.yield_` 对。**host** 值可以在 staged 控制流内被读取，但不能被其推进：`factor = 1.0` 之后在 `pl.range` 下执行 `factor = factor * 2.0` 会抛出 `TypeError`（“loop-carried host state”），因为函数体只 trace 一次，那一次迭代的值会被固化进每一次迭代以及循环之后的每一次读取。以新名字出现的逐迭代 host 临时量仍然合法。

staged 分支的汇合同理：只在单侧重新绑定 staged 名字是合法的 —— 汇合会把另一条路径的值串过来 —— 但只在单侧绑定随后又在汇合后读取的 **host** 名字会抛出 `TypeError`（“bound on only one branch”），因为该读取会悄悄内联那一侧的字面量。请在两个分支都绑定，或把它变成 staged。

## 边界情况

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| 某个旋钮的取值改变了程序，或出现了第二个 artifact | 该值是 host 并被固化进 trace | 声明为 `pl.Scalar[dtype]`，使它成为运行期参数 |
| **`'Scalar' object cannot be interpreted as an integer`** | Python `range()` 需要 host 边界 | 迭代 `pl.range(n)`；host `range` 无法表达运行期行程数 |
| **`loop-carried host state`** | 在 staged 控制流内更新了 host 值 | 用 `pl.const(...)` 播种后重新绑定，或每次迭代用新名字 |
| **`bound on only one branch`** | host 名字只在 staged `if` 的一个分支绑定，却在汇合后被读取 | 两个分支都绑定，或把它变成 staged |
| **`pl.while_ is parser-frontend syntax …`** | traced kernel 里用了 parse 期循环拼写 | 改写为 staged 条件上的普通 `while` |
| 传给 host 旋钮的 staged 值被拒绝 | 签名声明的是普通 Python 类型 | 把参数注解为 `pl.Scalar[dtype]` |
| staged 循环里的 host `print` 只触发一次 | 函数体只 trace 一次，只有 staged 结构被重复 | 在 host 循环里打印，或直接读打印出的程序 |
| 程序里没有源码写出的分支 | 谓词是 host 的 | 若必须由设备选择，就 stage 该谓词（读取 `pl.Scalar`） |

## 可运行示例

| 示例 | 展示 |
| ---- | ---- |
| `examples/tracing/01_two_domains.py` | host 因子与 staged 因子，以及各自构建出的程序 |
| `examples/tracing/02_control_flow.py` | `range` 展开与 `pl.range` 记录的对比，以及记录式为何与规模无关 |
| `examples/tracing/advanced/07_diagnostics.py` | host 携带状态守护与修复，以及由 body 解决的语句边界陷阱 |

每个示例都在模拟器上运行并断言它宣称的结构性事实，因此本页与示例不会在无人察觉时彼此偏离。

## 参见

- [类型](00-types.md) —— 声明参数所属世界的注解。
- [函数与程序](01-functions.md) —— `@pl.jit` 与 `@pl.jit_trace` 的区别，以及各前端要求的拼写。
- [控制流](02-control-flow.md) —— staged 循环形式及它们所依据的 SSA 规则。
- [作用域与放置](04-scopes.md) —— staged 语句在哪里运行。
