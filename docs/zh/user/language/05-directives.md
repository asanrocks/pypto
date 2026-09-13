# 编译期指令

在 tracing 前端下"编译期"意味着什么：`pl.const` 是仍然有效的带类型字面量构造，
而 `pl.static_print` / `pl.static_assert` 是 parser 专属的诊断 —— 在 kernel 被
trace 时它们什么都不做。

> **前置**：[函数与程序](01-functions.md)。

## Concept

tracing 前端把你的函数**当 Python 执行**，收集 IR 是执行的副作用 —— 不存在一个
可以检视源码文本的解析步骤。于是旧的"编译期指令"一分为二：

- `pl.const(value, dtype)` 是 tracing 原生构造：带类型的字面量，也是具体 staged
  值的唯一拼写 —— 裸的 host 字面量不能初始化 staged 状态（见[控制流](02-control-flow.md)）。
- `pl.static_print` 与 `pl.static_assert` 是 **AST parser** 的诊断：它们不产生
  IR，只在 parser 拥有前端的地方运行（round-trip 文本、`@pl.function_parser`
  fixtures）。在 tracing 下它们是**静默 no-op** —— 调用会执行，但没有任何效果：
  不打印任何东西，失败的 `static_assert` 也不会抛错。

知道这一点，它就不再是谜题：一条什么都没打印的 `static_print` 并不能证明"函数
没被解析过" —— 在 tracing 下它从来就不会打印。

## Quickstart：实际会发生什么

<!-- doctest: setup -->
```python
import pypto.language as pl
```

<!-- doctest: run -->
```python
@pl.function
def probe(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
    pl.static_print("x =", x)
    pl.static_assert(False, "expected 128 columns")
    return pl.tensor.muls(x, 2.0)


text = probe.as_python()
assert "static_print" not in text          # nothing printed, and nothing in the IR
assert "static_assert" not in text         # False did not raise either
```

两条语句都消失了。对 host 值的断言，直接的替代写法是一个会抛错的 host `if` ——
条件在 trace 时执行，未走到的分支不进 IR：

<!-- doctest: run -->
```python
COLS = 128


@pl.function
def checked(x: pl.Tensor[[64, COLS], pl.FP32]) -> pl.Tensor[[64, COLS], pl.FP32]:
    if COLS % 16 != 0:
        raise ValueError(f"{COLS} is not 16-aligned")
    return pl.tensor.muls(x, 2.0)


text = checked.as_python()
assert "ValueError" not in text            # the guard is host-side: no IR, no raise
```

## Mechanics

### parser 专属诊断

| 构造 | 在 parser 下 | 在 tracing 下 |
| ---- | ------------ | ------------- |
| `pl.static_print(*args)` | 解析期打印类型/值信息 | 静默 no-op，返回 `None` |
| `pl.static_assert(cond, msg)` | 为假时抛 `ParserError` | 静默 no-op —— 连 `False` 也不抛 |

两者在任何地方都可 import：使用它们的 kernel 在 tracing 下照常收集，只是检查
永远不会触发。仍然需要检查时：

- **会抛错的 host `if`** —— 针对 host 值上的条件，如 quickstart 所示；
- **verifier pass** —— 针对收集出的 IR 的检查（见 passes 文档）。

一个值得点名的陷阱：**永远不要绑定其结果**。`s = pl.static_print(...)` 绑定的是
no-op 的 `None`，而 `None` 是 TaskId 哨兵 —— 绑定器会把它具化成一个持有
`pl.system.task_invalid()` 的 `pl.Scalar[pl.TASK_ID]`，这个 op 会落进你的 IR。
语句位置的调用是安全的。

parser 通道还用 `static_print` 展示它对某个值*推断*出了什么。在 tracing 下没有
什么可推断的 —— 类型早已由产生它的算子定死 —— 而且 printed IR 会重述每一个类型，
所以需要确认某个事实时，`kernel.as_python()` 就是替代品。

### 带类型的常量

`pl.const(value, dtype)` 构造一个显式指定 dtype 的常量，而不是由字面量推断出的
默认类型。它的存在是为了让打印器能往返非默认类型的常量，当字面量的位宽有意义时
就该用它 —— 或者当字面量必须被 *stage* 时：

```python
step = pl.const(1, pl.INT32)
```

它在 trace 内外都可用；在 kernel 里它是 staged 初始化的拼写（完整规则见
[控制流](02-control-flow.md)）。

## 边界情况

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`static_print` 什么都没打印** | tracing 下它从不打印 | 读 printed IR：`kernel.as_python()` 重述每一个类型 |
| **`static_assert` 没抓住坏值** | tracing 下它是 no-op | host `if` + `raise`，或 verifier pass |
| **IR 里出现了 `Scalar[pl.TASK_ID]` / `system.task_invalid`** | 你绑定了 no-op 调用返回的 `None` | 永远不要绑定 `static_print` / `static_assert` 的结果 |
| **trace 时 `TypeError: 'NoneType' object is not subscriptable`** | traced 值上的 `x.shape` / `x.dtype` 是 `None` —— staged 值不向 Python 暴露 shape | 检查 host 常量，或从 printed IR 读 shape |
| **常量的位宽不对** | dtype 是从 Python 字面量推断的 | 用 `pl.const(value, dtype)` 钉住 |

## See Also

- [语言语法](06-syntax.md) —— 下标、运算符、名字在执行期的行为。
- [函数与程序](01-functions.md) —— 各个装饰器分别在什么时候 trace。
- [控制流](02-control-flow.md) —— `pl.const` 作为 staged 初始化的拼写。
- [Python IR 语法规范](../../dev/language/00-python_syntax.md) —— parser 的完整表面，`static_*` 检查仍在那里运行。
