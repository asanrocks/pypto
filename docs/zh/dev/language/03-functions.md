# 函数与程序结构

函数声明形式、参数方向、跨模块复用，以及如何把 IR 打印回 Python 语法。

## 常量与编译复用

`@pl.jit` 对可调用对象做追踪：函数体在追踪时作为普通 Python 执行，因此它从所在
模块或外层函数继承的名字，会在追踪发生时直接从该命名空间读取——使用的是值本身，
而不是它的任何源码形式。字面量（`int`、`float`、`bool`、`str`、`None`）、`pl` 的
dtype 与枚举常量（`pl.INT8`、`pl.Mem.Vec`、`pl.PadValue.zero`、`pl.NZ`），以及由
它们嵌套构成的 list / tuple 都可用，因为它们只是追踪过程观察到的普通 Python 值。
（“常量必须有源码形式”是旧 AST 前端的规则——该前端现已隔离为
`pl.jit_specializer`，它生成 `@pl.program` 源码并在只含 `pl` 与 `pld` 的命名空间中
重新解析。）

读取发生的时机取决于所在的通道（lane）。**eager 通道**——所有参数都带 facade
注解——在装饰时追踪并返回裸的 `ir.Function`，因此每个继承名字在装饰时就被固化进
该函数；之后重新绑定不会改变产物（用[函数](../../user/language/01-functions.md)
一文介绍的单函数 Program 路线编译它）。**lazy 通道**——裸 `pl.Tensor` 参数，或任何
依赖调用时机的东西——把追踪推迟到每次调用或 `compile()`，因此重新绑定会被下一次
追踪观察到：

```python
BLOCK = 32

@pl.jit
def slice_kernel(x: pl.Tensor) -> pl.Tensor:
    with pl.at(level=pl.Level.CORE_GROUP):
        result = pl.slice(x, [BLOCK, 128], [0, 0])
    return result

first = slice_kernel.compile(x)    # traced with BLOCK = 32
BLOCK = 64
second = slice_kernel.compile(x)   # re-traced with 64 rows
```

宿主旋钮（host knob）与缓存键：

- 带普通 Python 注解（或无注解）的参数是**宿主旋钮**：Python 标量实参在追踪时被
  固化为 `Const`，且它的*值*参与 dispatch 缓存键——值不同的两次调用不会共享产物。
- `pl.runtime(value, dtype)` 是 `pl.Scalar[...]` 参数的**运行时标量**写法：缓存键
  只携带 dtype，从不携带值。
- dispatch（`kernel(*args, config=...)`）按桥接签名缓存——每个张量的 shape、
  dtype 与 layout，每个宿主标量的值，每个运行时标量的 dtype——再加上下文所述的
  编译身份（compile identity）。函数体从模块继承的名字*不*被跟踪：缓存条目一旦
  变热，就会继续提供按旧值追踪的产物；请在首次 dispatch 之前重新绑定被引用的
  常量，或用 `compile()` 获得一次新追踪。
- `kernel.compile(*args, config=...)` 从不读写该缓存——每次调用都构建新产物；
  显式“先编译后调用”固定一次具体化（materialization）（03-entrance.md §5.5）。

每次追踪按其运行当时的各层命名空间取值；追踪期间修改被捕获的对象不会被任何键
跟踪。

### 编译选项与诊断请求

dispatch 与 `kernel.compile(*args, config=...)` 以相同方式解析 `config`：
`RunConfig` 的编译侧字段（策略、dump、诊断）转发给 `ir.compile`，而 `device_id`、
`codegen_only` 等运行时控制项作用于 dispatch。省略 `config` 时使用与
`RunConfig()` 相同的默认值：`a2a3sim` 平台、默认优化策略、关闭 pass dump。

两个表面的缓存行为不同。dispatch 缓存按桥接签名*加上编译身份*作为键——平台、
优化策略、分布式配置、`analyze_auto_scopes_for_deps`、生效的内存规划器，以及
L0C double-buffer 开关，每一项都从 `RunConfig` 和当前活跃的 `PassContext` 解析——
因此两种编译配置不会共享同一个产物。`kernel.compile()` 完全没有缓存：每次调用
都构建新产物。

因此诊断请求应经由 `compile()` 发出。`dump_passes=True`（或 `PassDumpLevel.NONE`
之外的 dump 级别）、`dump_ptoas_passes=True` 以及显式 `save_kernels_dir` 在该
路线上天然得到满足，重复请求会重新生成输出。dispatch 缓存刻意不把这些旋钮计入
键——它们不改变产物——因此命中热条目时产物被复用，请求的 dump *不会*输出；需要
输出时请通过 `compile()` 请求。诊断控制不会分裂 dispatch 缓存。

```python
from pypto.runtime import RunConfig

first = slice_kernel.compile(x)
second = slice_kernel.compile(x)  # a fresh build — compile() has no cache
dumped = slice_kernel.compile(
    x, config=RunConfig(dump_passes=True, save_kernels_dir="debug_kernel")
)
```

### 预备二进制而不执行

`kernel.compile(*args, config=...)` 就是“预备而不执行”的表面：它完成追踪、运行
pass 流水线、构建全部 kernel 与 orchestration 二进制，然后返回 `CompiledProgram`
而不做 dispatch。此过程不初始化 NPU、不创建运行时 worker；构建主机仍需安装目标
编译器、SDK，以及运行时的 Python/原生依赖。

```python
import pypto.language as pl
from pypto.runtime import RunConfig

@pl.jit
def add_three(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [16, 16])
        pl.store(pl.add(tile, 3.0), [0, 0], out)
    return out


import torch

x = torch.randn(16, 16, dtype=torch.float32)
out = torch.empty(16, 16, dtype=torch.float32)
config = RunConfig(platform="a2a3")
prepared = add_three.compile(x, out, config=config)  # shapes and dtypes only
# Later, on a host with an available NPU:
# prepared(x, out, config=config)
```

样例张量只按元数据桥接——shape 与 dtype——从不读取内容。标量实参遵循上文的参数
世界：普通注解参数是宿主旋钮，其值在追踪时固化；`pl.Scalar[...]` 参数接受
`pl.runtime(value, dtype)` 以保持为运行时实参。调用返回的对象时按完整参数列表
dispatch。编译错误直接传递给调用方；构建失败后可再次调用 `compile()` 重试。

[持久 JIT 缓存](../10-jit-cache.md) 一文记载的持久产物仓库、`warmup()` 方法，以及
仅用元数据的预热 CLI，都属于已隔离的 `pl.jit_specializer` 运行时
（`python/pypto/jit/`）；追踪 jit 只有上文描述的进程内 dispatch 缓存。

## 函数

```python
# Single return type
def function_name(param1: pl.INT64, param2: pl.FP32) -> pl.INT64:
    x: pl.INT64 = param1 + 1
    return x

# Multiple return types
def function_name(x: pl.INT64) -> tuple[pl.INT64, pl.INT64]:
    y: pl.INT64 = x + 1
    z: pl.INT64 = x * 2
    return y, z

# No return types
def function_name(x: pl.INT64):
    y: pl.INT64 = x + 1

# With function type
@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(n: pl.INT64) -> pl.INT64:
    return n + 1

@pl.function(type=pl.FunctionType.InCore)
def aicore_kernel(x: pl.INT64) -> pl.INT64:
    return x * 2
```

### 函数类型

| 类型 | 用途 | 描述 |
| ---- | ---- | ---- |
| `pl.FunctionType.Opaque` | 默认 | 未指定的函数类型 |
| `pl.FunctionType.Orchestration` | Host/AICPU | 控制流和依赖分析 |
| `pl.FunctionType.InCore` | AICore | AICore 子图执行（未特化） |
| `pl.FunctionType.AIC` | Cube 核心 | Cube 核心内核（特化的 InCore） |
| `pl.FunctionType.AIV` | Vector 核心 | Vector 核心内核（特化的 InCore） |
| `pl.FunctionType.Group` | 多核 | AIC + AIV 内核的协调调度组 |
| `pl.FunctionType.Graph` | 主机/AICPU | 可录制的编排片段，由 `host_build_graph` runtime 回放（详见下文） |

未指定类型时, 函数默认为 `Opaque`。

### Graph 片段

`pl.FunctionType.Graph` 把函数标记为可录制的编排片段。在 `host_build_graph`
runtime 下，每个调用点变成一次 task launch：runtime 在第一次调用时录制、之后回放，
于是 N 次调用只付一次建图代价，而不是 N 次：

```python
@pl.program
class Decoder:
    @pl.function(type=pl.FunctionType.Graph)
    def layer(self, cur, normed, next_hidden, wq, layer_base: pl.Scalar[pl.INDEX]):
        ...

    @pl.function
    def decode(self, cur, normed, next_hidden, wq):
        for i in pl.range(40):
            self.layer(cur, normed, next_hidden, wq, i * 5120)
```

一个 Graph 函数就是一份被录制的拓扑：runtime 用生成的 C++ 函数地址来标识这份录制，
因此不存在需要命名、也不需要保证唯一的 cache key。

### 参数方向

参数可以使用包装类型指定 `In` (默认)、`Out` 或 `InOut` 方向:

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(
    qi: pl.Tensor[[16, 128], pl.BF16],                   # In (default)
    output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],      # InOut
    result: pl.Out[pl.Tensor[[16, 128], pl.FP32]],        # Out
    scale: pl.Scalar[pl.FP32],                             # In (default)
) -> pl.Tensor[[16, 128], pl.FP32]:
    ...
```

| 方向 | 包装类型 | 描述 |
| ---- | -------- | ---- |
| `In` | 无 (默认) | 只读输入参数 |
| `Out` | `pl.Out[type]` | 只写输出参数 |
| `InOut` | `pl.InOut[type]` | 读写输入/输出参数 |

**约束:** `Scalar` 参数不能使用 `InOut` 方向 (会抛出 `ParserTypeError`)。

#### 写入 `Out` / `InOut` 参数

**裸赋值不会写入参数。** 它只是重新绑定 Python 名字: 参数 Var 指向一个新算出来的
张量，调用方的 buffer 完全没被碰过。程序照样能编译、能运行。调用方拿回什么取决于方向:
`Out` buffer 是新分配且从未初始化的，读出来是垃圾值; `InOut` buffer 里仍是调用方传进来的
输入，结果是悄悄地陈旧。

```python
out = pl.add(a, b)          # ❌ 什么也没写
out[:] = pl.add(a, b)       # ✅ 写入整个张量
```

要真正写入参数，请用下标形式:

| 写法 | 写入内容 | 适用场景 |
| ---- | -------- | -------- |
| `out[:] = <expr>` | 整个张量 | 结果就是整个输出 |
| `out[<slices>] = <expr>` | 该子窗口 | 只写输出的一部分 |
| `out = pl.assemble(out, <expr>, <offset>)` | `<offset>` 处的窗口 | 下标语法糖展开后的显式形式 |
| `out = <expr>` | **什么也不写** | 永远不要这么写——见下面的告警 |

只有第一行和第三行等价，且仅当切片覆盖全部范围、`<offset>` 全为 0 时才等价。

##### `OutParamWriteDropped` 告警

编译器会对丢失写入的裸赋值报告:

```text
[warning] [OutParamWriteDropped] (pipeline_input) Assigning to Out parameter 'out'
in function 'main' rebinds the name only — the caller's buffer is never written.
Use 'out[:] = <expr>' to write the whole tensor, or 'out[<slices>] = <expr>' for
a sub-window. at repro.py:12:9
```

该检查基于数据流而非语法。一个值可以经由 loop carry 流回该参数——`pl.store` 通过参数回写，
被携带的绑定保持着这个别名——那是真正的回写，因此不会告警:

```python
for col in pl.range(0, n, chunk):
    data = pl.store(local, [0, col], data)   # 重新绑定即循环的派生 carry
data = pld.tensor.allreduce(data, signal, ...)   # 被携带的 `data` 就是参数; 不告警
```

该检查刻意保守: 仅仅*读取*参数的值 (例如 `out = pl.add(out, b)`) 同样会丢失写入，
但不会被报告。要区分「读取参数」和「通过参数回写」需要算子注册表并未记录的逐算子写语义，
而对正确代码误报的代价高于漏报。需要全量写入时请写成 `out[:] = pl.add(out, b)`。

如果该检查对你的程序没有价值，可以用 `disabled_diagnostics` 关闭:

```python
disabled = passes.DiagnosticCheckSet()
disabled.insert(passes.DiagnosticCheck.OutParamWriteDropped)
ir.compile(program, disabled_diagnostics=disabled)
```

## 分支中定义的 `@pl.program` 类

在追踪前端下，`@pl.program` 在装饰时把类体作为 Python 执行——装饰器拿到的是
`class` 语句刚构建出的类对象，因此既不需要定位，也没有歧义需要消解。一个函数可以
在多个分支里定义同名类（它们的 `__qualname__` 完全相同）；每个定义都是独立的一次
装饰，各自产生自己的 `Program`：

```python
def make(case):
    if case == "add":
        @pl.program
        class Prog:                                # 追踪*这个*类体
            @pl.function
            def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                return pl.add(x, 1.0)
        return Prog

    @pl.program
    class Prog:                                    # 这个则追踪*这个*类体
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, 3.0)
    return Prog
```

闭包变量以同样的方式参数化单一定义——下面例子中的 `scale` 在追踪运行时从外层
作用域读取：

```python
def make(scale):
    @pl.program
    class Prog:                                    # 单一定义，参数化
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, scale)
    return Prog
```

本节过去描述的源码定位机制——按类体中方法的行号找到 `class` 语句，并以
`ParserSyntaxError` 拒绝真正无法区分的定义——只对解析器所有的
`@pl.program_parser` 入口（round-trip 固件）成立，该入口仍从源码收集类体。
追踪前端从不读取源码。

## 跨模块函数复用

在 `@pl.program` 类之外定义的函数可通过两种机制复用。

### 外部 `@pl.function` 调用

在 `@pl.program` 内部可按名称调用外部定义的 `@pl.function`。该函数会自动加入 Program，
并生成 `ir.Call(GlobalVar, args)`。

```python
@pl.function
def softmax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = softmax(x)   # ir.Call(GlobalVar("softmax"), [x])
        return y
```

**规则：**

- 使用函数的 `.name` 作为 GlobalVar（别名透明）
- 外部与内部函数名不得冲突
- 两个不同的外部函数具有相同 `.name` 是错误
- 同一外部函数从多个 method 调用时只加入一次

### `@pl.inline` 装饰器

`@pl.inline` 捕获函数以便在语句级内联。不会向 Program 添加函数——每次调用点展开函数体。

```python
@pl.inline
def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    return result

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = normalize(x)  # statements inlined in-place
        return y
```

**规则：**

- 实参个数必须与形参列表完全一致
- 内联定义处的闭包变量可用
- 内联函数可多次调用（每次展开相互独立）
- 支持嵌套内联调用

## 完整示例

### 张量操作 (派生 carry 的循环)

```python
import pypto.language as pl

@pl.function
def loop_sum(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
    total = pl.const(0, pl.INT64)
    for i in pl.range(n):
        total = total + i
    return total
```

累加器是一个带类型的 staged 常量——裸字面量 `0` 会在每次迭代内联字面量——而重绑定
`total = total + i` 本身就是循环的 carry; 无需任何声明。在 SSA pass 之后打印该函数会得到
以 `init_values=`/`pl.yield_()` 渲染 carry 的 round-trip 文本; 那是打印器的方言，
由 `pl.parse` 解析回来——绝不是 tracing 的输入。

### Tile 操作 (基于 Tile 的计算)

```python
import pypto.language as pl

@pl.program
class BlockExample:
    @pl.function
    def tile_add(
        self,
        input_a: pl.Tensor[[64, 64], pl.FP32],
        input_b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
        tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
        result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
        return result
```

## 打印 IR 节点

对任意 IR 节点调用 `as_python()` 获取其 Python 表示：

```python
print(stmt.as_python())          # "x: pl.Scalar[pl.INT64] = a + b"（默认 "pl" 前缀）
print(stmt.as_python("ir"))      # "x: ir.Scalar[ir.INT64] = a + b"（自定义前缀）
```

### 简洁模式 (Concise Mode)

传入 `concise=True` 可省略中间变量的类型标注。函数签名类型（参数和返回值）始终保留：

```python
print(func.as_python())                  # 详细模式（默认）：每个赋值都包含类型
print(func.as_python(concise=True))      # 简洁模式：省略中间类型标注
```

详细输出：

```python
def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP16]:
    y: pl.Tensor[[64, 128], pl.FP32] = pl.some_op(x)
    result: pl.Tensor[[64, 128], pl.FP16] = pl.cast(y, pl.FP16)
    return result
```

简洁输出：

```python
def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP16]:
    y = pl.some_op(x)
    result = pl.cast(y, pl.FP16)
    return result
```

自由函数 `ir.python_print(node)` 同样可用，支持相同的参数。

## 参考资料

- [Python IR 语法规范](00-python_syntax.md) —— 类型与表达式
- [语句与控制流](01-statements.md) —— 函数体内的语句形式
- [集成手写 C++ Kernel](04-external-kernels.md) —— 调用外部 kernel
