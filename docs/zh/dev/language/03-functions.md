# 函数与程序结构

> **旧 Python 接口示例：** 本页为兼容现有用法保留 `platform=` 调用，
> 这些调用会发出 `DeprecationWarning`；新代码应使用 `backend_type` 和
> `execution_mode`。参见[目标映射](../08-entry-points.md#迁移旧-platform-示例)。

函数声明形式、参数方向、跨模块复用，以及如何把 IR 打印回 Python 语法。

## JIT 实现结构

JIT 实现位于 `python/pypto/jit/`。公开的 `jit` 对象和 `JITFunction` 保留在
`decorator.py`；私有分析模块提供元数据，不持有编译对象缓存，也不负责调度 kernel。

| 模块 | 职责 |
| ---- | ---- |
| `decorator.py` | 装饰器、`JITFunction`、依赖发现、缓存所有权，以及编译/执行入口 |
| `_arguments.py` | 可选 torch 加载、张量实参元数据、注解和 signature/constexpr 校验 |
| `_ast_analysis.py` | 共享只读 AST 缓存、源码名称和动态维声明 |
| `_call_binding.py` | 调用点参数映射、constexpr 变体和跨依赖的动态维传播 |
| `_tensor_meta.py` | 按源码顺序推导局部张量元数据、处理分支/循环绑定及递归依赖返回值 |
| `specializer.py` | 专门化上下文、注解处理、程序源码组装与诊断源码位置映射 |
| `_specialize_body.py` | 有状态 AST 函数体改写、变量重命名与词法作用域处理 |
| `_specialize_values.py` | 专门化与缓存键共用的常量及 dtype/layout 源码渲染 |
| `_specialize_types.py` | 共享的 `DynDim`、`ShapeDim`、`TensorMeta` 定义；原有 `specializer` 导入路径保持有效 |
| `_source.py`、`cache.py`、`_persistent.py` | 请求命名空间快照、缓存键和持久产物解析 |

`JITFunction` 持有依赖图状态、编译对象缓存及其锁。缓存键构造与专门化继续共享
同一次请求的命名空间快照（namespace snapshot）。元数据推导通过显式回调（callback）
获取依赖，并将回调传入递归返回值分析，因此不需要在运行时导入装饰器模块。
共享 helper 通过其所属模块查找，使 AST 缓存和测试替换目标保持唯一。

## JIT 常量与编译复用

`@pl.jit` 的函数体会被转写成 `@pl.program` 源码，并在只含 `pl` 与 `pld` 的命名空间中
重新解析；因此函数体从所在模块或外层函数继承的名字，必须在其使用处被替换成能求值回
同一个值的源码文本。可替换的范围包括：字面量（`int`、`float`、`bool`、`str`、`None`）、
`pl` 的 dtype 与枚举常量（`pl.INT8`、`pl.Mem.Vec`、`pl.PadValue.zero`、`pl.NZ`）、
分布式枚举常量（`pld.ReduceOp.Max`、`pld.NotifyOp.AtomicAdd`、`pld.WaitCmp.Ge`），
以及由它们嵌套构成的 list / tuple——例如工厂函数选定的 shape、舍入模式或集合通信归约操作。
没有源码形式的值保持原样，于是该名字得以保留，由解析器报错。

所有可替换的名字同时进入编译键（compilation key），包括传递 JIT helper 和源码注解中使用
的常量。编译键直接对每个名字*实际生成的文本*取哈希，且与 specializer 使用同一个函数，
因此两者不会脱节：改变生成源码的常量必然改变键，不可替换的值则不贡献任何内容。
重新绑定被引用的常量会产生新的专门化（specialization）；修改无关全局变量，
或修改被正文局部变量遮蔽的同名全局变量，不会使正文依赖键失效。

```python
BLOCK = 32

@pl.jit
def slice_kernel(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[BLOCK, 128], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        result = pl.slice(x, [BLOCK, 128], [0, 0])
    return result

first = slice_kernel.compile()
BLOCK = 64
second = slice_kernel.compile()  # A distinct specialization with 64 rows.
```

名称解析、键构造和专门化共用一次调用的命名空间快照（namespace snapshot）。
捕获后重新绑定的常量影响下一次调用，不影响正在编译的产物。
即使常量同名，各 helper 也保留各自的命名空间。重新绑定被引用的 JIT helper
还会刷新依赖图。各并发调用在键构造和专门化期间始终使用自己捕获的依赖图。
哈希还包含各 helper 的函数类型、层级和 `auto_scope` 设置，因此即使函数源码相同，
重新绑定到具有不同编译属性的 helper 后也不会复用旧产物。

每次请求只复制一次各模块的全局命名空间，各 helper 通过闭包覆盖层共享这份快照。
未绑定的闭包 cell 仍会遮蔽同名全局变量。验证后的依赖图保留 Python 源码哈希；
注解绑定不变时，复用已解析的布局。外部源码文件仍在每次请求时检查。
被引用的闭包常量由源码依赖哈希覆盖，无需单独的闭包键组件。

快照仅复制绑定：此常量跟踪机制不支持编译期间修改任意配置对象
内部状态，或修改编译器/源码文件。持久复用还需要完整的
[产物身份与缓存策略](../10-jit-cache.md)。

### 动态维度元数据

解析特化程序前，JIT 会推导局部 `TensorMeta`，为仅有裸张量注解的 helper 补齐实参类型。
标量绑定使用 `int` 常量、已有张量轴的 `DynDim` 引用、不可变表达式元组或表示未知的
`None`。`width = n * 8` 保存为快照 `(ast.Mult, (n_dim, 8))`；赋值别名保留同一元组，
后续算术则创建不同快照。重新绑定操作数的源变量名不会改变已有快照。

reshape 推导在记录 `TensorMeta.shape` 时，才为标量表达式建立结果专属的 `DynDim`
符号；标量快照不进入张量类型。同一结果 shape 内重复使用同一快照时共享
符号；不同结果建立独立绑定。被调函数从实际张量参数的描述符（descriptor）读取维度。
原始算术及 reshape 校验仍保留在程序中，占位上界不会替代真实运行时尺寸。

`synthesized` 标记自动生成的占位符号，继续沿用既有注解覆盖及 helper 返回行为。

当运行时上下界之差可证明为常量时，下标视图也保留静态宽度。例如，
`src[base:base + 128, :]` 携带 `[128, src.shape[1]]` 元数据，因此在第二维一致时，
可以与相同类型的 `[128, N]` 分配结果合并。JIT 将整数加减和常数乘法转换为临时表达式，
复用 Parser 的算术分析器（arithmetic analyzer）证明宽度。每个二元操作必须有一个常量
操作数，以触发 INDEX 类型提升。此推导无法获知标量 dtype，因此保守拒绝两个运行时
操作数之间的运算及运行时取负。例如，`base + other:base + (other + 128)` 不能证明
宽度为 128：下界的加法可能在 INT32 中溢出，而上界表达式会提升为 INDEX。
宽度证明解析字面量、未被遮蔽的全局/闭包常量、constexpr 参数、当前张量元数据中的
直接静态 shape 读取，以及函数体转换器确实替换为字面量的 shape 别名。
直接 shape 读取也支持局部分配的张量，其维度由 Parser 折叠。
未解析切片保留的父张量范围会标记为非精确值，不能通过后续 shape 读取、张量别名、
分支合并或辅助函数返回值证明切片宽度。
辅助函数声明的动态轴会将回退范围替换为绑定到实际描述符的维度。
`width = 128` 或 `width: pl.Scalar[pl.INT32] = 128` 等局部标量赋值即使在维度
推导中具有已知值，也仍按符号处理：它们在特化后保留为变量，而非 INDEX 字面量。
运行时维度的样例上界不会作为证明。符号仅在同一对切片上下界内共享，
因此其他语句中的重绑定和别名不会建立新的相等关系。未解析的调用、读取、动态除法
和非线性乘积不会被当作可抵消的符号。无法证明的宽度保留已有的父维度回退行为，
带 step 的切片仍不受此元数据推导支持。分支合并仍要求 shape、dtype、layout
及动态轴身份兼容。此推导不改变切片操作及其存储生命周期。

### JIT 内联 helper 的 Tile 参数

内联 helper 可以接收显式声明类型的片上 Tile：

```python
@pl.jit.inline(auto_scope=False)
def keep_tile(tile: pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec]) -> pl.Tile:
    return tile
```

专门化（specialization）保留声明的形状、数据类型和内存空间，并折叠注解中引用的
模块与闭包常量。如果延迟求值的注解引用 helper 自身的 `pl.constexpr` 参数，
会先代入该调用的 constexpr 绑定，再求值注解；该绑定优先于模块或闭包中的同名常量。
不同绑定会生成各自的 helper 专门化。`p.Tile` 或直接导入的 `Tile` 等 DSL 别名
会转换为规范的 `pl` 路径，并保留 `TileView`、`MemRef` 元数据；用户恰好定义了
无关的同名变量 `pl`，也不会干扰生成结果。若工厂函数的局部常量仅用于立即求值的
Tile 注解、没有进入函数闭包，专门化可以保留原函数已经求值的注解。
即使部分形状轴依赖 helper 形参，其他独立的形状轴与 dtype 仍会保留，依赖形参的
部分则逐次代入调用参数；合并后的类型声明同样参与缓存身份计算。内存空间 constexpr
同样逐次代入，并保留独立的工厂形状。元数据替换需保持原有类别，无法确定类别时
会给出明确错误。若同一个表达式
同时引用形参与无法恢复的工厂局部值，请把该局部值作为另一个 constexpr 参数传入，
或在 helper 函数体中引用它以保留闭包。
仅被 Tile 参数类型引用的动态符号也会生成模块级
`pl.dynamic()` 声明，不再依赖 Tensor 参数同时引用这些符号。
生成的符号名与 Tile 或 Tensor 形参同名时也会保留声明；真实的 Scalar 维度形参仍使用局部绑定。
自由动态符号与不同的 Scalar 形参共用名字时，专门化会明确报告歧义；请为自由符号使用唯一的 `pl.dynamic()` 名字。

`InlineFunctions` 在代入实参之前检查 Tile 的数据类型、物理维数与形状，以及双方
已明确指定的内存空间是否冲突。泛型物理维度绑定到实参的物理尺寸，有效尺寸不参与
这一检查；支持 `2*M` 绑定到 `16`、得到 `M=8` 等整数仿射绑定。
推导前先代入已知 Scalar 实参：`n=8`、`n+M=16` 可推导出 `M=8`。
不同 Tile 实参可以独立绑定各自的泛型物理维度，helper 返回 Tile 形参时也遵循这一规则。
`tmp = second; return tmp` 等简单不可变别名也会沿用对应实参的返回类型。
与 Tensor 类型、显式 Tile 视图或 Scalar 形参共用的维度仍需满足共享约束，没有返回值的 helper 也会检查。
Scalar 实参值也约束只出现在视图中的维度：声明有效尺寸为 `n` 时，`n=8` 不能匹配实际有效尺寸 `3`。
`n - 1` 等依赖 Scalar 的复合视图表达式也会在代入实参后比较，覆盖有效尺寸、stride 和 offset。
这也包含通过 `rows = 2*n` 等共享维度间接依赖 Scalar 的表达式。
实参省略有效形状时使用物理形状，省略 offset 时使用零；依赖 Scalar 的 stride 约定需要实参具有显式 stride 元数据。
JIT 专门化也会保留嵌套 `pl.Tuple` 参数注解，包括其中完整的 Tile 类型、动态符号和逐次调用的 constexpr 绑定。
tuple 参数内的 Tile 也会递归检查相同的物理类型和视图约束。
为 Tile 注解生成的动态维度声明名必须使用 Python 规范化后可赋值的合法标识符，不能是 Python 关键字，也不能覆盖 DSL 别名 `pl` 或 `pld`。
内联（inlining）直接代入调用方的 Tile，无需复制，并保留其有效形状
（valid shape）。专门化也会保留非内联函数中可求值的 Tile 注解。
这会保留类型声明，但不会扩展设备派发接口；普通 InCore 入口仍不支持 Tile 参数。

### 编译选项与诊断请求

JIT 调用和 `kernel.compile()` 在查找进程内产物之前解析编译选项。
省略 `config` 时使用与 `RunConfig()` 相同的默认值：`a2a3sim`、默认优化策略，
以及关闭 pass dump。影响产物的选项进入缓存键，包括实际内存规划器、运行时 ABI，
以及由 `PYPTO_EMIT_PTO_LOC` 控制的源码位置输出。
`device_id`、`codegen_only` 和单独的 `save_kernels` 等运行控制不进入键。

以下请求每次都重新编译，不读取或添加产物缓存条目：

- `dump_passes=True` 或除 `PassDumpLevel.NONE` 外的 dump 级别，
  以及 `dump_ptoas_passes=True`。
- `compile_profiling=True`、活跃的 `CompileProfiler`，或通过
  `PYPTO_COMPILE_PROFILING` 启用的编译性能分析。
- 显式指定 `save_kernels_dir`。
- 显式启用 `PYPTO_EMIT_DEBUG_RUNNER` 或 `PYPTO_REBUILD_FROM_PTO`。
  `0`、`false`、`no` 关闭此请求（忽略大小写和两端空白）；已设置为空字符串时，
  保留启用/默认行为。
- 显式诊断设置，或包含 instrument、验证/诊断设置不同于流水线默认值的活跃
  `PassContext`。

因此已有缓存的 kernel 仍会输出请求的 dump 和报告；重复请求会重新生成输出。
诊断编译失败不会改变普通缓存条目。JIT 为每次编译自动分配唯一输出目录，
避免新的诊断请求覆盖缓存产物；显式指定的输出目录按请求使用。
诊断控制不进入编译键。
显式设置与活跃 `PassContext` 冲突时，缓存命中与首次编译都会报告相同错误。
仅选择规划器或运行时的普通上下文仍能复用匹配的缓存产物。

`PYPTO_PROG_BUILD_DIR` 只指定新建私有构建和运行目录的父目录，不请求重新编译。
已有缓存对象保留原目录，共享产物仍位于缓存中。如果每次调用都需要生成新的、
可供检查的输出目录，应使用 `save_kernels_dir`。

```python
from pypto.runtime import RunConfig

cached = slice_kernel.compile()
slice_kernel.compile(config=RunConfig(dump_passes=True, save_kernels_dir="debug_kernel"))
assert slice_kernel.compile() is cached
```

### 预备二进制而不执行

`kernel.warmup()` 与 `kernel.compile()` 使用相同的特化、配置和进程内对象缓存，
并在返回前完成所有 kernel 与 orchestration 二进制的准备。它不会初始化 NPU，
也不会创建运行时 worker。构建主机仍需安装目标编译器、SDK，以及运行时的 Python
和原生依赖。

```python
import pypto.language as pl
from pypto.runtime import RunConfig

@pl.jit
def add_three(
    x: pl.Tensor[[16, 16], pl.FP32],
    out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [16, 16])
        pl.store(pl.add(tile, 3.0), [0, 0], out)
    return out

config = RunConfig(platform="a2a3")
prepared = add_three.warmup(config=config)  # No sample tensor allocation.
# Later, on a host with an available NPU:
# prepared(x, out, config=config)
```

与 `compile()` 一样，可以提供样本张量；预热只读取元数据，不读取张量内容。
张量注解完整时可以完全省略张量实参：标量参数是运行期值，这里无需提供其值，
动态维度沿用现有编译规则。`pl.constexpr` 参数是例外：它的值决定选中哪份产物，
因此必须通过关键字或签名默认值给出。
调用返回的编译对象时，需提供包含标量实参在内的完整参数列表——每个标量的值在此处读取；
constexpr 参数不在该列表中，产物已经把它解析掉了。
`codegen_only` 等仅影响执行的设置不会禁止二进制准备。

返回值是 `compile()` 选中的同一个编译对象，新编译对象仍保留 IR。
预热覆盖 `DistributedCompiledProgram` 的所有芯片级子构建，以及多 orchestration
`CompiledProgram` 的每个 orchestration 子构建。它不会调用分布式对象的
`prepare()`，后者用于创建执行所需的活动 worker。编译错误直接传递给调用方；
二进制构建失败后，可以再次调用 warmup 重试。诊断和显式输出请求仍按上文规则重新编译。

启用[持久缓存](../10-jit-cache.md) 后，warmup 通过[运行时协议](../09-artifact-store.md)
自动发布或复用 READY 产物。持久缓存默认关闭；只读未命中、不支持的输入和存储故障
可以生成私有结果。从缓存恢复的结果 `.program is None`；需要 IR 时关闭持久缓存，
或使用 `specialize()`/`lower()`。公共缓存策略、统计和仅用元数据预热的 CLI 参见
[JIT 持久缓存](../10-jit-cache.md)。

## 追踪前端：`@pl.jit_trace`

同一个 JIT kernel 的 IR 可以由两套前端生成，装饰器按定义逐个选择。
`@pl.jit` 解析函数体（即上文所述的路径）；`@pl.jit_trace` 执行函数体的插桩副本，
并记录实际发生的操作。导入 `pypto.language` 不会改变任何事情：每个定义仍然
使用自己装饰器指定的那条路径。

```python
@pl.jit_trace
def scale_relu(
    x: pl.Tensor,
    scale: float,
    out: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [64, 64])
        pl.store(pl.relu(pl.mul(tile, scale)), [0, 0], out)
    return out
```

`@pl.jit_trace` 与 `@pl.jit` 共享入口（`specialize`、`lower`、`compile`，
以及传入 torch tensor 的宿主调用）。家族拼写中，已验证的是：`incore` 与
`inline` 被调函数，以及 `host` 与 `graph` 入口——它们都是解析路径同名拼写的别名，
保留同一个函数种类、level、role 与 kwarg 表面（见下文
[已支持的表面](#已支持的表面)）。`opaque` 与 `extern` 在装饰时抛出 `TypeError`：
`@pl.jit.extern` 是没有 Python 函数体可供追踪的声明，`opaque` 则允许追踪器词汇表
以外的函数体。helper 必须与调用它的入口声明在同一条路径上：
追踪函数体调用 AST 路径的 `@pl.jit` kernel 会直接报错，而不是静默切换前端；
解析器路径的循环拼写（`pl.while_`、`pl.cond`）在追踪中同样报错——追踪方言
使用普通 Python `while`。

trace 在哪里结束、共享的编译器从哪里开始 —— 四条通道、程序组装，以及迁移等价性证据度量的是什么 —— 见
[Tracing 前端边界](08-tracing-frontends.md)。

### 收集时机与复用

装饰时不追踪任何内容。第一次结构化使用——`.fn`、`.specialize()`、`.lower()`、
`.compile()`，或传入 torch tensor 的宿主调用——会执行一次函数体并保留记录，
之后兼容的使用直接复用该记录。若注解没有确定某个 shape（裸 `pl.Tensor`），
仅凭声明无法追踪，该 kernel 每次调用都会重新追踪。在函数体中读取该 shape
从第一次调用起即可用：`x.shape` 的 lowering 与解析前端一致——静态维度折叠为
常量，动态维度是一条运行时 `pl.tensor.dim` 读取，在编排函数体中直接命名参数
声明的 extent 符号。裸注解因此不妨碍依赖 shape 的函数体，只是去掉了声明时
的那次追踪。

分发缓存以产物契约为键：tensor 的 shape、dtype 与 layout，host knob 的取值，
以及编译身份（platform、strategy、distributed config、memory planner、dbC
标志、runtime ABI、Buffer IR、源码位置）。tensor 的*内容*和
`pl.Scalar[...]` 参数的*取值*不进入键——标量在生成程序中仍是运行时参数，
因此一份产物服务所有取值。

`config=RunConfig(...)` 调用通过所有 jit 路径共用的 `RunConfig.compile_kwargs()`
映射转发编译侧字段。凡是要*运行*编译器的请求——pass 转储、PTOAS 转储、编译
profiling、显式输出目录、或携带 instrument 的外层 `PassContext`——绝不从缓存
应答：每次都重新编译，随后的普通兼容调用仍复用原有的热产物。

### 参数世界

注解决定实参是运行时数据还是编译期常量：

| 注解 | 世界 | 行为 |
| --- | --- | --- |
| `pl.Tensor`、`pl.Tile`、`pl.Out[...]`、`pl.Scalar[dtype]` | 运行时参数 | 进入生成签名；取值在分发时到达 |
| 普通 Python 注解（`float`、`int`）或无注解 | host knob | 在追踪时烘焙进函数体；改变取值会专门化新产物 |

传给运行时声明位置的 host 标量会提升为 `Scalar`。把 host 标量
传给 tensor 位置，或把已 staged 的值传给 host knob，都会抛出指名参数的
`TypeError`：参数世界由声明决定，而不由调用点决定。

facade 声明除了参数世界，还拥有参数的*类型*。该路径从调用点推导被调函数的
参数（正是这种推导让调用得到特化），但推导被声明所界定：位于声明槽位整数格内的
整数实参会被*转换*到声明的 dtype——`pl.Scalar[pl.INT32]` 参数接受 INDEX 值并按
INT32 使用，两个方向都如此，且转换会绑定到一个临时量
（`_rank_int32 = pl.cast(r, pl.INT32)`），使发出的调用携带声明的类型。两套前端
按同一条准入规则（`pypto.language.scalar_conversion`）发出同一条语句，因此被调
函数对所有整数调用方只有一个签名，打印出的 IR 就是手写拼写。格之外的一切保留其
拒绝两个固定宽度 dtype 之间、浮点、`pl.BOOL`、`pl.TASK_ID`、打包 FP4，以及
张量/tile 的 dtype、shape、布局不匹配；`pl.Tensor[[8, 8], pl.FP32]` 参数仍
拒绝 `[16, 8]` 或 BF16 实参。拒绝时抛出的 `TypeError` 同时列出两种类型，修复
方式在调用点（`pl.cast`）或声明处。规则本身与它为何收窄见用户文档的
“声明的标量 dtype 会做转换”（`docs/zh/user/language/00-types.md`）。裸 Python
整数不做转换而是做*检查*：它不携带 dtype，因此超出范围的值在每次派发时都会被
拒绝（标量值从不参与编译产物缓存键），而带类型的值则按上述规则转换。声明的动态维度（`pl.Tensor[[N, 8], ...]`）不属于矛盾：
它声明的是“调用点传什么就是什么”，任何具体维度都会特化它。一个符号只表示一个
维度：若 `N` 出现在多个参数中，为它提供的每个具体维度都必须一致。给两个 `N`
形状的参数分别传入 `[16, 8]` 与 `[32, 8]` 会被拒绝；两者可能不同时应声明两个
符号。Torch 没有 INDEX dtype，其 `int64` 存储是 host 载体，因此 torch `int64`
实参进入声明的 INDEX 槽位时会桥接为 INDEX；显式的 `pl.INT64` facade 进入该槽位
同样属于准入转换，会被记录为保值的类型变更而不是被省略。裸标记同理被拒绝：
`vlen: pl.INDEX` 只声明 dtype 而不声明参数世界，本会变成 host knob，因此错误
信息直接给出 `pl.Scalar[pl.INDEX]` 拼写。声明的**布局**以同样的方式桥接：torch
tensor 不携带布局，因此 `pl.Tensor[[N, K], pl.FP16, pl.NZ]` 的 `NZ` 打包取自声明，
而形状与 dtype 仍以实参为准——与声明维度矛盾的实参依然会被拒绝。

**关键字用于指名已声明的参数。** 调用可以按名字绑定任意参数——
`helper(x, out=out, factor=2.0)`——追踪器在任一路径读取之前就把该绑定折叠进
声明顺序，因此专门化路径的注解、被调路径的变体键、派发桥接以及打印出的程序
看到的都是一次位置调用。于是拼写永远不进入产物：上述两种写法专门化出同一个
程序，这正是共享代码可以保留更清晰写法的原因。`device=` 与 `attrs=` 是调用点
元数据而非参数（detached 调用会派发到该设备），`config=RunConfig(...)` 在顶层
调用中选择编译开关。每种畸形拼写都按名字拒绝，而不是被静默丢弃：未知关键字、
同一参数既按位置又按名字绑定、某个参数未被绑定、以及多余的位置实参。

host knob 在 staged 控制流内部是只读的。函数体只追踪一次，所以循环推进的
host 值——`for _ in pl.range(n)` 中的 `factor = factor * 2.0`——会把那一代表
迭代的结果烘焙进每一次迭代，以及循环之后对它的任何读取。绑定检查会抛出
`TypeError`，并指向 staged 拼写：用 `pl.const(...)` 绑定状态，让循环携带它。
读取 host knob（或在函数体内绑定新的 host 临时变量）仍然合法——没有任何东西
被携带。

### 已支持的表面

`examples/` 下每个被装饰的定义都在该路径上：显式输出（`pl.Out` / `pl.InOut`）
入口及其 `incore`、`inline` 被调函数；针对运行时
标量的 staged `for` / `if` / `while`；host knob；运行时标量参数；循环携带 tile 的
tile assemble；`pl.parallel` 配合原子累加；scope 优化（`pl.split`、`deps=`）；
valid-shape 循环；`host` / `graph` 入口及其逐 rank 分发与录制重放；以及上述分发缓存契约。

该路径强制三种拼写，它们都是相对解析器路径的有意差异：

| 解析器路径 | 追踪路径 | 原因 |
| --- | --- | --- |
| `vlen: pl.INDEX`、`scale: pl.FP32` | `pl.Scalar[pl.INDEX]`、`pl.Scalar[pl.FP32]` | 裸 dtype 标记只声明 dtype 而不声明参数世界，因此在装饰时被拒绝，错误信息给出该拼写 |
| `for i, (acc,) in pl.range(n, init_values=(acc0,)):` + `pl.yield_` | 循环前绑定 `acc = acc0`，循环体末尾重新绑定 | 携带通道由重新绑定推导；`init_values=` 是同一组通道的打印器/解析器往返拼写，解析器仍会打印它 |
| `inline` helper 参数上的 `pl.Out[pl.Tensor]` | 裸 `pl.Tensor` | helper 函数体在调用处被拼接，该参数本就是调用方变量的原地别名 |

第四种是 host 字面量进入 staged 控制流：在 `pl.range` 下重新绑定 `factor = 1.0`
会报错，携带标量需要类型化拼写 `pl.const(1.0, pl.FP32)`。

#### 函数体拼接：`pl.jit_trace.body`

`@pl.jit_trace.body` 声明一个可复用的**语句片段**，并像函数一样调用它。片段在调用处展开，
因此内核读起来是 `scale_block(a, out, row0, k)`，而不是某个放置方法：**不会产生函数、
`GlobalVar` 或 `ir.Call`** —— 程序只获得该函数体的语句。host 调用会被拒绝；函数体只在
活跃的 trace 内展开。

```python
@pl.jit_trace.body
def scale_block(a, out, row0, k):     # 没有 IR 注解：没有被调函数
    tile = pl.load(a, [row0, 0], [16, 64])
    pl.store(pl.mul(tile, k), [row0, 0], out)

@pl.jit_trace
def entry(a: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.FP32],
          out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(4):
            row0 = chunk * 16
            scale_block(a, out, row0, k)
    return out
```

放置策略随声明一起给出：`@pl.jit_trace.body(auto_scope=False)` 让手动编写的 scope 区域
合法，而调用方自己的策略仍管辖其他一切；在调用处再传 `auto_scope=` 会被拒绝 —— 一个片段
只有一种含义。这是内核作者的拼写；下面的框架作者拼写保留显式的放置。

#### 函数体拼接：`pl.authoring.body(...).splice(...)`

`pl.authoring.body` 一次性预备一个 Python 函数（源码重写与编译）并返回可复用的函数体；`.splice(...)`
在追踪内核内的每次出现处将其放置到调用者插入点：**不会产生函数、`GlobalVar` 或
`ir.Call`** —— 程序只获得该函数体的语句。host 调用会被拒绝；该放置仅在活跃的 trace
内执行。

这是框架作者的拼写：它让放置保持可见（`.splice` / `.inline` / `.incore`），适用于需要多种
放置的同一个函数体。`@pl.jit_trace.body` 是它之上的一种策略 —— 一个以 `ctx.splice_body`
为函数体的 `@pl.authoring.recipe` —— 因此本仓库之外的框架也能用公开的 authoring 表层构造
出同一个面向内核的拼写。

因为没有被调函数，也就没有 ABI，所以**注解不起作用**：`def twice(x)` 无需
`pl.Tensor[[...]]` 声明，而同一个函数体可以服务多种参数类型（一次出现是 `Tensor`，
下一次是 `Tile`）。这正是它与 `@pl.jit_trace.inline` 的差别：后者收集一次函数体、
发出调用，因此必须声明参数与结果类型。

```python
@pl.jit_trace.body
def twice(x):                 # 没有 IR 注解：没有被调函数
    return pl.add(x, x)

@pl.jit_trace
def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [64, 64])
        pl.store(twice(tile), [0, 0], out)
        out[:, :] = twice(a)            # 同一个函数体，第二种参数类型
    return out
```

`return` 是拼接局部的：值直接交给调用方的表达式位置，因此函数体可以交付一个由它创建、
没有任何参数背书的值。但内部仍然是 staged 的：拼接会被粘进调用方的控制流，所以跨过
函数体**内部** `pl.range` / `pl.if` 的值，仍遵守与外层函数相同的 staged 规则（以及相同
的 `pl.const` 拼写）。

本波次每个 trace 只允许一次拼接：拼接另一个函数体的函数体（嵌套或 host 递归）会被明确
拒绝，而不是错误拼接。`@pl.jit_trace.body` 用自己的词汇报告这一拒绝（一个函数体不能在
另一个函数体内被调用），它要复用的语句应写进调用它的函数体，或声明为
`@pl.jit_trace.inline` 被调函数。函数体自有的 scope 策略随函数体一起传递：
`body.splice(..., auto_scope=False)` 让手动编写的 scope 区域合法，而调用方自己的策略
仍管辖其他一切 —— 这就是 `@pl.jit_trace.inline(auto_scope=False)` 的拼接写法。

拼接的参数是调用方值的**原地别名**，与 inline 拼接遵循同一条规则。重新绑定参数的函数体
——包括在 `pl.at` scope 内的循环中携带的重新绑定，也就是每个 Qwen3 helper 的形状 —— 直接
写穿调用方的变量，因此调用方能看到更新，外层循环也能推导出携带通道。把名字交还给它自己
（`out = splice_into(x, out)`，写穿惯用法）不产生任何语句。

返回值可通过参数缓冲区或入口新建的静态形状分配交付。入口声明的 `pl.Out` 缓冲区及能追溯到它的
值保持原有写回行为。检查沿着值进行（`return_lineage`），而不是看变量名，因此“返回自己写过的缓冲区”的被调
结果也算已交付：

```python
@pl.jit_trace.inline
def project(x: pl.Tensor[[64, 64], pl.FP32], y: pl.Tensor[[64, 64], pl.FP32]):
    y[:, :] = pl.add(x, 1.0)
    return y

@pl.jit_trace
def tuple_of_results(
    x: pl.Tensor[[64, 64], pl.FP32],
    a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    qa = project(x, a)
    qb = project(x, b)
    return qa, qb                # 两个已交付结果，按扁平形式展开
```

多返回值可以写成一个 staged 的 `pl.Tuple`——产出 `TupleType` 的那个算子。在返回位置上，Python tuple
字面量并不是 host 容器：它按解析器路径的扁平合约展开（`return a, b` 即 N 个结果），而 list 字面量与
`pl.Tuple(a, b)` 是那个结构化的单个值。两者推导出同一个扁平签名
（`-> tuple[pl.Tensor[...], pl.Tensor[...]]`），因此两条路径一致，唯一差异是物化出的
`ReturnStmt`；把返回值写成 `pl.Tuple(...)` 从来不是必需的。PROGRAM 模式会将入口返回的
顶层静态形状新分配转换为隐藏的 `Out` 参数。host 分配并返回缓冲区，公开签名不增加参数；
两条前端路径共享此降级。混合结果保持源码返回顺序，重复结果保持别名关系。动态新建输出、
没有交付映射的返回视图及无法提升生命周期的分配会被拒绝。KERNEL 模式仍要求结果由外部参数承载。

返回元组的被调函数可以直接返回：`return callee(x, a, b)` 与先绑定结果再返回
`pl.Tuple(first, second)` 具有相同的交付来源（delivery lineage）。分类沿着元组值的元素进行，
经过多层转发调用时也保持一致。

声明的 `->` 会与函数体实际产生的值核对，而不是直接盖在签名上。若声明为 `[16, 8]`
而函数体返回 `[8, 8]`，或在双返回值声明下只返回一个值，追踪会立即抛出 `TypeError`，
指明出错位置及两种类型。不同的 staged 返回路径也必须一致：两个分支返回不同类型时
会被拒绝，而不是留下空签名，因为空签名会让后续每个调用者都看不到这一分歧。
没有返回值的函数体保留其声明类型——jit 入口用 `pl.Out` 声明自己写入的输出缓冲区。

`@pl.function` / `@pl.program` 声明仍使用 AST 路径。外部 kernel 没有可追踪的 Python 函数体，
因此始终是声明：在追踪函数体内调用 `@pl.jit.extern` 桩函数会并入入口的 assembly，
而 `pl.jit_trace` 本身没有 `extern` 表单。超出已验证表面的其余家族拼写（`opaque`）
在装饰时抛出 `TypeError`，而不是追踪一条未验证的路径。

这份清单就是边界：需要其中未列出的能力的 kernel 仍留在解析路径上，
而追踪函数体调用解析路径上的 kernel 会直接报错，不会悄悄离开该前端。

### 解析函数体调用追踪 inline 辅助函数

`@pl.jit` 函数体可以调用 `@pl.jit_trace.inline` 辅助函数，因此共享的计算叶子
可以先于其调用者迁移：

```python
@pl.jit_trace.inline(auto_scope=False)
def affine(x: pl.Tensor, gain: pl.constexpr):
    return pl.mul(x, gain)

@pl.jit
def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = affine(a, gain=2.0)
    return out
```

辅助函数在调用点被特化，并以 inline 函数的形式加入调用者的 program，
随后由 `InlineFunctions` 内联展开——不会把追踪 ABI 函数带入 program，
也不会与追踪路径共享产物。由此有三条规则：

- **仅限 inline，且逐层传递。** 辅助函数自身函数体调用到的函数也必须是 inline。
  若它调用了追踪的 `incore` / `opaque` / `graph` 被调函数，会指出该被调函数并报错：
  在解析 program 中组合追踪 ABI 函数不受支持，反向调用同样不受支持。
- **声明即为契约。** 调用点提供的 staged 参数类型用于特化辅助函数，并会与其注解
  校验（`pl.Tensor[[64, 64], pl.BF16]` 会拒绝 FP32 实参）。解析依赖会按调用点
  重新声明，而追踪辅助函数不会。
- **编译期取值按调用点折叠。** `pl.constexpr` 或无注解参数的取值来自调用点自身的
  源码——字面量、调用者的某个 `pl.constexpr` 参数，或模块级/闭包常量——折叠后的
  文本计入调用者的缓存标识，与解析依赖的常量一致。签名默认值不会被读取，
  每次调用都必须显式传入。

staged 实参由其解析后的 IR 类型决定——即该值在调用者中此处的类型，包括布局
（layout）、视图（view）、有效区域（valid region）、存储空间（memory
space）与分布式类型（distributed kind）——因此计算表达式
（`pl.reshape(tile, [32, 128])`、切片视图）与普通名称一样可解析，辅助函数
收集到的正是调用实际传递的内容。该类型在调用生成之前会与辅助函数的声明校验。
调用的实参在生成任何内容前完成校验，因此多余、未知、重复或缺失的实参都会
报错，而不会在生成的调用中静默消失。辅助函数的函数体在构建调用者 program
时收集，因此已编译的调用者热调用不会重新观测任意 Python 状态——这类状态
变化只有在调用者被重新构建后才成为新的特化。

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

该检查基于数据流而非语法。一个值可以不提参数名就流回该参数——例如经由 loop carry——
那是真正的回写，因此不会告警:

```python
for col, (d,) in pl.range(0, n, chunk, init_values=(data,)):
    d = pl.store(local, [0, col], d)
    staged = pl.yield_(d)
data = pld.tensor.allreduce(staged, signal, ...)   # `staged` 就是 `data`; 不告警
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

## `@pl.program` 如何定位类定义

`@pl.program` 是*从源码解析*类体的，因此它必须先找到生成被装饰对象的那条 `class`
语句。仅凭类名无法确定这一点: 同一个函数可以在多个分支里定义同名类，它们的
`__qualname__` 完全相同。

装饰器通过类体中各方法的行号来消歧，因此每个分支都按自己的源码解析:

```python
def make(case):
    if case == "add":
        @pl.program
        class Prog:                                # 解析*这个*类体
            @pl.function
            def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                return pl.add(x, 1.0)
        return Prog

    @pl.program
    class Prog:                                    # 这个则解析*这个*类体
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, 3.0)
    return Prog
```

当多个定义确实无法区分时，装饰器会抛出 `ParserSyntaxError` 并列出全部候选行号，
而不是任选其一——选错就会编译出你从未写过的类体。此时请给各个类取不同的名字，
或者只定义一次、用闭包变量参数化:

```python
def make(scale):
    @pl.program
    class Prog:                                    # 单一定义，参数化
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, scale)
    return Prog
```

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

### 张量操作 (带 iter_args 的循环)

```python
# pypto.program: my_program
import pypto.language as pl

def loop_sum(n: pl.INT64) -> pl.INT64:
    sum_init: pl.INT64 = 0
    for i, (sum,) in pl.range(n, init_values=(sum_init,)):
        sum = pl.yield_(sum + i)
    return sum
```

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

通用 Function attrs 打印为函数体前导区的 `pl.func_attr({...})` 指令。
解析后它们仍属于 Function 元数据，HOST SubWorker 也遵循这一规则：
捕获的 Python `InlineStmt.body` 排除这些指令，同时保留实现和 docstring。
引用参数的 attrs 绑定到函数自身的参数。前导区必须位于可执行语句之前
（允许开头的 docstring）；指令位置错误或键重复都会报错，SubWorker 亦然。

### 打印器实现

私有的 `IRPythonPrinter` 声明位于 `src/ir/transforms/python_printer.h`。
各实现文件共享同一个访问器（visitor）实例，因此递归打印表达式、类型和语句时，
变量重命名、签名上下文（signature context）、缩进及作用域状态保持一致。

| `src/ir/transforms/` 中的源文件 | 职责 |
| ------------------------------- | ---- |
| `python_printer.cpp` | 公开入口与格式化回调、名称消歧、函数/程序打印及依赖排序 |
| `python_printer_expr.cpp` | 表达式优先级、常量、调用/提交及属性值 |
| `python_printer_stmt.cpp` | 控制流、作用域、注释、SSA yield 及窗口缓冲区声明 |
| `python_printer_type.cpp` | 类型标注、形状、内存引用（memory reference）及张量/Tile 视图 |

公开声明仍位于 `include/pypto/ir/transforms/printer.h`。
子表达式打印器继承外层的变量名称映射，保留形状表达式、内存偏移和槽位索引中的绑定。

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
