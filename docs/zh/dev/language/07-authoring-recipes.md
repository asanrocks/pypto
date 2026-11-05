# 授权 Recipe

`@pl.authoring.recipe` —— 通用授权层：作用于授权上下文的程序。

构造 recipe（[06-construct-recipes.md](06-construct-recipes.md)）是一种特化协议：单个 `yield` 即是 body 边界的生成器。其下的授权运行时更通用 —— 一个 **recipe** 是一个普通函数，接收一个授权上下文并组合 *追踪效果*（tracing effects）：压入函数、向其中收集 body、终结它、发出调用、向当前 trace 拼接 body、打开 region。该层的职责是让这些效果可组合，同时不暴露 tracer（无 builder、无 `PtoFrame`、无解释器）：

```python
import pypto.language as pl


@pl.authoring.body
def squared(x):
    return pl.mul(x, x)


@pl.authoring.recipe
def scaled_square(ctx, body, x, k):
    fn = ctx.push_function(
        pl.authoring.FunctionContract(name="scaled_square", function_type=pl.FunctionType.Inline)
    )
    collected = fn.collect_body(body, args=(x,))
    artifact = fn.finalize(collected)
    return ctx.emit_call(artifact, x) if k == 1 else artifact
```

## 分层

```text
kernel author          recipes, pl.authoring.body compositions, pl.* users
    │ composes
authoring tier          @pl.authoring.recipe / @pl.authoring.construct + the context views
    │ drives
tracer                  rewriter, interpreter, Sema/Frame
    │
IR builder              closed statement/scope vocabulary
```

`@pl.authoring.construct` 是一种*受约束的 recipe 协议* —— 一个 region、一个 body 边界、由 trace 协议驱动。`@pl.authoring.recipe` 是不受约束的形式：任意效果序列、任意数量的 body、不要求 `yield`。两者接收的是同一授权会话的视图；recipe 上下文可以做构造上下文能做的一切（region 组合），外加函数与 body 效果。

词汇表在必须封闭的地方保持封闭：recipe 组合既有的 IR 构造和既有的函数类别。需要新降低形状的 recipe 首先是一个框架变更。

## 导入根

`pypto.language.authoring` 是授权层的规范命名空间 —— 框架作者的导入根。包按职责拆分：

```text
authoring/
├── contracts.py    FunctionContract、ParamSpec、产物类型
├── session.py      授权会话与函数单元生命周期
├── construct.py    @pl.authoring.construct 与 construct 协议
├── recipe.py       @pl.authoring.recipe、@pl.authoring.body、组合入口
└── dispatch.py     分发效果（submit / deps 降低）
```

该包是授权层的**唯一导入根**：任何授权表面都不会重导出到顶层 `pl` 命名空间。顶层留给 kernel 作者；框架作者（以及标准库 —— 它是参考客户端，而非拥有私有访问的兄弟）经由该根导入。

## 语言表面分类

授权层的边界表现为对整个公开语言表面的分类，使用一条规则：

> **按特征降低（lowering）到什么分类，而不是按它返回什么。** 如果其含义是*结构* —— region、循环、函数、调用、submit、绑定、终结符 —— 它属于授权家族。如果它降低为*staged 值之上的数据流操作* —— 表达式位置消费的单个操作节点 —— 它属于值层。其余的是前端或 host 运行时 API。

授权层拥有的能力家族：

```text
region       scope、carrier、with/for recipe 生命周期
body         collect、splice、inline、InCore 放置
function     函数单元、contract、终结、调用
dispatch     submit、SPMD submit、依赖、task 捕获
control      return、yield、break、continue
metadata     函数属性、cache policy、dump 状态
binding      task ID、循环变量、storage、被 carry 的结果
```

公开表面的分类：

| 表面 | 降低为 | 分类 |
| ---- | ------ | ---- |
| `pl.at`、`pl.scope`、`pl.manual_scope`、`pl.graph`、`pl.cluster` | scope | 授权 —— region 效果（recipe） |
| `pl.spmd`（两种拼写）、`pl.split_aiv` | 分发 scope + carrier / SplitAiv region | 授权 —— region 效果（recipe） |
| `pl.authoring.body(...)` 的 `.splice(...)` | 拼接 | 授权 —— body 效果（公开的 splice 拼写） |
| `pl.authoring.body` 的 `.inline` / `.incore` | 函数单元 + 调用 | 授权 —— callee 组合 |
| `pl.submit`、`pl.spmd_submit` | `ir.Submit` —— 结构语句 | 授权 —— 分发效果（标准库拼写是同一效果之上的策略包装；`ctx.submit` 是 recipe 拼写） |
| `pld.notify` / `wait` / `defer_wait` | 同步语句 | 值层 —— 分布式操作（任何 recipe 主体都可以作为普通操作调用它们；不存在专用的授权能力，增加它是一个需要真实 recipe 用例的框架变更） |
| `pl.func_attr` | 函数属性（序言指令） | metadata 效果（`FunctionContract` 携带 attrs；marker 是源码拼写） |
| `pl.set_cache_policy` / `dump_tag` / `no_dep` | 随值携带的 Var metadata | metadata 标记（该层消费其事实） |
| `pl.bind_storage` | storage 声明 + 基址绑定 | 绑定效果 |
| `pl.range` / `parallel` / `unroll` / `pipeline` | `ForStmt` 原语 | 框架拥有的条目（host 执行等价）；该层的 `ctx.open_loop` 组合 `ForStmt` region，配方授权真实循环而标准库条目保持 host 行为 |
| 显式 carry / 结果绑定 | Frame 拥有的 SSA 中介 | 框架原语 —— pass 侧推理通道 |
| `if` / Python `while` | 解释器原生控制流 | 框架原语 —— 控制能力路由进入它们但不能创建它们 |
| scope 种类、语句种类、函数类别 | 封闭的 IR 词汇表 | 框架拥有 |
| `tensor` / `tile` / `unified` / `array` / `system` / `prefetch` 操作（含分配与块身份查询） | 表达式位置的操作节点 | **值层 —— 永不迁移**（facade 代数与操作发射拥有它们） |
| `pld.remote_load` / `put` / `get` 及分布式值操作 | 操作节点 | 值层 |
| `pl.jit` / `pl.jit_trace` / `pl.function` / `program` / `loads` | —— | 前端（特化、缓存、ABI、分发；解析通道是独立后端） |
| `while_`、`cond`、`static_print`、`static_assert` | —— | 解析通道辅助 / 编译期诊断 |
| `split(...)`、`cross_core_slot(...)` | —— | 纯配置描述符 —— recipe 实参 |
| `pld.CommCtx` 会话、`prefetch` 异步上下文、`AsyncEvent` / `AsyncSession` | 无 IR | host 运行时 API —— 在 IR 授权层之外 |

规则固定的两个结论：

1. **值层需要自身的内部清理，而非迁移。** 其问题（统一分发、操作发射辅助）属于 facade 代数；将它们与授权层混淆会模糊本设计唯一的干净边界。
2. **标准库是参考客户端。** 标准库特征应当可以通过公开授权能力实现，tracer 仅提供适配器。一个需要 `PtoFrame`、解释器栈或裸 builder 的 recipe，要么表明存在真正的新框架原语，要么表明缺少一个语义能力 —— 绝不表明标准库需要另一个私有约定。

## 会话及其单元

一次 recipe 调用拥有一个**授权会话**。会话要么是借用的（包装活动 trace 的 frame），要么是根（它自己打开入口函数）。会话拥有的一切都是栈形的：

```text
session
    ├── the caller frame (borrowed) or the root function unit
    └── pushed function units, LIFO
```

一个**函数单元**是一个压入的函数：一次性 Sema、其 frame，以及一个生命周期：

```text
created → collecting → finalized
```

## 效果词汇表

| 效果 | 归属 | 用途 |
| ---- | ---- | ---- |
| `ctx.push_function(contract)` | 会话 | 从一个 `FunctionContract`（名称、函数类别、level、role、attrs、可选 params/returns）打开子函数单元。 |
| `fn.collect_body(body, args=...)` | 函数单元 | 在单元内执行一个已准备的 body：全新词法 frame，参数来自 contract 或由 staged 实参推导。每单元恰好一次收集。 |
| `fn.finalize(collected)` | 函数单元 | 契约步骤：检查返回路径、固定声明的返回、铸造 `FunctionArtifact`。恰好一次；在会话单元间 LIFO。 |
| `ctx.emit_call(artifact, *args)` | 调用者 frame | 在调用者的 trace 中为已终结的 artifact 发出调用点（含向活动程序装配的注册）。 |
| `ctx.materialize_entry(artifact)` | 根会话 | 将根单元的 artifact 标记为入口 —— jit 封装所包装的语义步骤。 |
| `ctx.splice_body(body, *args)` | 调用者 frame | 借用式收集：在调用者当前插入点执行已准备的 body，不创建函数或调用。 |
| `ctx.region(...)` / `ctx.open_scope(...)` / ... | 会话 | 构造层的 region 组合，保持不变（见 [06-construct-recipes.md](06-construct-recipes.md)）。 |
| `ctx.submit(callee, *args, ...)` | 调用者 frame | 分发效果 —— `pl.submit` / `pl.spmd_submit` 共享的降低（`core_num=` 是 SPMD 启动规格）。返回结构 Tuple facade；用 `ctx.bind` 绑定其投影。 |
| `ctx.bind(name, value)` | 调用者 frame | 绑定效果 —— recipe 代码是普通 Python，其解包从不经过重写器路由。落入 Frame 自身的绑定路径：携带 `__trace_bind__` 的投影一次性命名其共享节点，名字映射到一个可变的 `ir.Var`，返回的 facade 包装该 Var。 |
| `ctx.control.emit_return(...)` 等 | 会话 | 语义控制能力（见下）。 |

## 不变量

1. **每个效果都有显式归属。** body 被收集进一个单元或借用的调用者 frame —— 绝不进“当前恰好活动的”目标。
2. **单元按 LIFO 终结**，且 recipe 在仍拥有未终结单元或未关闭 region 时不能完成。
3. **每单元一次收集。** 第二个 body 需要第二个单元（底层 Sema 设计为一次性）。
4. **收集不发出调用；终结也不发出。** artifact 在 `emit_call` / `materialize_entry` 之前是惰性的。（收集*确实*会关闭单元的 builder 函数 —— runner 的正常完成；终结拥有的是*契约*：返回路径、声明的返回、artifact 铸造。）
5. **已终结的 artifact 不可变**，可安全向上传递。
6. **活单元不能逃离其 recipe 调用** —— 会话关闭后使用的单元句柄是一个显式错误。
7. **body 收集使用全新词法 frame**；定义点全局与闭包保持存活；staged 类型约束 body 内部。
8. **失败是原子的。** recipe 内的异常丢弃外层 trace（tracer 的一次性原则）；没有部分状态存活。
9. **上下文暴露能力，绝不暴露 tracer 内部。** builder、`PtoFrame`、解释器都不会到达 recipe 代码。
10. **拒绝拼接嵌套。** 每个 trace 一次拼接。

## 已准备的 body（`pl.authoring.body`）

`@pl.authoring.body` 一次性准备一个 Python 函数（源码重写与编译）并返回一个 `TraceableBody`。准备只缓存代码形状的事实。已准备的 body 由显式操作放置：

| 放置 | 含义 |
| ---- | ---- |
| `body.splice(*args)` | 在调用者插入点、每次出现时执行 —— 借用收集。 |
| `body.inline(*args)` | 压入 `Inline` 函数单元、收集、终结、发出调用。 |
| `body.incore(*args, level=...)` | `InCore` 函数单元的相同组合。 |

当装配活动时，组合按程序构建去重：同一 body 与结构相同的实参类型是一个函数，而不是每个调用点一个（jit 封装的逐调用变体通道仍是特化、缓存的前端；授权组合是显式、非特化的拼写）。

推导签名（无 `params=` 压入的单元）来自 staged 实参的类型 —— 每个实参必须是 staged 且静态形状的；host 值或动态_extent 需要显式 `FunctionContract`。

### 标准库作为参考客户端：`@pl.jit_trace.body`

面向内核的拼接拼写就建在这一层上，而它正是本层最锐利的验收测试：一个只花四行公开表层的策略，而不是一条特权路径。内核作者只声明一种含义，然后像函数一样调用：

```python
@pl.jit_trace.body
def scale_block(a, out, row0, k): ...

scale_block(a, out, row0, k)          # 面向内核的拼写
```

仓库之外的框架仅凭本页就能得到同一个下降：

```python
prepared = pl.authoring.body(kernel)

@pl.authoring.recipe
def expand(ctx, *args, **kwargs):
    return ctx.splice_body(prepared, *args, **kwargs)
```

标准库在其上增加的是策略而非能力：装饰器形式、body 从内核保留的 `__name__`/`__doc__` 元数据、声明处的放置策略（`auto_scope=`），以及用 “body” 而非 “splice” 词汇表达的拒绝。如果某个面向内核的拼写需要本表层不具备的能力，修复应落在本层 —— 而不是绕过它的私有捷径。

## 控制能力

`ctx.control` 将语义终结符与转移暴露为上下文操作 —— 为棘手的用户自定义 hook 提供的逃生舱，与框架集成而非裸露：

| 能力 | 目标选择 |
| ---- | -------- |
| `emit_return(*values)` | 活动函数（或拼接本地边界 —— 拼接自身的返回规则适用）。 |
| `emit_yield(*operands)` | 最近的结果产生 region —— `pl.intrin.yield_` 通道。 |
| `emit_break()` / `emit_continue()` | 最近的合法循环，仅 staged —— recipe 不能转移 Python 控制流，因此 host 循环目标是显式错误。 |

每个旋钮都经由 Frame 自身的守卫（挂起的结果绑定、挂起的 carry 组、region 接受）。在无 staged 控制流打开时发出 `emit_return` 后，frame 的 region 即被终结：其后进入该 region 的每个后续层级效果 —— 跨越 recipe 调用，该标记随 trace 存续 —— 都作为死代码被拒绝。在 staged 分支内发出的返回是另一条可选返回路径（与 parser 对 `if` 内 `return` 的处理完全一致），不终结任何东西。在层级边界之外，终结符之后的语句仍是作者的契约（tracer 对终结符之后语句的自有纪律）。

## 与入口的关系

入口家族（`@pl.jit`、`@pl.jit_trace`、`.inline`、`.incore`）是*前端*：它们拥有特化、缓存与分发。授权运行时是其下的效果组合层 —— splice 的公开拼写是 `pl.authoring.body(...).splice(...)`，上面的组合方法是 callee 放置的显式拼写。解析前端仍是同一 IR 之上的独立后端。

## 另见

- [Construct Recipes](06-construct-recipes.md) —— 同一会话上的受约束 region 协议。
- [Functions and Program Structure](03-functions.md) —— contract 所描述的函数类别。
