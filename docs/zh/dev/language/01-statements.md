# 语句与控制流

tracing 前端收集的语句形式，以及 SSA pass 如何从 rebinding 推导 phi 节点。类型系统与表达式
文法参见 [Python IR 语法规范](00-python_syntax.md)；每个 parser 时代的拼写及其替换形式参见
[迁移指南](../../../design/04-migration-guide.md)。

打印出的 round-trip 方言是另一套文本。IR printer 会把带 iter args 的 lowering 循环渲染为
`init_values=` 加 assignment 形式的 `pl.yield_`，把 SSA 形式的 while 渲染为
`pl.while_(init_values=...)` 加首条 `pl.cond(...)`。那是 parser 的文法（[IR
Parser](../ir/07-parser.md)）——它出现的地方都是正确的，但绝不是 tracing 输入。不要把它抄回
kernel。

## 语句 (Statement)

### 赋值

```python
x = expr                             # 标注可选
y: pl.Tensor[[4], pl.FP32] = tensor_op(a)
```

tracing 下局部标注永远不会流入 IR：值的类型来自产生它的算子。复述类型的标注是文档；与实际
类型不符的标注则被静默忽略成死文本——迁移时请自行审计（指南 §4.4）。函数声明的 `->` 是例外：
body 不满足的返回契约会在装饰期报错。

增量赋值（augmented assignment）会经由操作数的二元算子 lowering 成一次 rebind，因此只要该
算子存在就能用：staged `Scalar` 支持 `acc += 1`。Tile 与 Tensor 包装类不重载算术运算，所以
对它们做 `acc += ...` / `acc[...] += ...` 会在 trace 期触发 Python 自身的
`TypeError: unsupported operand type(s) for +=: 'Tile' and 'Tile'` ——请改写显式形式
（`acc = pl.add(acc, ...)`）。矩阵归约应使用带 `init_cond` 的 `matmul_acc`：

```python
acc[t0 : t0 + R, :] = pl.matmul_acc(
    acc[t0 : t0 + R, :], x_k, w_k, b_trans=True, init_cond=(k0 == 0)
)
```

第一个 K 步只覆盖该窗口，后续步骤在该窗口内累加，其他窗口保留原值。
编译器可将局部累加器中大小一致的行窗口打包为连续的 L0C 窗口；参见
[累加器逻辑行窗口](../passes/14-flatten_tile_nd_to_2d.md#累加器逻辑行窗口)。

### If 语句 (staged)

staged `if` 会 trace **两个**分支；各分支用普通赋值发布其值，SSA pass 合成 merge（phi 节点）：

```python
# If with both branches — same names, plain assignments
if condition:
    y1 = value1
else:
    y1 = value2

# Multiple values at once
if condition:
    y1, y2 = value1, value2
else:
    y1, y2 = value3, value4
```

**要点:**

- 两个分支赋值相同的名字；收敛是 pass 的事。
- staged 值就是普通赋值；跨越 join 的**宿主**值需要 `pl.const(value, dtype)`——一个名字在
  一个分支是 staged、在另一个分支是宿主字面量，会被拒绝。
- 读取只在部分分支绑定的名字是响亮报错（`UnboundLocalError` 纪律）：在 `if` 之前绑定，或在
  每个分支都绑定。
- 谓词必须是 BOOL 类型——对整型 `Scalar` 写 `if k:` 会被拒绝；写显式判断（`if k != 0:`）。
  **宿主**谓词在 trace 期只走一条路径（这是真正的元编程，不是回退）。

### For 循环

```python
# 简单循环 (1-3 个位置参数，类似 Python 的 range())
for i in pl.range(stop):                    # start=0, step=1
for i in pl.range(start, stop):             # step=1
for i in pl.range(start, stop, step):       # 完整形式

# 循环携带值由 rebinding 推导——无需任何声明。
s = pl.const(0, pl.INT64)                   # 类型化 staged 种子
for i in pl.range(n):
    s = s + i                               # rebind 本身就是 carry
s_final = s                                 # 读取最后一次迭代的结果

# 并行 for 循环 (同样支持 1-3 个参数)
for i in pl.parallel(stop):
for i in pl.parallel(start, stop, step):
    body_statements
```

**要点:** `pl.range()` 不再接受 `init_values=`（也不再有随之而来的 iter 元组解包）。循环的
携带值就是 body 中自引用地 rebind 的那些名字；同一循环可以 rebind 多个名字，body 内的单分支
定义也合法——缺失的分支读取上一次迭代的值，这是 pass 拥有的循环携带语义。种子必须是 staged
的：宿主字面量种子（`s = 0`）会在 trace 期被拒绝（`typed staged initialization: ... bind a
typed constant first: s = pl.const(..., dtype)`），因为它会在每次迭代内联该字面量而不是推导出
carry。用算子结果或 `pl.const(value, dtype)` 做种子。`pl.parallel()` 生成 `ForKind.Parallel`
循环, `pl.range()` 生成 `ForKind.Sequential` (默认)。

### While 循环

staged while 循环是对 **staged** 谓词的裸 `while`；`pl.while_` 和 `pl.cond()` 作为输入拼写
已不存在。body trace 一次代表性迭代，carry 的推导方式与 `pl.range` 完全一致：

```python
# 自然 while：条件就是 while 头部表达式
i = pl.const(0, pl.INT64)     # 类型化 staged 种子
while i < n:                  # staged 谓词
    i = i + 1                 # rebind 就是 carry
```

种子必须是 staged 的：标注形式的种子（`i: pl.Scalar[pl.INT64] = 0`）在 tracing 下**不会**
staged——局部标注永远不流入 IR——于是计数器保持宿主值，循环退化（谓词对字面量折叠，body 无
任何贡献；宿主界的循环则在 trace 期执行，body 按宿主迭代次数逐次发射，完全没有 WhileStmt）。

staged 还是宿主由*值*决定，而不是关键字：如果谓词及其读取的名字都是宿主值——普通 Python
整数、张量形状、任何不是算子产生的东西——`while` 就在 trace 期运行。这是真正的元编程
（形状驱动的展开），不是回退。

### 作用域上下文管理器 (Scope Context Managers)

| 形式 | Scope 类型 | 说明 |
| ---- | ---------- | ---- |
| `pl.at(level=pl.Level.CORE_GROUP)` | `InCore` | CORE_GROUP 级固定边界 outline |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(MODE)])` | `InCore` | InCore + 跨核 split 提示 |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(slot_num=N)])` | `InCore` | InCore + 跨核 pipe 槽位数 |
| `pl.at(level=pl.Level.HOST)`（或任意非 `CORE_GROUP` 级别） | `Hierarchy` | 分布式层级作用域 |
| `pl.cluster()` | `Cluster` | AIC+AIV 协同调度组 |
| `pl.graph("name")` | `Graph` | 可录制的编排区域；名字必填，并成为提取函数的名字 —— 见 [OutlineGraphScopes](../passes/08-outline_graph_scopes.md) |
| `with pl.spmd(N)` / `for i in pl.spmd(N)` | `Spmd`（for-form 内嵌 `InCore`） | SPMD 多 block 派发——见 [pl.spmd](#plspmd-多-block-派发) |
| `pl.spmd(N, optimizations=[pl.split(MODE)])` | `Spmd(InCore(split=MODE))` | split 提示作用于内层 InCore（两种形式均适用） |
| `pl.spmd(N, optimizations=[pl.cross_core_slot(slot_num=N)])` | `Spmd(InCore(slot_num=N))` | 槽位数作用于内层 InCore（两种形式均适用），可与 `pl.split(MODE)` 组合 |
| `pl.scope(mode=pl.ScopeMode.MANUAL)` / `pl.manual_scope()` | `Runtime(manual=true)` | orchestrator 的 MANUAL scope——由用户管理任务排序。两种 `auto_scope` 模式下都可用（它是依赖语义选择）。见[手工依赖原语](02-manual_dependencies.md#手工依赖原语) |
| `pl.scope()` | `Runtime(manual=false)` | orchestrator 的 AUTO scope（`SIMPLER_SCOPE()`）。手写它需要 `@pl.function(auto_scope=False)`（默认 `auto_scope=True` 下由编译器决定 AUTO 放置）。见 [MaterializeRuntimeScopes](../passes/49-materialize_runtime_scopes.md) |

#### `pl.spmd` 多 block 派发

`pl.spmd(N)` 把一个 kernel 派发到 `N` 个 block。形式：

- `with pl.spmd(N): ...` —— body **既可以**是调用已声明 InCore kernel 的*派发型* body（`SpmdScopeStmt(body=<stmts>)`，无内层 InCore 包裹），**也可以**是一段*内联*块，自动外包成一段隐式 InCore 区域（与 for-form 相同，只是不自动绑定索引）。区分依据是语义而非语句数量：body 若读取 `pl.tile.get_block_idx()`，即为内联 body 并被包裹；否则即为派发型 body，无论包含多少条语句都不加包裹。若 body 既不读取索引、也不派发 `self.<kernel>(...)` 调用，则会被拒绝。不捕获 producer TaskId。
  - 当 body 唯一的语句是显式的 `with pl.at(<CORE_GROUP level>, ...):` 时，该嵌套 scope 本身*就是* InCore 载体：它会被当作普通嵌套 scope 解析，而不会被二次包裹（`level` 可用位置或关键字形式，也可带 `as tid` / `name_hint=`）。printer 正是以这种形式输出 `Spmd(InCore(...))`，因此这也是该 IR 能够 round-trip 的原因。当 body 已提供载体时，`optimizations=` 必须写在该 `pl.at(...)` 上；写在 `pl.spmd(...)` 行会被拒绝，无论载体自身是否也带有该项。
  - 派发型 body 只能启动**一个** kernel。它经由 `FindFirstInnerCall` 下降，而后者在第一个调用处即停止，因此第二个派发不会被启动而是被静默丢弃；parser 会直接拒绝这种写法。提升出的临时变量与 tuple 投影不算派发，不计入数量。
- `for i in pl.spmd(N): ...` —— 循环变量绑定到每个 block 的索引（`pl.tile.get_block_idx()`）；body 自动外包成一段隐式 InCore 区域。
- `with pl.spmd(N, deps=[...]) as tid: ...` —— **捕获形式**：与 `with pl.at(...) as tid:` 对称。body 形态与上面的普通形式相同，并额外在 `tid` 中捕获该分发的 grid 级 producer `pl.Scalar[pl.TASK_ID]`（可用作 `deps=` 边、存入 `pl.array.create(N, pl.TASK_ID)`、或跨入 `pl.manual_scope`）。TaskId 捕获与内联 body 正交——这是该形式相比普通形式唯一多出来的能力。lower 成一个 `ir.Submit`，其尾部 tuple 元素即 grid TaskId；`core_num` / `sync_start` 记录在该 `Submit` 自身的字段上（launch spec 属于启动点，而非外包出的被调函数）。参见下文“手动依赖原语”小节。
- `out, tid = pl.spmd_submit(kernel, *args, core_num=N)` —— **submit 形式**：将 kernel 在 `N` 个 block 上分发，同时捕获该分发的 producer `pl.Scalar[pl.TASK_ID]`（针对已声明 kernel 的 `pl.submit` 版本）。参见下文“手动依赖原语”小节。

以上三种形式也都接受 `allow_early_resolve=True`（布尔字面量；与 `pl.submit` / `pl.at` 相同的 early-dispatch 选项）。即使不写 `as tid` 也会强制走 `ir.Submit` 形态，并 lower 为 `Arg::set_allow_early_resolve(true)`。在嵌套于 `pl.cluster()` 内的 `pl.spmd` 上会被拒绝（此类 scope 会被 unwrap 进 Group 函数、永远不会产生 Submit，提示会丢失）。

可选 `optimizations=[...]`。各条目彼此正交，可在同一列表中组合
（例如 `[pl.split(MODE), pl.cross_core_slot(slot_num=4)]`）：

| 条目 | 适用形式 | 作用 |
| ---- | -------- | ---- |
| `pl.split(MODE)` | 两种均适用 | 给内层 InCore 设置 `split_` 字段（跨核数据搬运提示，由 `ExpandMixedKernel` / `MemoryReuse` 消费）。with-form 会在原 call 外多包一层 `InCoreScopeStmt` 来承载该字段。 |
| `pl.cross_core_slot(slot_num=N)` | 两种均适用 | 给内层 InCore 设置 `slot_num` 属性——自动跨核 pipe 的槽位数（环深），由 `ExpandMixedKernel` 消费。它只决定数据通道大小，**不**划分计算，因此可与 `pl.split_aiv` 区域共存（而 `pl.split(...)` 不能）。省略时沿用默认深度：每个活跃方向 2 个槽位。 |

> `pl.split(MODE, slot_num=N)` 是该槽位数的已废弃别名，会发出警告——参见
> [ExpandMixedKernel](../passes/24-expand_mixed_kernel.md#覆盖槽位数slot_num)。

示例参见 [作用域与放置](../../user/language/04-scopes.md)。

### Yield 语句

裸 `yield` 在 tracing 下是 rewrite-unsafe 构造——前端执行 body，而生成器对 IR 的影响无法被
观测——因此会在 rewrite 期被拒绝。发布值的途径是：

- `return` —— 携带结果（可以是多个）离开函数；
- `pl.intrin.yield_(...)` —— 在推导会遮蔽数据流处声明 staged 分支/循环值的汇编通道（迁移指南
  附录 A）。`pl.yield_` 仍是它的已废弃别名；会发出警告；
- 打印出的 IR 把 merge 与 carry 渲染为 assignment 形式的 `pl.yield_`——这是给 parser 的
  round-trip 文本，不是输入。

### Break 和 Continue

```python
break              # 退出最内层循环
continue           # 跳到下一次迭代
```

**限制:** 仅当**最内层**封闭循环为顺序循环 (`pl.range`) 或 `while` 时有效。当最内层循环为 `pl.parallel()` 或 `pl.unroll()` 时不支持——两者都会在 trace 期被拒绝（`'break' not supported in parallel loops` / `... in unrolled loops`）。在外层 `pl.parallel` 循环内嵌套的内层 `pl.range` 循环中使用 `break` 是合法的。**注意:** 代码生成后端对 `break`/`continue` 的支持跟踪在 [#448](https://github.com/hw-native-sys/pypto/issues/448) 中。

### 编译期调试 (Compile-Time Debugging)

`pl.static_print()` 和 `pl.static_assert()` 是 AST parser 的诊断构造：它们**不生成任何 IR**，
只在 parser 拥有前端的地方运行（round-trip 文本、`@pl.function_parser` fixture）。在 tracing
前端下它们是**静默 no-op**——body 照常执行，调用没有任何效果，失败的 `static_assert` 不会
抛错。把断言移进一个会 raise 的宿主 `if`，或移进 verifier pass（指南 §8）。

```python
@pl.function
def func(x: pl.Tensor[[128, 64], pl.FP16]) -> pl.Tensor[[128, 64], pl.FP16]:
    pl.static_print("input:", x)          # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_print(f"input: {x}")        # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_assert(True)                # 静默通过
    pl.static_assert(N > 32, "N too small")  # 在解析期检查闭包变量 N
    return x
```

（上面的输出注释属于 parser 通道；tracing 下这四行都不产生任何输出。）

| 函数 | 用途 | 失败时 |
| ---- | ---- | ------ |
| `pl.static_print(*args)` | 将变量类型/值打印到 stdout | 需要 ≥1 个参数 |
| `pl.static_assert(cond, msg="")` | 断言编译期条件 | 抛出 `ParserError`——仅 parser 通道；tracing 下静默 |
| `pl.dump_tag(tensor)` | 把某个张量标记为运行期选择性 dump 的目标 —— 声明式逐张量标记（在 Orchestration 作用域，或被 orch 内联的 Inline helper 中均可使用 —— 见 [运行期 DFX](../03-runtime-dfx.md#选择性张量-dump)） | 在非 Orchestration / Inline 函数中使用、或参数不是裸变量名时抛出 `TypeError` |

**要点：**

- 三者均为语句级构造（不能用在表达式中）
- `static_print` 接受变量、常量、字符串标签（原样打印）和 f-string 的简单 `{expr}` 占位符（格式化为 IR）。不支持转换标志（`!r`、`!s`、`!a`）和格式说明符（`:...`）。
- `static_assert` 支持闭包变量表达式（如 `N > 32`）和 IR 常量；消息参数必须是字符串字面量
- `dump_tag` 接受单个绑定在外层 Orchestration（或 Inline）作用域内的张量变量名；tracing 前端会兑现它——标记在 trace 期被消耗，并自始至终以 Var 身份（而非名字）跟踪到 codegen。在显式 `self.kernel(...)` 调用点，它把该张量记录到每个后续消费它的 Call 的 `dump_vars` 上；在 `@pl.jit` / `with pl.at(level=...)` 风格（派发由 outline pass 合成）下，它改为写入所在 scope 的 `dump_vars`，再由 outliner 映射到合成派发的实参上（见 [运行期 DFX](../03-runtime-dfx.md#选择性张量-dump)）。若需要在单次 task 启动处显式列出 dump 目标，请用 `pl.submit(...)` / `pl.at(...)` 上的 `dumps=[...]` kwarg（与 `deps=` 对称）
- parser 通道的输出即使后续解析失败也会显示——适用于调试解析错误

### 语句序列

```python
stmt1            # Natural Python sequencing
stmt2
stmt3
```

## SSA 如何推导

tracer 收集 body 实际执行的内容并自由重命名绑定，因此普通 Python 就是正常输入——
[ConvertToSSA](../passes/04-convert_to_ssa.md) 在流水线早期运行，拥有全部图级数据流。源代码
层面不为 merge 或 carry 做任何声明：

```python
# If: merge 点的 phi 节点由两次赋值合成
if condition:
    y1 = pl.add(x, 1.0)
else:
    y1 = pl.add(x, 2.0)
# y1 = phi(x + 1.0, x + 2.0) —— ConvertToSSA 之后

# For: 循环携带值就是自引用的 rebind
s = pl.const(0, pl.INT64)
for i in pl.range(10):
    s = s + i
s_final = s  # 捕获最终值
```

唯一仍然显式声明 carry 的地方是汇编通道（`pl.intrin.carry` / `yield_` / `leak`，迁移指南
附录 A）——编译器 fixture，以及希望直接点名循环 iter args 的 kernel 代码在用（任务图依赖循环
同样可以从 rebind 推导 carry；指南 §6.3）。

printer 对同一控制流的 round-trip 渲染是显式方言：`for i, (s,) in pl.range(10,
init_values=(...)):` 加 body 内 assignment 形式的 `pl.yield_`，SSA 形式的 while 则是
`pl.while_(init_values=...)` 加首条 `pl.cond(...)`。那是 `pl.parse` 消费的文本（见
[IR Parser](../ir/07-parser.md)）；tracing 输入从不那样拼写 carry。

## 参考资料

- [Python IR 语法规范](00-python_syntax.md) —— 类型与表达式
- [手工依赖原语](02-manual_dependencies.md) —— 退出自动依赖跟踪
- [函数与程序结构](03-functions.md) —— 函数类型与参数方向
- [控制流](../../user/language/02-control-flow.md) —— 同一组规则的用户侧讲述
- [迁移指南](../../../design/04-migration-guide.md) —— 每个 parser 时代拼写及其 tracing 替换
