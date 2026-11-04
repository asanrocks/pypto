# MaterializeEntryReturns Pass

PROGRAM（artifact）编译期间，把入口新建的、静态形状的结果缓冲区导出为由
host 分配的隐藏 `Out` 参数。对外签名保持作者声明的样子；只有运行时 ABI
增加交付结果所需的参数。

## 概述

运行时的外部结果 ABI 是整缓冲区的：入口交付的每个值都必须落在 host
能够传入并读回的参数里。一个*返回自己分配的缓冲区*的入口 ——

```python
out = pl.create_tensor([8, 8], dtype=pl.FP32)
...
return out
```

—— 交给调用者的东西无可读取：host 手上没有这个分配的句柄。本 pass 就是
修补这一点的边界适配器，并且与前端无关：`@pl.jit` 入口与 `@pl.jit_trace`
入口返回同样的新建分配时，经由同一个变换交付（created-output 测试套件对
两个装饰器参数化）。

该 pass 是基于既有 pass 接口的 Python program pass
（`python/pypto/ir/_entry_outputs.py`），不新增核心 IR 节点 —— 它改变的
不是 IR 结构，而是入口的 *ABI*：运行时派发所依据的参数表。这也是它作为
边界适配器而非普通优化的原因：它必须恰好运行一次，且运行在拥有该 ABI 的
编译模式下。

## 示例

面向作者的 kernel（tracing 前端；解析前端书写同样的形状）：

```python
@pl.jit_trace
def add_one(x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
    out = pl.create_tensor([8, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [8, 8]), 1.0), [0, 0], out)
    return out
```

pass 之前，入口持有该分配，host 无可读取（pass 前入口的
`ir.python_print` 输出）：

```python
def add_one(self, x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
    out: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.tile.store(pl.tile.adds(pl.tile.load(x, [0, 0], [8, 8], [8, 8]), 1.0), [0, 0], out)
    return out
```

Default 流水线之后（仅入口；其余 program 略）：

```python
def add_one(
    self,
    x__ssa_v0: pl.Tensor[[8, 8], pl.FP32, pl.MemRef(...)],
    entry_result_0: pl.Out[pl.Tensor[[8, 8], pl.FP32, pl.MemRef(...)]],
) -> pl.Tensor[[8, 8], pl.FP32]:
    pl.func_attr({"entry_output_abi": True, "entry_result_0": 1,
                  "entry_result_count": 1, "entry_synthesized_count": 1})
    ...
```

`tensor.create` 消失了；该分配变成了隐藏的 `entry_result_0` 参数，而
`entry_result_0: 1` 记录逻辑结果 0 由物理参数 1 交付。host 调用分配该
缓冲区、派发、再返回它 —— 对外调用仍是 `add_one(x)`。

## 运行时机

**流水线位置**：在
[ConvertTensorToTileOps](12-convert_tensor_to_tile_ops.md) 之后、
[OptimizeOrchTensors](14-optimize_orch_tensors.md) 之前。它要求该转换产出的
属性（`SSAForm`、`SplitIncoreOrch`、`NormalizedStmtStructure`、
`IncoreTileOps`），并先于编排层优化运行，使后续每个 pass —— 以及
codegen —— 看到的都是最终的物理参数表。

**激活条件**：只有 artifact 编译会请求 host 结果 ABI。`ir.compile`
（PROGRAM 模式）先运行 `prepare_program_outputs`，它解析入口的返回签名并
在入口上标记 `entry_output_abi` —— 但仅当运行时无元数据的回退规则
（"纯 `Out` 参数，按声明顺序"）已无法描述交付时才标记：新的顶层分配、
返回顺序不同于声明顺序、重复结果、或由 `InOut` 承载的结果。没有该属性，
本 pass 不做任何事。

**不适用**：结构性 `lower()`（不请求 host 结果 ABI）；KERNEL 编译（每个
结果必须别名为外部参数 —— 借用缓冲区契约）；超出唯一 traced host 入口的
分布式程序；带 L3+ 编排的程序；无单一可解析入口的程序。`tensor.create`
缓冲区尚未位于顶层的入口（普通 lowering 会把它们外提）在最终编译边界
重新检查，而不是在此处报错。

## 变换内容

1. 临时把入口顶层的 `tensor.create` 语句当作 `Out` 参数，运行共享的
   整缓冲区返回溯源分析 —— 即
   [NormalizeReturnOrder](30-normalize_return_order.md) 所规范化的同一套
   `returned_param_indices` / `delivered_param_indices` 遍历 —— 证明入口
   实际返回了哪些缓冲区。决定因素是来源（provenance）而非声明类型：转发
   的被调结果按它携带的值分类。
2. 若每个交付的值都已位于某个参数中，则不做存储变换；仅当来源映射与回退
   规则不一致时，才持久化该映射。
3. 否则：删除每个被提升的分配，为每个被返回的不同缓冲区追加一个隐藏
   `Out` 参数（`entry_result_<i>`，与入口参数名去重），将该参数替换进
   函数体，并记录 `result_indices` —— 每个物理参数承载哪个逻辑结果下标，
   重复下标保留别名。混合结果（新建缓冲区与参数承载的结果并存）保持源
   返回顺序。

## 拒绝情形（fail closed）

| 拒绝 | 原因 |
| ---- | ---- |
| 没有静态张量形状的新建输出 | host 必须在派发前分配隐藏缓冲区 |
| 新建输出带 `manual_dep=True` | create 语句 —— 生命周期粒度依赖豁免的唯一载体 —— 在提升时被删除；隐藏参数无法保留该语义 |
| 整缓冲区 ABI 无法交付的结果位置（标量或 tile 值的新建结果） | 运行时没有可打包的参数槽位 |
| 来源不完整 —— 返回值既不由参数承载，也不是新的顶层静态分配 | 宁可失败，也不交付空结果 |

## 向下游保留什么

`entry_result_*` 函数属性在后续 pass 中携带该映射，不依赖返回顺序规范化。
artifact 元数据（`compiled_meta.json`，schema 4）持久化
`synthesized_outputs` 与 `result_indices`；host 派发每次调用都分配隐藏
缓冲区；`CompiledProgram.from_dir` 重载恢复同样的分配与打包规则，而
`CompiledProgram.result_indices` 是直接派发与 ChipWorker 路径共享的唯一
打包规则。

## API

| Python | 层级 |
| ------ | ---- |
| `materialize_entry_returns()` —— 注册于 Default 流水线 | Program 级 |

## 另见

- [ConvertTensorToTileOps](12-convert_tensor_to_tile_ops.md) —— 本适配器所需 `IncoreTileOps` 属性的生产者。
- [NormalizeReturnOrder](30-normalize_return_order.md) —— 两者共同消费的返回溯源分析。
- [函数与程序结构](../language/03-functions.md) —— 作者视角下的 created outputs。
