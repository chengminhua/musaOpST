# 输出契约

[返回功能手册](README.md)

输出契约先检查输出数量、shape、dtype，再比较数值。shape 不同不会靠广播“比较成功”，即使元素数量相同也必须报告契约错误。输出可为单张量或列表/元组；列表顺序必须与 `outputs` 声明一致。

## 参数

| 字段 | 说明 |
|---|---|
| `shape`、`dtype` | 每个输出必填，描述实际待测输出 |
| `name` | 可选说明名称，不改变输出的位置顺序 |
| `compare` | 当前输出的比较政策，覆盖用例和 dtype 政策 |
| `layout` | 可检查 `strides`、`storage_offset`、`contiguous` |
| `alias_input` | 要求与某个展平输入索引或唯一输入名共享存储 |
| `shape: "reference"` | 由独立参考确定数据相关 shape，同时必须指定 rank 和 max_elements |

实际输出 dtype 必须符合契约；参考结果可以使用更高计算精度，不要求和实际输出采用相同 dtype。

## 多输出使用示例

[core-output-contracts.json](examples/core-output-contracts.json) 返回 float32 数值和 int64 索引，分别采用容差和严格相等比较：

```bash
python -m musaopst validate docs/manual/examples/core-output-contracts.json
python -m musaopst run docs/manual/examples/core-output-contracts.json --output runs/manual-output-contracts
```

预期 1 个 CPU `PASS`，报告中有两项 `comparison.outputs`。对应配置片段：

```json
{
  "outputs": [
    {"name": "shifted", "shape": [4], "dtype": "float32", "compare": {"atol": 0.000001, "rtol": 0.000001}},
    {"name": "indices", "shape": [4], "dtype": "int64", "compare": {"exact": true}}
  ]
}
```

真实 MUSA 的混合 dtype 示例：

```bash
python -m musaopst run examples/suites/contracts_matrix.json --case torch_mixed_dtype_outputs --device 0 --output runs/manual-output-contracts-gpu
```

## 布局与别名

以下输出契约片段要求输出是输入 0 的非连续别名，而不仅是值相同的复制品：

```json
{
  "shape": [4],
  "dtype": "float32",
  "layout": {"strides": [2], "storage_offset": 3, "contiguous": false},
  "alias_input": 0
}
```

可以运行 `core-layouts.json` 中的 `inplace_alias` 用例：

```bash
python -m musaopst run docs/manual/examples/core-layouts.json --case inplace_alias --output runs/manual-output-contracts-alias
```

别名信息在输出复制成快照之前采集。TensorList 的索引按深度优先展平，仅张量叶子计数，`None` 不占索引；名称必须能唯一定位一个张量输入。alias 证明共享底层存储，不等同于逻辑首元素地址相同，后者可结合 offset/strides 限定。

## 数据相关输出

索引筛选等算子输出长度取决于输入数据。配置片段：

```json
{"shape": "reference", "dtype": "int64", "rank": 2, "max_elements": 64}
```

参考先产生期望 shape；工具检查参考 rank、元素上限以及实际输出是否与参考一致，不从实际输出反过来定义正确答案。当前现成的此类设备示例主要是 torch_musa 筛选；其他后端仍需开发支持变长输出分配的适配器。

常见错误包括返回了两项但只声明一项、标量 `[]` 误写为 `[1]`、把 int64 索引声明成 float32、复制输出却要求 alias。它们会进入 `comparison.contract_errors`，并导致精度失败。
