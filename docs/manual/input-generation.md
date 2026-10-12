# 生成输入

[返回功能手册](README.md)

工具按配置生成主机输入，先量化为目标 dtype，再分别复制给待测实现和参考函数。同一配置、种子和依赖环境可复现随机序列；需要跨运行固定实际数值时，使用保存的输入快照或文件输入。

## 分布与参数

`distribution` 写在每个张量输入中。省略时使用标准正态分布。

| `kind` | 参数 | 取值范围或含义 |
|---|---|---|
| `normal` | `mean`、`std` | 默认 0、1；std ≥ 0 |
| `uniform` | `low`、`high` | 默认 0、1；low < high |
| `beta` | `alpha`、`beta` | 两者均 > 0 |
| `laplace` | `loc`、`scale` | scale ≥ 0 |
| `triangular` | `left`、`mode`、`right` | left ≤ mode ≤ right，且 left < right |
| `integers` | `low`、`high` | 整数区间 `[low, high)`，不含 high |
| `bernoulli` | `p` | 真值概率，0 ≤ p ≤ 1 |
| `fixed` | `values`、可选 `imag` | 标量填充或与生成 shape 相符的矩形数组 |

`seed` 位于用例层，范围为 `[0, 2^64)`。每个用例建立独立的随机生成器，并按输入声明顺序消费随机数；不要假设交换输入顺序后，各输入还保留原值。

## 使用示例

[core-input-generation.json](examples/core-input-generation.json) 用参数化方式覆盖上表八种分布，并使用 CPU identity 核对量化后的数据：

```bash
python -m musaopst validate docs/manual/examples/core-input-generation.json
python -m musaopst data docs/manual/examples/core-input-generation.json --output runs/manual-input-generation-data
python -m musaopst run --prepared runs/manual-input-generation-data --output runs/manual-input-generation
```

`data` 生成实际输入和 `preparation.json`，不执行算子、不宣称精度通过。最后一步才执行比较，预期得到 8 个 CPU `PASS`。各用例的 `inputs.npz` 保存已量化的实际值。

浮点分布的配置片段：

```json
{
  "shape": [2, 17],
  "dtype": "float16",
  "distribution": {"kind": "uniform", "low": -1, "high": 1}
}
```

固定复数的配置片段：

```json
{
  "shape": [2],
  "dtype": "complex64",
  "distribution": {"kind": "fixed", "values": [1, -2], "imag": [0.5, 3]}
}
```

固定值为单个数时会填满张量；为数组时必须与 `base_shape`（若有）或 `shape` 相同，不能依赖广播。空张量可用 `"values": []`。非固定的复数分布会分别生成实部和虚部。

## dtype 与常见问题

配置协议版本 2 接受 `float16/bfloat16/float32/float64`、`int8/int16/int32/int64`、`uint8/uint16/uint32/uint64`、`bool`、`complex64/complex128`。这些是平台可描述的数据类型；后端与算子支持范围应分别确认。

整数边界测试建议使用 `fixed` 或 `integers`。固定整数会检查范围与整数性，避免大整数先转成浮点后丢失低位。普通连续分布转成整数会发生类型转换，不适合作为整数边界样本。

BF16 使用先舍入到目标精度、再以 float32 容器保存的方式；参考函数接收的是已量化的数值。FP16 同样先舍入，不会把未量化的随机值直接送给参考。CPU 普通 NumPy callable 不提供原生 BF16 张量返回协议，应在 MUSA 后端验证 BF16 输出。

配置允许的 dtype 不代表所有 GPU kernel 都支持它。后端不支持时会显式报错或返回 `UNSUPPORTED`，不会回退 CPU。`source` 与 `distribution` 不能同时出现；读取现成数据请看 [文件输入](file-input.md)，冻结输入的工作流见 [独立造数与准备](data-preparation.md)。
