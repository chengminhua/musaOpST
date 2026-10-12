# 张量布局

[返回功能手册](README.md)

布局描述实际内存访问方式。工具支持连续布局、带空隙的非连续布局、显式元素 stride 和 storage offset，以及基于底层 shape 的转置和正步长切片。设备执行后会记录实际输入布局；不能把输入转连续后仍声称测试了非连续布局。

## 参数与范围

| 字段 | 含义 |
|---|---|
| `layout: "contiguous"` | 默认连续布局 |
| `layout: "strided"` | 每个逻辑元素间留空隙的预设布局，最内层元素 stride 为 2 |
| `strides` | 每个维度的元素步幅，不是字节数；指定后自动使用 explicit 布局 |
| `storage_offset` | 逻辑首元素相对底层存储的元素偏移，默认 0 |
| `base_shape` | 生成或文件读取时的底层 shape |
| `transforms` | 按顺序执行 transpose/slice，最后必须得到声明的 `shape` |

显式 stride 必须为正，并满足工具的非重叠校验。负 stride、零 stride、重叠视图不支持。`transforms` 不能和显式 strides、非零 storage offset 或 `layout: "strided"` 混用。任意 dtype/布局是否能由某个 kernel 执行，仍由该入口的能力决定。

## 使用示例

[core-layouts.json](examples/core-layouts.json) 包含显式输入步幅、转置后切片、原地输出别名三个 CPU 示例：

```bash
python -m musaopst validate docs/manual/examples/core-layouts.json
python -m musaopst run docs/manual/examples/core-layouts.json --output runs/manual-layouts
```

预期 3 个 CPU `PASS`。显式输入的配置片段：

```json
{"shape": [4], "dtype": "float32", "strides": [2], "storage_offset": 3}
```

它表示依次访问底层存储索引 3、5、7、9。输出若为新分配连续张量完全合法；要强制输出也保持某种布局，必须另写 [输出布局契约](output-contracts.md)。

转置和切片示例片段：

```json
{
  "shape": [4, 3],
  "base_shape": [3, 8],
  "dtype": "float32",
  "transforms": [
    {"kind": "transpose", "axes": [1, 0]},
    {"kind": "slice", "start": [1, 0], "stop": [8, 3], "step": [2, 1]}
  ]
}
```

先生成 `[3,8]`，转置为 `[8,3]`，再取第一个维度的 1、3、5、7 行，得到 `[4,3]`；实际元素 strides 为 `[2,8]`，offset 为 1。`slice` 每个参数都按维度列出，stop 不包含在结果中，step 必须为正。

在已配置 MUSA 的环境运行对应 GPU 示例：

```bash
python -m musaopst run examples/suites/contracts_matrix.json --case torch_musa_transpose_slice_input --device 0 --output runs/manual-layouts-gpu
```

结果的 `inputs` 与 `backend_metadata.inputs` 提供构造侧和设备侧布局，`input_layout_checks` 记录已检查字段。NPZ 快照保存逻辑数值，不能仅凭 NPZ 数组自身的 strides 判断原设备布局；回放会结合配置重建布局。

shape 正确但 stride 不支持时，不应主动加 `.contiguous()` 绕过测试。可以明确写一个只支持连续布局的新用例，并保留原用例的限制说明。
