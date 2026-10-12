# 输入绑定

[返回功能手册](README.md)

输入可以是普通张量、显式缺省值、TensorList，也可以为空列表。待测函数和参考函数使用同一棵输入结构及同一套绑定规则，确保两边计算同一份输入所代表的数值。

## 支持范围与参数

| 配置 | 调用语义 |
|---|---|
| 默认 `binding: "positional"` | 顶层输入按声明顺序传入位置参数 |
| 用例 `binding: "named"` | 所有顶层输入都要有 `name`，作为关键字参数传入 |
| 输入自己的 `binding` | 覆盖该输入的绑定方式；命名输入需有 `name` |
| `kind: "absent"` | 显式传入 Python `None`，不是省略参数 |
| `kind: "tensor_list"` | `items` 保留为一个列表参数，不展开为多个函数参数 |
| `inputs: []` | 零张量输入；仍可通过 `attrs` 传标量属性 |

`attrs` 总是作为关键字参数传入。命名输入不能与属性重名，顶层输入名称不能重复。TensorList 子项不能声明命名绑定。结构层允许空列表，但具体算子可以拒绝空 TensorList。

## 可直接运行的 CPU 示例

[core-input-binding.json](examples/core-input-binding.json) 包含命名 identity、可选参数缺省、可选参数存在、TensorList 求和四个例子：

```bash
python -m musaopst validate docs/manual/examples/core-input-binding.json
python -m musaopst run docs/manual/examples/core-input-binding.json --output runs/manual-input-binding
```

预期 4 个 CPU `PASS`。其中可选输入和 TensorList 使用位置参数，命名 identity 的待测与参考函数都接受参数 `x`。

命名可选输入的配置片段：

```json
{
  "binding": "named",
  "inputs": [
    {"name": "a", "shape": [9], "dtype": "float32"},
    {"name": "b", "kind": "absent"}
  ]
}
```

它对应 `fn(a=tensor, b=None, **attrs)`。将第二项替换为 `{"name":"b","shape":[9],"dtype":"float32"}` 即传入实际张量。具体 GPU 示例：

```bash
python -m musaopst run examples/suites/contracts_matrix.json --case torch_musa_named_optional_absent --device 0 --output runs/manual-input-binding-gpu
```

## TensorList 与零输入

以下配置片段对应一个 `xs=[tensor0,tensor1]` 参数：

```json
{
  "binding": "named",
  "inputs": [{
    "name": "xs",
    "kind": "tensor_list",
    "items": [
      {"shape": [9], "dtype": "float32"},
      {"shape": [9], "dtype": "float32"}
    ]
  }]
}
```

零输入 fill 的完整 GPU 用例已放在示例矩阵中：

```bash
python -m musaopst run examples/suites/contracts_matrix.json --case torch_musa_zero_input --device 0 --output runs/manual-input-binding-zero
```

该用例通过 `attrs.shape`、`attrs.dtype`、`attrs.value` 决定填充值和输出规格，预期输出 `[2,9]` 的 float32 张量。

当前五种 GPU 路径均有可选输入、TensorList、零输入示例，但 C++、Triton、TileLang、原生路径的内建 TensorList 示例限两个张量；框架支持结构不等于任意 kernel 自动支持任意长度。命名绑定报 `unexpected keyword argument` 时应检查两侧函数签名，而不是随意改输入名称。接入方式见各 [后端文档](README.md#后端接入)。
