# 用例配置

[返回功能手册](README.md)

配置文件描述待测入口、实际输入、独立参考、输出契约及测试政策。`validate` 在启动设备 worker 前检查字段和全部参数组合；通过校验不代表后端一定支持该算子或 dtype。

## 格式与主要字段

新配置使用 JSON，并写明 `"schema_version": 2`。这是配置协议版本，和软件发布版本独立。工具仍能读取 `schema_version: 1` 的配置，保持其默认行为；需要下述扩展字段时应使用版本 2。

| 位置 | 字段 | 说明 |
|---|---|---|
| 根对象 | `schema_version`、`cases` | 必填；`cases` 是非空数组 |
| 根对象 | `defaults` | 可统一设置 compare、benchmark、repeat、timeout_s、required、reference_policy、profile、seed |
| 用例 | `name`、`backend` | 名称及后端；展开后名称必须唯一 |
| 用例 | `entry`、`reference` | `模块:函数`，二者必须独立且不能是同一入口 |
| 用例 | `inputs`、`outputs` | 输入描述和输出契约；允许零输入，不允许零输出 |
| 用例 | `attrs`、`attr_types` | 标量属性及可选的显式类型约束 |
| 用例 | `parameters`、`symbols` | 有限取值列表及符号 shape 的有限绑定 |
| 用例 | `constraints` | 对展开后的具体参数组合做校验 |
| 用例 | `seed`、`timeout_s` | 随机种子、单用例超时秒数；默认 0 和 120 |
| 用例 | `required`、`skip`、`tags` | 是否必需、是否显式跳过、筛选标签 |

后端名称为 `cpu`、`torch_musa`、`musa_cpp`、`triton`、`tilelang`、`native`。CPU 是平台测试模式，不会被记为 GPU 验证。未知字段、重复 JSON key、非标准 JSON 的 NaN/Infinity 都会被拒绝。

## 完整最小示例

将下面内容保存为项目根目录下的 `my-case.json`：

```json
{
  "schema_version": 2,
  "cases": [{
    "name": "add",
    "backend": "cpu",
    "entry": "musaopst.examples.cpu_ops:add",
    "reference": "musaopst.examples.references:add",
    "inputs": [
      {"shape": [17], "dtype": "float32"},
      {"shape": [17], "dtype": "float32"}
    ],
    "outputs": [{"shape": [17], "dtype": "float32"}],
    "seed": 42,
    "compare": {"atol": 0.00001, "rtol": 0.00001}
  }]
}
```

```bash
python -m musaopst validate my-case.json
python -m musaopst run my-case.json --output runs/manual-configuration
```

预期展开 1 个用例；执行结果为 CPU 模式的 `PASS`。输出包含 `results.json`、`summary.md` 和 `junit.xml`。再次执行要使用新的输出目录。

## 参数化、符号和属性

完整示例见 [core-configuration.json](examples/core-configuration.json)，可直接运行：

```bash
python -m musaopst validate docs/manual/examples/core-configuration.json
python -m musaopst run docs/manual/examples/core-configuration.json --output runs/manual-configuration-matrix
```

示例对 `N=[4,8]` 和 `dtype=[float32,float64]` 做笛卡尔积，生成 `axpby[0]` 至 `axpby[3]` 四个用例，并记录 `parameters_selected`。下面是其中的配置片段：

```json
{
  "symbols": {"N": [4, 8]},
  "parameters": {"dtype": ["float32", "float64"]},
  "inputs": [
    {"shape": ["${N}"], "dtype": "${dtype}"},
    {"shape": ["${N}"], "dtype": "${dtype}"}
  ],
  "attrs": {"alpha": 0.5, "beta": -0.25},
  "attr_types": {"alpha": "float", "beta": "float"}
}
```

`${N}` 必须占据整个 JSON 值，不支持 `"2*${N}"` 或名称中拼接占位符。`symbols` 只接受非负整数，`parameters` 可接受 JSON 值；两者不能重名。属性类型支持 `bool`、`int`、`float`、`str` 和相应的 `list[type]`；这里的 `float` 类型声明也接受整数值。

## 组合约束

约束可引用属性 `{"attr":"名称"}`、展平后的张量维度 `{"input":0,"axis":0}` 或 JSON 常量。例如：

```json
{
  "constraints": [
    {"left": {"input": 0, "axis": 0}, "op": "eq", "right": {"input": 1, "axis": 0}},
    {"left": {"input": 0, "axis": 0}, "op": "divisible", "right": 4}
  ]
}
```

支持 `eq/ne/lt/le/gt/ge/divisible`，不执行任意表达式。普通配置中任何组合违反约束，整个配置校验失败，不会悄悄丢弃该组合。需要随机候选过滤时使用 [有限 Fuzz](fuzz.md)。

静态参数展开不意味着复用同一编译产物；相关区别见 [动态 shape](dynamic-shapes.md)。完整字段规则以 [配置校验实现](../../musaopst/config.py) 为准。
