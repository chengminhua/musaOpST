# 编译期常量输入

[返回手册目录](README.md)

## 功能说明

固定输入值和编译期常量不同。`distribution.kind: fixed` 只固定输入数据；`constant: true` 表示该输入需要在构建阶段嵌入算子代码。当前仅 Triton 和 TileLang 的 `constant_scale` 示例支持此声明，其他后端显式报告不支持。

## 支持范围与参数

- 两个输入：连续非空的一维向量，以及 `shape: []` 的有限标量。
- 两者 dtype 相同，可选 float32、float16、bfloat16。
- 第二个输入配置 `constant: true`，第一个不声明常量。
- 工具先量化并上传实际输入，构建阶段读取实际标量后嵌入 constexpr 或 TIR。修改常量会产生新的特化。
- 任意 tensor 常量、任意算子的常量折叠不在此例支持范围内。

## 使用示例

现成配置包括支持与不支持的组合，先校验，再选支持的后端执行：

```bash
python -m musaopst validate examples/suites/constants_matrix.json
python -m musaopst run examples/suites/constants_matrix.json --case 'triton_compiled_constant_*' --device 0 --output runs/manual-constants-triton
python -m musaopst run examples/suites/constants_matrix.json --case 'tilelang_compiled_constant_*' --device 0 --output runs/manual-constants-tilelang
```

这两条 `run` 需要对应的 MUSA JIT 环境。直接执行全部配置时，未实现该功能的后端会保留 `UNSUPPORTED` 并使套件返回非零，不会当作普通固定输入运行。

## 查看结果

查看用例 `backend_metadata.launch` 中的 `compiled_constants`、实际量化值和 `artifact_id`。Triton 的标识来自编译 kernel；TileLang 当前记录生成设备源码摘要，不把它称为已测量的二进制摘要。独立参考始终使用同一份量化后的输入值。

低精度的 JSON 字面量未必等于实际输入，例如 1.1 量化后会发生舍入。以快照和 `quantized_value` 为准，不能只对照配置文字。
