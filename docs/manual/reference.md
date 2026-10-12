# 独立参考计算

[返回功能手册](README.md)

参考函数计算期望结果，必须独立于待测实现。它接收同一份实际输入的独立副本，避免待测函数原地修改输入或复用输出缓冲区污染参考。没有参考结果或没有完成比较，不能标记为精度通过。

## 参考函数与精度政策

用例的 `reference` 使用 `模块:函数` 格式。函数接收普通 NumPy 输入，参数绑定与待测入口一致；可返回单个数组或多个数组组成的列表/元组。参考函数不应调用待测设备 kernel。

`reference_policy` 控制参考输入的计算精度：

| 取值 | 浮点输入 | 复数输入 | 整数、布尔输入 |
|---|---|---|---|
| `float64`（默认） | 转为 float64 | 转为 complex128 | 保持精确类型 |
| `float32` | 转为 float32 | 转为 complex64 | 保持精确类型 |
| `preserve` | 保持输入存储精度 | 保持输入精度 | 保持精确类型 |

这些转换都发生在目标 dtype 量化之后。例如 float16 的 0.1 先变为实际可表示值，再升到 float64；参考不会偷偷用未量化的 0.1。BF16 的主机存储容器是 float32，`preserve` 保留这个容器及量化数值，并不让 NumPy 自动变成 BF16 算术。

政策约束参考函数收到的输入。参考函数内部若自行转换 dtype、选择不同累加精度，应在函数和用例说明中写清楚。整数参考保持输入精度可避免大整数经浮点转换丢失精确值，但不代表整数运算没有溢出，需要参考实现自行考虑算法语义。

## 使用示例

[core-reference.json](examples/core-reference.json) 对同一组 float16 固定数据分别使用三种参考精度：

```bash
python -m musaopst validate docs/manual/examples/core-reference.json
python -m musaopst run docs/manual/examples/core-reference.json --output runs/manual-reference
```

预期 3 个 CPU identity 用例通过。输出契约仍是 float16；参考结果可以是 float32/float64。检查每个用例的 `inputs.npz` 和 `reference.npz`，可以看到精度提升没有改变输入已经量化后的数值。

自定义 Softmax 参考函数模板如下；这是函数代码示例，应保存为可导入的 Python 模块，再配置对应入口：

```python
import numpy as np

def softmax_reference(x, dim=-1):
    if x.size == 0:
        return np.empty_like(x)
    shifted = x - np.max(x, axis=dim, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=dim, keepdims=True)
```

使用该函数时，用例可以包含以下片段：

```json
{
  "reference": "my_reference:softmax_reference",
  "reference_policy": "float64",
  "attrs": {"dim": -1}
}
```

已有可运行的 GPU Softmax 用例见 [gpu_smoke.json](../../examples/gpu_smoke.json)，不同后端通过同一套参考和比较政策进行检查。

## 常见错误

`entry` 与 `reference` 完全相同会被配置校验拒绝；写成两个不同名称但内部转调同一个待测实现，也不构成独立参考。命名输入使用 `a`，参考函数却仅接受 `x`，会造成关键字参数错误；请同步两侧签名，或采用位置绑定。

参考抛异常时记录 `REFERENCE_FAILED`，不继续假定精度通过；参考 shape 错误则在输出契约阶段失败。安全演示见 [预期失败](expected-failures.md)。
