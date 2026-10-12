# 精度比较

[返回功能手册](README.md)

比较器在输出契约通过后逐元素计算数值误差。默认不忽略任何超差元素，支持多输出、整数精确比较、复数分量比较和明确的 NaN/Inf 政策。增加统计指标不会取消基础数值与契约检查。

## 基础参数与判定

| 参数 | 默认值 | 含义 |
|---|---|---|
| `atol` | `1e-5` | 绝对容差 |
| `rtol` | `1e-5` | 相对容差，以参考值绝对值为基准 |
| `exact` | 未启用 | 为 true 时要求严格相等；整数、布尔建议显式设置 |
| `nan` | `reject` | `equal` 仅允许对应位置都为 NaN |
| `inf` | `reject` | `same_sign` 仅允许对应位置为同符号无穷 |
| `max_mismatch_ratio` | `0` | 允许的超差元素比例，必须显式放宽才会忽略少量超差 |
| `max_failures` | `8` | 报告中保存的失败元素详情数量上限，不改变判定 |

有限值的容差公式为 `abs(actual-reference) <= atol + rtol*abs(reference)`。相对误差为 `abs(actual-reference)/abs(reference)`；参考为零时，零误差记 0，非零误差记 Infinity。复数分别检查实部、虚部，一个分量失败即判该元素失败；最大绝对和相对误差采用两个分量中的最大值。

整数严格相等比较不会先转换为浮点，因而可识别 int64/uint64 极值附近的差异。NaN/Inf 违反政策时始终失败，不能被 `max_mismatch_ratio` 放过。空输出没有数值样本，只能证明空输出契约与对应检查满足，不能据此推断非空输入精度。

## 逐输出政策

政策按以下顺序覆盖：工具默认值 → 根对象 `defaults.compare` → 用例 `compare` → `compare_by_dtype` 中当前输出 dtype 的政策 → 当前输出 `compare`。`metrics` 子对象按指标名称合并。

以下是配置片段，不是完整配置：

```json
{
  "compare": {"atol": 0.00001, "rtol": 0.00001},
  "compare_by_dtype": {
    "float16": {"atol": 0.001, "rtol": 0.001},
    "int64": {"exact": true}
  },
  "outputs": [
    {"shape": [4], "dtype": "float16"},
    {"shape": [4], "dtype": "int64", "compare": {"exact": true}}
  ]
}
```

版本 1 仍采用原有比较行为；要使用逐输出和新增指标请声明配置协议版本 2。容差应根据算子误差预算设置，不能仅为消除失败而放宽。

## 额外指标

| `metrics` 参数 | 判定 | 定义 |
|---|---|---|
| `rmse_max` | RMSE ≤ 阈值 | 有限值对误差模长平方均值的平方根 |
| `cosine_min` | 余弦 ≥ 阈值 | 实数使用普通余弦；复数使用 Hermitian 内积实部的余弦 |
| `ulp_max` | 最大 ULP ≤ 阈值 | 参考先舍入到输出 dtype，再比较有序 IEEE 编码距离 |

两个零向量的余弦为 1，只有一边是零向量时为 0。ULP 仅用于 float16、bfloat16、float32、float64，正负零距离为 0。指标仅在显式给出阈值时参与判定；所有指标与基础判定取“同时满足”。没有有限数值对时指标为 null；非空输出不能把未测出的指标说成达标。

## 使用示例

[core-comparison.json](examples/core-comparison.json) 包含浮点指标、int64 极值和复数分量三个 CPU 用例：

```bash
python -m musaopst validate docs/manual/examples/core-comparison.json
python -m musaopst run docs/manual/examples/core-comparison.json --output runs/manual-comparison
```

预期 3 个 CPU `PASS`。报告的 `comparison.outputs` 会给出 mismatch_count、mismatch_ratio、最大绝对/相对误差、失败索引，以及显式指标的阈值和结果。安全的错误数值用例见 [预期失败](expected-failures.md)。

## 自定义比较器

用 `"custom": "my_compare:check"` 指定扩展比较器。函数签名为 `check(actual, reference, contract, policy)`，收到独立副本，应返回包含布尔 `passed` 的 JSON 可序列化字典。例如以下函数模板额外限制均值偏差：

```python
import numpy as np

def check(actual, reference, contract, policy):
    bias = float(np.mean(actual.astype(np.float64) - reference.astype(np.float64))) if actual.size else 0.0
    return {"passed": abs(bias) <= 1e-4, "mean_bias": bias}
```

自定义函数的通过结果无法覆盖 shape、基础容差或 NaN/Inf 失败；函数抛异常、返回格式错误也会比较失败。不要用 `.all()` 的两个布尔值相等来代替逐元素数值比较。

无需重新执行设备即可调整政策检查已有快照，详见 [离线重比较](recompare.md)。比较器实现见 [compare.py](../../musaopst/compare.py)。
