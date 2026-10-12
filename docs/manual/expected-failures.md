# 预期失败

[返回功能手册](README.md)

预期失败用于验证测试器确实抓住错误，或确认算子在不支持的输入上给出约定异常。匹配必须同时考虑实际状态、发生阶段和异常类型；不能把任意失败都当作负向用例成功。

## 配置参数

`expect` 位于用例对象中：

| 字段 | 要求 |
|---|---|
| `status` | 必填；指定预期的实际状态，如 `ACCURACY_FAILED` |
| `stage` | 预期失败时必填，且必须与报告中的 `last_stage` 相同 |
| `exception` | 预期失败时必填，异常类型名称必须精确匹配 |
| `message_contains` | 可选，要求错误消息包含指定子串 |

允许的状态为 `PASS`、`ACCURACY_FAILED`、`BUILD_FAILED`、`EXECUTION_FAILED`、`REFERENCE_FAILED`、`TIMEOUT`、`CONFIG_ERROR`、`UNSUPPORTED`。失败阶段可为 `inputs/build/execute/reference/compare/timeout/repeat/benchmark/post_benchmark_compare`，需按实际发生位置填写。

环境故障、清理故障、严重设备故障、跳过以及 GPU 超时不能被普通负向政策吞掉。用于测试超时路径时应采用受控 CPU worker，不应故意运行可能损坏设备上下文的 kernel。配置文件无法解析或整体展开失败发生在用例运行之前，也不能用 `expect` 让配置绕过校验。

## 使用示例

[core-expected-failures.json](examples/core-expected-failures.json) 包含一个正确用例、一个错误数值用例和一个参考异常用例：

```bash
python -m musaopst validate docs/manual/examples/core-expected-failures.json
python -m musaopst run docs/manual/examples/core-expected-failures.json --output runs/manual-expected-failures
```

预期汇总为 `PASS: 1`、`EXPECTED_FAILURE: 2`，命令退出码为 0，表示测试期望满足。两个负向记录不会计入精度通过。

错误数值用例的政策片段：

```json
{
  "expect": {
    "status": "ACCURACY_FAILED",
    "stage": "compare",
    "exception": "ComparisonFailure"
  }
}
```

参考函数异常的政策片段：

```json
{
  "expect": {
    "status": "REFERENCE_FAILED",
    "stage": "reference",
    "exception": "RuntimeError",
    "message_contains": "controlled offline fixture failure"
  }
}
```

在 MUSA 环境中可以运行只扰动正常返回值的安全例子：

```bash
python -m musaopst run examples/suites/contracts_matrix.json --case torch_safe_value_negative --device 0 --output runs/manual-expected-failures-gpu
```

它不运行越界或死锁 kernel，期望原始精度状态失败并匹配声明。

## 如何读结果

符合预期的负向记录使用 `status: "EXPECTED_FAILURE"`，同时保留 `observed_status`、`last_stage`、`exception_type`、`expectation`、`expectation_met`。检查 `observed_status` 才能看到实际失败原因。

若原本预期失败却实际成功，会得到 `EXPECTATION_FAILED`。若失败了但阶段或类型不符，则保留实际失败状态，并标记期望未满足。把示例的 `exception` 改成不符的名称即可安全检验该路径，不需要制造 GPU 异常。

只有负向用例的测试集可以在期望全部匹配时返回 0，但精度通过数量仍为 0，不能称为算子精度通过。全部跳过的测试集不会满足整体通过条件。CI 需要同时看退出码、精度通过数量和覆盖状态，详见 [报告](reports.md)。
